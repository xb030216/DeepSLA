#!/usr/bin/env python3
"""
token_line_features_longcode.py

Long-context aware token+line feature extractor.
Supports arbitrarily long source files by splitting into character-aligned segments,
ensuring each segment token length <= segment_max_tokens, and then concatenating
token-level hidden states and offsets into a global sequence for line-level pooling.

Usage: similar args to original token_line_features.py, plus:
  --segment_max_tokens     : max tokens per segment (default tokenizer.model_max_length)
  --segment_stride_tokens  : overlap tokens between consecutive segments (default 256)
  --max_chars_per_segment  : hard cap on chars to search per segment (safety)

Notes:
  - This implementation does a binary-search on char_end per segment to find the largest
    character slice that tokenizes to <= segment_max_tokens tokens. It's robust but can be slower.
  - We always compute and store global char offsets for tokens so that token->line mapping
    uses a consistent global char coordinate system.
  - Uses process_sample_forward(model, device, input_ids, attention_mask, num_last_layers)
    from original script to fetch hidden states.
"""

import os
import time
import json
import argparse
import logging
import tempfile
import gc
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig

# ---------------- logging ----------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("token_line_features_longcode")

# ---------------- try to import user's LineLevelDataset if available ----------------
try:
    from datasets.line_level_dataset import LineLevelDataset
    _HAS_LINELEVEL = True
    logger.info("Found project datasets.line_level_dataset -> will prefer it for accurate line spans.")
except Exception:
    LineLevelDataset = None
    _HAS_LINELEVEL = False
    logger.info("datasets.line_level_dataset not found; using built-in fallback dataset.")

# ---------------- Fallback dataset ----------------
class FallbackCSVChunkDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 2048):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _parse_labels(self, x):
        if x is None or x == "":
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                if x.strip().startswith("["):
                    parsed = eval(x)
                    if isinstance(parsed, (list, tuple)):
                        return [int(y) for y in parsed]
                parts = [p.strip() for p in x.split(",") if p.strip() != ""]
                return [int(p) for p in parts if p.isdigit()]
            except Exception:
                return []
        return []

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = row.get("buggyCode", "")
        labels = self._parse_labels(row.get("bugLineNum", ""))
        return {"code": str(code), "labels": labels, "idx": int(idx)}


# ---------------- utils ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_save_torch(obj: Any, path: str, compress: bool = True):
    torch.save(obj, path, _use_new_zipfile_serialization=compress)

def write_json_atomic(obj: Any, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def parse_line_offsets(code: str) -> List[Tuple[int, int]]:
    lines = code.splitlines(keepends=True)
    spans = []
    pos = 0
    for ln in lines:
        start = pos
        end = pos + len(ln)
        spans.append((start, end))
        pos = end
    if len(spans) == 0:
        spans = [(0, 0)]
    return spans

def token_offsets_to_line_spans(token_offsets: List[Tuple[int,int]], line_spans_char: List[Tuple[int,int]]):
    T = len(token_offsets)
    L = len(line_spans_char)
    token_mid = []
    for (a,b) in token_offsets:
        if a is None or b is None:
            token_mid.append(None)
        else:
            token_mid.append((a + b) / 2.0)
    token_to_line = [-1] * T
    for tidx, mid in enumerate(token_mid):
        if mid is None:
            token_to_line[tidx] = -1
            continue
        assigned = False
        for li, (ls, le) in enumerate(line_spans_char):
            if ls <= mid < le:
                token_to_line[tidx] = li
                assigned = True
                break
        if not assigned:
            token_to_line[tidx] = max(0, L - 1)
    for i in range(T):
        if token_to_line[i] == -1:
            assigned = False
            for j in range(i - 1, -1, -1):
                if token_to_line[j] != -1:
                    token_to_line[i] = token_to_line[j]
                    assigned = True
                    break
            if not assigned:
                for j in range(i + 1, T):
                    if token_to_line[j] != -1:
                        token_to_line[i] = token_to_line[j]
                        assigned = True
                        break
            if not assigned:
                token_to_line[i] = 0
    line_token_indices = {i: [] for i in range(L)}
    for tidx, li in enumerate(token_to_line):
        if 0 <= li < L:
            line_token_indices[li].append(tidx)
    line_spans_tokens = []
    for i in range(L):
        toks = line_token_indices.get(i, [])
        if not toks:
            line_spans_tokens.append((0, 0))
        else:
            s = min(toks); e = max(toks) + 1
            line_spans_tokens.append((s, e))
    return line_spans_tokens

def quantize_tensor_int8(tensor: torch.Tensor):
    max_abs = float(tensor.abs().max().item())
    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / 127.0
    scaled = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8).cpu().numpy()
    return scaled, float(scale)

# ---------------- model forward helper ----------------
@torch.no_grad()
def process_sample_forward(model, model_device, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_last_layers: int):
    outputs = model(input_ids=input_ids.unsqueeze(0).to(model_device),
                    attention_mask=attention_mask.unsqueeze(0).to(model_device),
                    output_hidden_states=True,
                    return_dict=True)
    hidden_states = outputs.hidden_states
    last = hidden_states[-num_last_layers:]
    cpu_list = [hs.squeeze(0).detach().cpu() for hs in last]
    return cpu_list

# ---------------- segmentation helpers ----------------
def tokenize_with_offsets(tokenizer, text: str):
    # return input_ids (list), attention_mask (list), offsets (list of (s,e) ints or (0,0))
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    ids = enc.get("input_ids", [])
    attn = enc.get("attention_mask", [])
    offs = enc.get("offset_mapping", [])
    # ensure types
    input_ids = [int(x) for x in ids]
    attention_mask = [int(x) for x in attn]
    # offsets may contain tuples or lists
    offsets = []
    for o in offs:
        try:
            a, b = o
            if a is None or b is None:
                offsets.append((0,0))
            else:
                offsets.append((int(a), int(b)))
        except Exception:
            offsets.append((0,0))
    return input_ids, attention_mask, offsets

def find_max_char_end_for_segment(code: str, start_char: int, tokenizer, segment_max_tokens: int, max_chars_limit: int=20000):
    """
    Find the largest char_end (>= start_char+1) such that tokenizer(code[start_char:char_end])
    produces <= segment_max_tokens tokens (including special tokens).
    Uses exponential expansion then binary search on char_end.
    max_chars_limit: safety cap to avoid scanning extremely large char ranges indefinitely.
    Returns char_end (exclusive) and the tokenization result for that slice.
    """
    n = len(code)
    if start_char >= n:
        return start_char, ([], [], [])
    # quick lower bound
    lo = start_char + 1
    hi = min(n, start_char + min(max_chars_limit, max(1000, segment_max_tokens * 8)))
    # exponential expand hi until token_count > segment_max_tokens or end reached
    last_ok = None
    # helper to get token length quickly using tokenizer.encode (but we want offsets too sometimes)
    def token_len_for_slice(s, e):
        slice_text = code[s:e]
        enc = tokenizer(slice_text, return_offsets_mapping=True, add_special_tokens=True)
        ids = enc.get("input_ids", [])
        return len(ids), enc

    # try initial hi until overflow
    step = 0
    while True:
        step += 1
        tl, enc = token_len_for_slice(start_char, hi)
        if tl <= segment_max_tokens:
            last_ok = (hi, enc)
            if hi >= n:
                # reached end
                return n, (enc.get("input_ids", []), enc.get("attention_mask", []), enc.get("offset_mapping", []))
            # expand
            span = hi - start_char
            # double span but cap to n and max_chars_limit
            new_span = min(n - start_char, span * 2)
            if new_span == span:
                # can't expand
                break
            hi = min(n, start_char + new_span, start_char + max_chars_limit)
            if hi == start_char + span:
                break
            if step > 30:
                break
            continue
        else:
            # hi token_len > limit, need to binary search between lo..hi
            break

    # if last_ok is None, it means even small hi produced >limit; we must shrink hi towards lo
    lo = start_char + 1
    # Binary search on char_end in [lo, hi]
    best_end = start_char + 1
    best_enc = None
    L = lo
    R = hi
    while L <= R:
        mid = (L + R) // 2
        tl, enc = token_len_for_slice(start_char, mid)
        if tl <= segment_max_tokens:
            best_end = mid
            best_enc = enc
            L = mid + 1
        else:
            R = mid - 1
    if best_enc is None:
        # fallback: take minimal 1-char slice tokenized (should rarely happen)
        enc = tokenizer(code[start_char:start_char+1], return_offsets_mapping=True, add_special_tokens=True)
        return min(n, start_char+1), (enc.get("input_ids", []), enc.get("attention_mask", []), enc.get("offset_mapping", []))
    # return best_end and the encoding for that slice
    return best_end, (best_enc.get("input_ids", []), best_enc.get("attention_mask", []), best_enc.get("offset_mapping", []))

def compute_segments_for_code(code: str, tokenizer, segment_max_tokens: int, segment_stride_tokens: int, max_chars_per_segment: int=20000):
    """
    Produce a list of segments; each segment is a dict:
      {"char_start": int, "char_end": int, "input_ids": [...], "attention_mask": [...], "offsets": [...]}.
    Segments are overlapping in tokens by segment_stride_tokens.
    """
    n = len(code)
    segments = []
    char_pos = 0
    # If code is empty
    if n == 0:
        return segments

    while char_pos < n:
        char_end, enc = find_max_char_end_for_segment(code, char_pos, tokenizer, segment_max_tokens, max_chars_per_segment)
        input_ids, attn_mask, offsets = enc
        # offsets are relative to the segment's text (0..len(segment_text))
        segment = {
            "char_start": char_pos,
            "char_end": char_end,
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "offsets": offsets
        }
        segments.append(segment)

        # determine next char_pos by mapping segment token index (end_token - segment_stride_tokens) to char offset
        T = len(input_ids)
        if segment_stride_tokens <= 0 or segment_stride_tokens >= T:
            # move to char_end (no overlap or stride >= tokens)
            next_char_pos = char_end
        else:
            # find token index to start next window such that overlap = segment_stride_tokens
            start_token_idx = max(0, T - segment_stride_tokens)
            # offsets may contain (0,0) for special tokens; find first meaningful token from start_token_idx
            chosen_token_idx = None
            for ti in range(start_token_idx, T):
                try:
                    off = offsets[ti]
                    a, b = off
                    if a is None or b is None or (a == 0 and b == 0 and ti == 0):
                        continue
                    chosen_token_idx = ti
                    break
                except Exception:
                    continue
            if chosen_token_idx is None:
                # fallback move to char_end
                next_char_pos = char_end
            else:
                off = offsets[chosen_token_idx]
                token_char_relative = off[0] if off[0] is not None else 0
                next_char_pos = segments[-1]["char_start"] + int(token_char_relative)
                # ensure progress
                if next_char_pos <= char_pos:
                    next_char_pos = char_end
        if next_char_pos <= char_pos:
            # ensure forward progress
            next_char_pos = char_end
        char_pos = next_char_pos
        # safety: avoid infinite loop
        if char_pos == n:
            break
    return segments


def extract_chunk_long(
    model,
    tokenizer,
    dataset_obj,
    model_name: str,
    save_line_mean: bool = True,
    num_last_layers: int = 4,
    keep_token_hidden: bool = True,
    save_layerwise: bool = False,
    quantize_int8: bool = False,
    max_samples: int = -1,
    progress: bool = True,
    segment_max_tokens: int = 2048,
    segment_stride_tokens: int = 256,
    max_chars_per_segment: int = 20000
):
    """
    Strict NO-OFFSET ablation extract_chunk_long:
      - DOES NOT use token offsets or segment concatenation.
      - For each sample, splits raw_code into lines via splitlines(),
        then for each line performs tokenizer -> model forward -> multi-layer fusion.
      - Keeps labels aligned to raw_lines (1-based indices expected in parsed_labels).
      - Returns sample_record entries similar to original script, but:
          - token_hidden and token_offsets are set to None/[] (not used)
          - line_spans provided as character spans (parse_line_offsets) for reference (not token spans)
    """
    model.eval()
    samples = []
    loader = DataLoader(dataset_obj, batch_size=1, shuffle=False)
    model_device = next(model.parameters()).device

    it = enumerate(loader)
    if progress:
        it = tqdm(it, total=len(loader), desc="Extracting samples (NO-OFFSET)", unit="samp")

    processed = 0
    for idx, item in it:
        if 0 <= max_samples <= processed:
            break
        processed += 1
        try:
            # Get raw code and labels
            if _HAS_LINELEVEL and isinstance(dataset_obj, LineLevelDataset):
                try:
                    row = dataset_obj.rows[idx]
                    raw_code = row.get("buggyCode", "")
                    parsed_labels = row.get("bugLineNum", [])
                except Exception:
                    raw_code = ""
                    parsed_labels = []
            else:
                code = item["code"] if not isinstance(item["code"], (list, tuple)) else item["code"][0]
                raw_code = code
                parsed_labels = item.get("labels", [])

            # Build character-level line spans for reference (char_start, char_end)
            # NOTE: for NO-OFFSET ablation we will NOT use token offsets; this is just metadata.
            line_char_spans = parse_line_offsets(raw_code)  # list of (s,e)

            # Strict No-Offset: split by lines as the single source of truth
            raw_lines = raw_code.splitlines()
            num_lines = len(raw_lines)

            # Prepare per-line vectors (may be expensive because one forward per line)
            line_vecs = []
            stacked_layerwise_all = []  # optional store for layerwise (if requested)
            # H will be inferred from first non-empty line's fused vector
            H = None

            for line_text in raw_lines:
                # If the line is empty (or only whitespace) reserve a zero placeholder
                if line_text.strip() == "":
                    line_vecs.append(None)
                    if save_layerwise:
                        stacked_layerwise_all.append(None)
                    continue

                # Tokenize and forward this single line
                enc = tokenizer(line_text, return_tensors="pt", truncation=True, add_special_tokens=True)
                input_ids = enc["input_ids"].squeeze(0).to(model_device)
                attention_mask = enc["attention_mask"].squeeze(0).to(model_device)

                # forward (process_sample_forward expects input_ids.unsqueeze(0) internally)
                last_layers = process_sample_forward(model, model_device, input_ids, attention_mask, num_last_layers)
                # last_layers: list of tensors, each (T_line, H) on CPU

                # Stack last layers -> (N, T, H) then fuse along layers -> (T, H)
                stacked = torch.stack(last_layers, dim=0)  # N, T_line, H
                fused_tokens = stacked.mean(dim=0)  # T_line, H

                # Pool tokens into one line vector. Use mean pooling (consistent with baseline_model).
                fused_line = fused_tokens.mean(dim=0)  # H

                # Save CPU copy
                line_vecs.append(fused_line.cpu())

                if save_layerwise:
                    # Keep stacked (N, T, H) on CPU for potential diagnostics (may be large)
                    stacked_layerwise_all.append(stacked.cpu())
                else:
                    stacked_layerwise_all.append(None)

                # infer hidden dim if not set
                if H is None:
                    H = fused_line.size(0)

                # free intermediate cpu tensors from last_layers if any
                del last_layers
                torch.cuda.empty_cache()
                gc.collect()

            # If all lines were empty, define H := 0
            if H is None:
                H = 0

            # Replace None placeholders with zero vectors of size H
            if H == 0:
                # empty or all-empty file; set shape (num_lines, 0)
                if len(line_vecs) > 0:
                    line_features_tensor = torch.zeros((num_lines, 0))
                else:
                    line_features_tensor = torch.zeros((0, 0))
            else:
                filled = [(torch.zeros(H) if v is None else v) for v in line_vecs]
                line_features_tensor = torch.stack(filled, dim=0)  # (num_lines, H)

            # Build labels vector strictly aligned to raw_lines (1-based line numbers in parsed_labels)
            labels_vec = torch.zeros((num_lines,), dtype=torch.uint8)
            try:
                # parsed_labels may be string, list, etc.
                if isinstance(parsed_labels, str):
                    # attempt to parse simple formats like "[1,2]" or "1,2"
                    try:
                        if parsed_labels.strip().startswith("["):
                            parsed = eval(parsed_labels)
                            parsed_labels = parsed
                        else:
                            parsed_labels = [int(x) for x in parsed_labels.split(",") if x.strip().isdigit()]
                    except Exception:
                        parsed_labels = []
                if isinstance(parsed_labels, (list, tuple)):
                    for ln in parsed_labels:
                        try:
                            li = int(ln)
                            if 1 <= li <= num_lines:
                                labels_vec[li - 1] = 1
                        except Exception:
                            continue
                else:
                    # single numeric
                    try:
                        li = int(parsed_labels)
                        if 1 <= li <= num_lines:
                            labels_vec[li - 1] = 1
                    except Exception:
                        pass
            except Exception:
                # be robust to any parsing issues
                pass

            # Prepare sample_record similar to original schema:
            # Keep token_* fields empty / None since we do not compute them in NO-OFFSET mode.
            # For line_spans we store the character-level spans (for reference), not token spans.
            sample_record = {
                "num_tokens": 0,  # no global tokens concatenated in NO-OFFSET
                "hidden_dim": int(H) if H else None,
                "line_spans": line_char_spans,      # char-level spans (metadata)
                "token_offsets": [],                # empty (not used)
                "line_labels": labels_vec,          # aligned to raw_lines
                "num_lines": int(num_lines),
                "meta": {"model_name": model_name, "orig_idx": int(idx), "ablation": "NO_OFFSET"}
            }

            # Optionally save token_hidden (we didn't compute any), and layerwise if requested
            if keep_token_hidden:
                sample_record["token_hidden"] = None
            else:
                sample_record["token_hidden"] = None

            if save_layerwise:
                # Save layerwise per-line stacks (this can be very large). Keep as list (N, T, H) or None.
                # Convert None entries to empty tensors to keep consistent types if desired.
                sample_record["token_hidden_layers_per_line"] = stacked_layerwise_all
            # Save line features (half precision to save space, consistent with original)
            if save_line_mean:
                sample_record["line_features"] = line_features_tensor.half().cpu()
            else:
                sample_record["line_features"] = None

            samples.append(sample_record)

            # cleanup
            del line_vecs, filled, line_features_tensor
            if save_layerwise:
                del stacked_layerwise_all
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as re:
            logger.exception("RuntimeError during sample processing: %s", re)
            torch.cuda.empty_cache()
            gc.collect()
            samples.append({"error": f"runtime_error: {str(re)}", "meta": {"orig_idx": int(idx)}})
            continue
        except Exception as e:
            logger.exception("Exception during sample processing: %s", e)
            torch.cuda.empty_cache()
            gc.collect()
            samples.append({"error": f"exception: {str(e)}", "meta": {"orig_idx": int(idx)}})
            continue

    return samples

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser(description="Extract token+line features for adapter training (long-context aware)")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with buggyCode and bugLineNum")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save chunked features")
    parser.add_argument("--chunk_size", type=int, default=400, help="Rows per chunk")
    parser.add_argument("--max_length", type=int, default=2048, help="(deprecated) tokenizer max length hint")
    parser.add_argument("--num_last_layers", type=int, default=4, help="Number of last layers to fuse")
    parser.add_argument("--save_layerwise", action="store_true", help="Also save last N layers (very large)")
    parser.add_argument("--keep_token_hidden", action="store_true", default=True, help="Save token-level hidden states (default True)")
    parser.add_argument("--quantize_int8", action="store_true", help="Quantize fused token_hidden to int8 (saves disk but is lossy)")
    parser.add_argument("--save_line_mean", action="store_true", default=True, help="Save per-line mean features")
    parser.add_argument("--compress", action="store_true", help="Use torch zipfile serialization (compressed)")
    parser.add_argument("--start_chunk", type=int, default=0, help="Resume from this chunk index")
    parser.add_argument("--chunks_limit", type=int, default=-1, help="If >0, stop after this many chunks")
    parser.add_argument("--device", type=str, default=None, help="Preferred device (e.g., cuda). If None, use model device_map default")
    parser.add_argument("--use_8bit", action="store_true", help="Load model in 8-bit (bitsandbytes) if supported")
    parser.add_argument("--max_samples", type=int, default=-1, help="For debugging: process at most this many samples overall")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress")
    parser.add_argument("--debug", action="store_true", help="Enable debug verbosity")
    # long-context specific
    parser.add_argument("--segment_max_tokens", type=int, default=None, help="Max tokens per segment (default tokenizer.model_max_length)")
    parser.add_argument("--segment_stride_tokens", type=int, default=256, help="Overlap tokens between segments")
    parser.add_argument("--max_chars_per_segment", type=int, default=20000, help="Safety cap on chars scanned per segment")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    ensure_dir(args.save_dir)
    manifest_path = os.path.join(args.save_dir, "manifest.json")
    stats_path = os.path.join(args.save_dir, "extract_stats.json")
    failed_path = os.path.join(args.save_dir, "failed_samples.json")

    logger.info(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = None
    load_kwargs = {"low_cpu_mem_usage": True}
    try:
        load_kwargs["torch_dtype"] = torch.float16
    except Exception:
        pass
    if args.use_8bit:
        load_kwargs["load_in_8bit"] = True
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map="auto", **load_kwargs)
        logger.info("Loaded model via AutoModelForCausalLM (device_map=auto).")
    except Exception as e:
        logger.warning("AutoModelForCausalLM load failed: %s. Falling back to AutoModel.", e)
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, device_map="auto", **load_kwargs)
        logger.info("Loaded model via AutoModel (device_map=auto).")

    model.eval()

    # default segment_max_tokens fallback to tokenizer.model_max_length if not provided
    if args.segment_max_tokens is None:
        seg_max = getattr(tokenizer, "model_max_length", 2048)
    else:
        seg_max = args.segment_max_tokens

    reader = pd.read_csv(args.csv_path, chunksize=args.chunk_size, iterator=True, dtype=str, keep_default_na=False)
    stats = []
    failed_samples = []
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            logger.info(f"Loaded existing manifest with {len(manifest)} chunk entries.")
        except Exception:
            manifest = {}

    for i, chunk_df in enumerate(reader):
        if i < args.start_chunk:
            logger.info(f"Skipping chunk {i} (start_chunk={args.start_chunk})")
            continue
        if args.chunks_limit > 0 and (i - args.start_chunk) >= args.chunks_limit:
            logger.info("Reached chunks_limit, stopping.")
            break

        chunk_file = os.path.join(args.save_dir, f"chunk_{i:04d}.pt")
        if os.path.exists(chunk_file):
            logger.info(f"Chunk file exists: {chunk_file} -> skipping (resume).")
            continue

        logger.info(f"Processing chunk {i} with {len(chunk_df)} rows")
        t0 = time.time()
        if _HAS_LINELEVEL:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmpf:
                tmpname = tmpf.name
                chunk_df.to_csv(tmpname, index=False)
            try:
                dlset = LineLevelDataset(tmpname, tokenizer, max_length=seg_max)
                logger.debug("Using project LineLevelDataset for accurate mapping.")
            except Exception as e:
                logger.warning("Failed to build LineLevelDataset from temp CSV: %s -> using fallback dataset", e)
                dlset = FallbackCSVChunkDataset(chunk_df, tokenizer, max_length=seg_max)
            finally:
                try:
                    os.remove(tmpname)
                except Exception:
                    pass
        else:
            dlset = FallbackCSVChunkDataset(chunk_df, tokenizer, max_length=seg_max)

        try:
            samples = extract_chunk_long(
                model=model,
                tokenizer=tokenizer,
                dataset_obj=dlset,
                model_name=args.model_name,
                save_line_mean=args.save_line_mean,
                num_last_layers=args.num_last_layers,
                keep_token_hidden=args.keep_token_hidden,
                save_layerwise=args.save_layerwise,
                quantize_int8=args.quantize_int8,
                max_samples=args.max_samples,
                progress=args.progress,
                segment_max_tokens=seg_max,
                segment_stride_tokens=args.segment_stride_tokens,
                max_chars_per_segment=args.max_chars_per_segment
            )
        except Exception as e:
            logger.exception("Extraction failed for chunk %d: %s", i, e)
            failed_samples.append({"chunk_idx": i, "error": str(e)})
            torch.cuda.empty_cache()
            gc.collect()
            continue

        num_saved = len(samples)
        manifest_entry = {
            "chunk_file": os.path.basename(chunk_file),
            "num_samples": num_saved,
            "hidden_dim": samples[0]["hidden_dim"] if num_saved > 0 and "hidden_dim" in samples[0] else None,
            "num_last_layers": args.num_last_layers,
            "save_layerwise": bool(args.save_layerwise),
            "keep_token_hidden": bool(args.keep_token_hidden),
            "quantize_int8": bool(args.quantize_int8),
            "segment_max_tokens": seg_max,
            "segment_stride_tokens": int(args.segment_stride_tokens),
            "timestamp": time.time()
        }
        manifest[os.path.basename(chunk_file)] = manifest_entry

        try:
            safe_save_torch(samples, chunk_file, compress=args.compress)
        except Exception as e:
            logger.exception("Failed to save chunk file %s: %s", chunk_file, e)
            try:
                torch.save(samples, chunk_file)
            except Exception as e2:
                logger.exception("Fallback save also failed: %s", e2)
                failed_samples.append({"chunk_idx": i, "error": f"save_failed: {str(e2)}"})
                continue

        t_chunk = time.time() - t0
        peak = 0.0
        try:
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            peak = 0.0
        stats.append({
            "chunk_idx": i,
            "rows_in_chunk": int(len(chunk_df)),
            "samples_saved": int(num_saved),
            "saved_path": os.path.basename(chunk_file),
            "time_s": round(t_chunk, 3),
            "gpu_peak_GB": round(peak, 3)
        })
        logger.info(f"Saved chunk_{i:04d}.pt ({num_saved} samples) | time {t_chunk:.2f}s | peak GPU {peak:.2f}GB")

        try:
            write_json_atomic(manifest, manifest_path)
        except Exception as e:
            logger.warning("Failed to write manifest: %s", e)
        try:
            write_json_atomic(stats, stats_path)
        except Exception:
            pass
        if failed_samples:
            try:
                write_json_atomic(failed_samples, failed_path)
            except Exception:
                pass

        torch.cuda.empty_cache()
        gc.collect()

    logger.info("All requested chunks processed. Writing final manifest/stats.")
    try:
        write_json_atomic(manifest, manifest_path)
    except Exception as e:
        logger.warning("Failed to write final manifest: %s", e)
    try:
        write_json_atomic(stats, stats_path)
    except Exception as e:
        logger.warning("Failed to write extract stats: %s", e)
    if failed_samples:
        try:
            write_json_atomic(failed_samples, failed_path)
            logger.info(f"Failed samples (if any) written to {failed_path}")
        except Exception:
            pass

    logger.info("Extraction complete.")


if __name__ == "__main__":
    main()
