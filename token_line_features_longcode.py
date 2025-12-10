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
    # 注意：这里取了最后的几层   middle的设置为last = hidden_states[12:20]，可能实际上是12-19层
    # last = hidden_states[-num_last_layers:]
    last = hidden_states[12:20]
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

# ---------------- core extraction adapted for long files ----------------
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
    model.eval()
    samples = []
    loader = DataLoader(dataset_obj, batch_size=1, shuffle=False)
    model_device = next(model.parameters()).device

    it = enumerate(loader)
    if progress:
        it = tqdm(it, total=len(loader), desc="Extracting samples", unit="samp")

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

            # compute global line char spans
            line_char_spans = parse_line_offsets(raw_code)

            # compute segments (each with local offsets)
            segments = compute_segments_for_code(raw_code, tokenizer,
                                                 segment_max_tokens=segment_max_tokens,
                                                 segment_stride_tokens=segment_stride_tokens,
                                                 max_chars_per_segment=max_chars_per_segment)
            if len(segments) == 0:
                # empty code
                sample_record = {
                    "num_tokens": 0,
                    "hidden_dim": None,
                    "line_spans": [],
                    "token_offsets": [],
                    "line_labels": torch.zeros((len(line_char_spans),), dtype=torch.uint8),
                    "num_lines": len(line_char_spans),
                    "meta": {"model_name": model_name, "orig_idx": int(idx)}
                }
                samples.append(sample_record)
                continue

            # For each segment, we need to compute hidden states for the whole segment text.
            # We already tokenized when computing segments (tokenizer returned offsets, ids)
            # But we need tensors for model forward (with special tokens); using tokenizer(...) again
            # per segment with truncation=False would be fine but slower; we'll re-tokenize segment_text.
            token_hidden_list = []
            token_offsets_global = []  # list of (char_s_global, char_e_global) per token, including special tokens
            token_counts = []
            stacked_layerwise = []  # if save_layerwise

            # Keep track of total token count for concatenation in CPU
            for seg in segments:
                seg_text = raw_code[seg["char_start"]:seg["char_end"]]
                # tokenize with offsets and without truncation, include special tokens
                enc = tokenizer(seg_text, truncation=False, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)
                offsets = enc.get("offset_mapping")
                if offsets is None:
                    offsets = [(0,0)] * input_ids.size(0)
                else:
                    offsets = offsets.squeeze(0).tolist()

                # offsets are relative to seg_text; convert to global char coords
                global_offsets = []
                for (a,b) in offsets:
                    if a is None or b is None:
                        global_offsets.append((0,0))
                    else:
                        ga = int(a) + int(seg["char_start"])
                        gb = int(b) + int(seg["char_start"])
                        global_offsets.append((ga, gb))

                # forward pass for this segment (on GPU)
                try:
                    last_layers = process_sample_forward(model, model_device, input_ids, attention_mask, num_last_layers)
                except RuntimeError as re:
                    logger.exception("RuntimeError during forward for segment: %s", re)
                    # try to reduce batch/force gc and retry once
                    torch.cuda.empty_cache()
                    gc.collect()
                    last_layers = process_sample_forward(model, model_device, input_ids, attention_mask, num_last_layers)

                # fuse last layers => fused tokens for this segment (T_seg, H)
                if save_layerwise:
                    stacked = torch.stack(last_layers, dim=0)  # (N, T_seg, H) on CPU
                    fused = stacked.mean(dim=0)  # (T_seg, H)
                else:
                    fused = torch.stack(last_layers, dim=0).mean(dim=0)

                # Append
                token_hidden_list.append(fused)  # CPU tensor
                token_offsets_global.extend(global_offsets)
                token_counts.append(fused.size(0))
                if save_layerwise:
                    # stacked is CPU already from process_sample_forward
                    stacked_layerwise.append(stacked)

                # free GPU caches
                del last_layers
                torch.cuda.empty_cache()
                gc.collect()

            # Concatenate token_hidden across segments to form full sequence
            if len(token_hidden_list) == 0:
                fused_full = torch.zeros((0, 0))
            else:
                fused_full = torch.cat(token_hidden_list, dim=0)  # (T_full, H)
            T_full = fused_full.size(0)
            H = fused_full.size(1) if T_full > 0 else 0

            # compute token_offsets list (already global) - ensure int pairs
            token_offsets_list = [(int(a), int(b)) if (isinstance(a, int) and isinstance(b, int)) else (0,0) for (a,b) in token_offsets_global]

            # compute line->token spans using global token offsets and global line char spans
            line_token_spans = token_offsets_to_line_spans(token_offsets_list, line_char_spans)

            # compute per-line features by pooling fused tokens
            line_vecs = []
            for (ts, te) in line_token_spans:
                if te <= ts:
                    line_vecs.append(torch.zeros(H, dtype=fused_full.dtype))
                else:
                    vec = fused_full[ts:te, :].mean(dim=0)
                    # 求和
                    # vec = fused_full[ts:te, :].sum(dim=0)
                    # 最大
                    # vec = fused_full[ts:te, :].max(dim=0)[0]
                    # 标准差
                    # vec = fused_full[ts:te, :].std(dim=0)
                    # 中位数
                    # vec = fused_full[ts:te, :].median(dim=0)[0]
                    # 范数
                    # vec_norm = fused_full[ts:te, :].norm(dim=0, p=2)  # L2范数
                    # LLMAO 行内最后一个token的索引为 te-1（因为te是结束索引，不含） 直接取最后一个token的隐藏状态作为行级特征
                    # last_token_idx = te - 1
                    # vec = fused_full[last_token_idx, :]  # 形状 (H,)，与原格式一致
                    line_vecs.append(vec)
            if len(line_vecs) > 0:
                line_features_tensor = torch.stack(line_vecs, dim=0)
            else:
                line_features_tensor = torch.zeros((0, H), dtype=fused_full.dtype)

            # build labels vector
            L = len(line_token_spans)
            labels_vec = torch.zeros((L,), dtype=torch.uint8)
            try:
                for ln in parsed_labels:
                    if isinstance(ln, (list, tuple)):
                        for x in ln:
                            if 1 <= int(x) <= L:
                                labels_vec[int(x) - 1] = 1
                    else:
                        xi = int(ln)
                        if 1 <= xi <= L:
                            labels_vec[xi - 1] = 1
            except Exception:
                pass

            # prepare sample record
            sample_record = {
                "num_tokens": int(T_full),
                "hidden_dim": int(H) if H else None,
                "line_spans": line_token_spans,
                "token_offsets": token_offsets_list,
                "line_labels": labels_vec,
                "num_lines": int(L),
                "meta": {}
            }

            # save token_hidden or quantized int8
            if keep_token_hidden:
                if quantize_int8 and not save_layerwise:
                    scaled_int8, scale = quantize_tensor_int8(fused_full)
                    sample_record["token_hidden_int8"] = scaled_int8
                    sample_record["token_hidden_int8_scale"] = float(scale)
                else:
                    sample_record["token_hidden"] = fused_full.half().cpu()
                if save_layerwise and len(stacked_layerwise) > 0:
                    # need to concatenate stacked_layerwise along token dim
                    stacked_concat = torch.cat(stacked_layerwise, dim=1)  # (N, T_full, H)
                    sample_record["token_hidden_layers"] = stacked_concat.half().cpu()
            else:
                sample_record["token_hidden"] = None

            if save_line_mean:
                sample_record["line_features"] = line_features_tensor.half().cpu()
            else:
                sample_record["line_features"] = None

            sample_record["meta"]["model_name"] = model_name
            sample_record["meta"]["orig_idx"] = int(idx)

            samples.append(sample_record)

            # free temporaries
            del fused_full, token_hidden_list
            if save_layerwise:
                del stacked_layerwise
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
