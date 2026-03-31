#!/usr/bin/env python3
"""
token_line_features_longcode_jsonl.py

JSONL-backed variant of token_line_features_longcode.py.
It preserves the original long-context extraction pipeline and only swaps the
input layer from CSV to JSONL.

Expected JSONL fields are compatible with files like:
  baseline_model/dataset_hf/val.jsonl

Preferred fields:
  - code: full source code string
  - line_labels: per-line binary labels, e.g. [0, 1, 0, ...]

Also supported as fallbacks:
  - lines: list of source lines (used when code is absent)
  - buggyCode: CSV-style raw code field
  - bugLineNum: CSV-style 1-based buggy line indices
"""

import argparse
import ast
import gc
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from token_line_features_longcode import (
    effective_args_dict,
    ensure_dir,
    extract_chunk_long,
    load_json_config,
    logger,
    safe_save_torch,
    write_json_atomic,
)


class JSONLChunkDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int = 2048):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def _parse_sequence(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if parts:
                return parts
        return []

    def _normalize_code(self, row: Dict[str, Any]) -> str:
        code = row.get("code")
        if code is None or code == "":
            code = row.get("buggyCode", "")
        if (code is None or code == "") and isinstance(row.get("lines"), list):
            code = "".join(str(line) if str(line).endswith("\n") else f"{line}\n" for line in row["lines"])
            code = code.rstrip("\n")
        return str(code or "")

    def _from_binary_line_labels(self, values: List[Any]) -> List[int]:
        bug_lines = []
        for idx, value in enumerate(values):
            try:
                flag = int(value)
            except Exception:
                flag = 1 if value else 0
            if flag == 1:
                bug_lines.append(idx + 1)
        return bug_lines

    def _from_direct_line_numbers(self, values: List[Any]) -> List[int]:
        bug_lines = []
        for value in values:
            try:
                bug_lines.append(int(value))
            except Exception:
                continue
        return bug_lines

    def _looks_like_binary_labels(self, values: List[Any]) -> bool:
        parsed = []
        for value in values:
            try:
                parsed.append(int(value))
            except Exception:
                if isinstance(value, bool):
                    parsed.append(int(value))
                else:
                    return False
        return all(v in (0, 1) for v in parsed)

    def _parse_bug_line_numbers(self, row: Dict[str, Any]) -> List[int]:
        line_labels = self._parse_sequence(row.get("line_labels"))
        if line_labels:
            if self._looks_like_binary_labels(line_labels):
                return self._from_binary_line_labels(line_labels)
            return self._from_direct_line_numbers(line_labels)

        bug_line_num = self._parse_sequence(row.get("bugLineNum"))
        return self._from_direct_line_numbers(bug_line_num)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "code": self._normalize_code(row),
            "labels": self._parse_bug_line_numbers(row),
            "idx": int(idx),
        }


def load_jsonl_rows(path: str, chunk_size: int) -> Iterable[List[Dict[str, Any]]]:
    chunk: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL line {line_no} in {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_no} in {path} is not a JSON object.")
            chunk.append(record)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def build_arg_parser(config_defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    defaults = config_defaults or {}
    input_default = defaults.get("jsonl_path", defaults.get("csv_path"))

    parser = argparse.ArgumentParser(
        description="Extract token+line features for adapter training from JSONL (long-context aware)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file. CLI args override config values.")
    parser.add_argument("--dump_config", type=str, default=None, help="Write the effective arguments to a JSON file, then continue.")
    parser.add_argument("--print_config", action="store_true", help="Print the effective configuration before running.")
    parser.add_argument("--model_name", type=str, default=defaults.get("model_name"), required=("model_name" not in defaults), help="HuggingFace model id")
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default=input_default,
        required=(input_default is None),
        help="JSONL with code and line_labels or bugLineNum",
    )
    parser.add_argument("--save_dir", type=str, default=defaults.get("save_dir"), required=("save_dir" not in defaults), help="Directory to save chunked features")
    parser.add_argument("--chunk_size", type=int, default=defaults.get("chunk_size", 400), help="Rows per chunk")
    parser.add_argument("--max_length", type=int, default=defaults.get("max_length", 2048), help="(deprecated) tokenizer max length hint")
    parser.add_argument("--num_last_layers", type=int, default=defaults.get("num_last_layers", 4), help="Number of last layers to fuse")
    parser.add_argument("--save_layerwise", action="store_true", default=bool(defaults.get("save_layerwise", False)), help="Also save last N layers (very large)")
    parser.add_argument("--keep_token_hidden", action="store_true", default=bool(defaults.get("keep_token_hidden", True)), help="Save token-level hidden states (default True)")
    parser.add_argument("--quantize_int8", action="store_true", default=bool(defaults.get("quantize_int8", False)), help="Quantize fused token_hidden to int8 (saves disk but is lossy)")
    parser.add_argument("--save_line_mean", action="store_true", default=bool(defaults.get("save_line_mean", True)), help="Save per-line mean features")
    parser.add_argument("--compress", action="store_true", default=bool(defaults.get("compress", False)), help="Use torch zipfile serialization (compressed)")
    parser.add_argument("--start_chunk", type=int, default=defaults.get("start_chunk", 0), help="Resume from this chunk index")
    parser.add_argument("--chunks_limit", type=int, default=defaults.get("chunks_limit", -1), help="If >0, stop after this many chunks")
    parser.add_argument("--device", type=str, default=defaults.get("device"), help="Preferred device (e.g., cuda). If None, use model device_map default")
    parser.add_argument("--use_8bit", action="store_true", default=bool(defaults.get("use_8bit", False)), help="Load model in 8-bit (bitsandbytes) if supported")
    parser.add_argument("--max_samples", type=int, default=defaults.get("max_samples", -1), help="For debugging: process at most this many samples overall")
    parser.add_argument("--progress", action="store_true", default=bool(defaults.get("progress", False)), help="Show tqdm progress")
    parser.add_argument("--debug", action="store_true", default=bool(defaults.get("debug", False)), help="Enable debug verbosity")
    parser.add_argument("--segment_max_tokens", type=int, default=defaults.get("segment_max_tokens"), help="Max tokens per segment (default tokenizer.model_max_length)")
    parser.add_argument("--segment_stride_tokens", type=int, default=defaults.get("segment_stride_tokens", 256), help="Overlap tokens between segments")
    parser.add_argument("--max_chars_per_segment", type=int, default=defaults.get("max_chars_per_segment", 20000), help="Safety cap on chars scanned per segment")
    return parser


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    config_defaults = {}
    if pre_args.config:
        config_defaults = load_json_config(pre_args.config)
        logger.info("Loaded config from %s", pre_args.config)

    parser = build_arg_parser(config_defaults)
    args = parser.parse_args()

    if args.debug:
        logger.setLevel("DEBUG")

    ensure_dir(args.save_dir)
    effective_config = effective_args_dict(args)
    effective_config_path = os.path.join(args.save_dir, "effective_config.json")

    if args.print_config:
        print(json.dumps(effective_config, indent=2, ensure_ascii=False))

    if args.dump_config:
        dump_parent = os.path.dirname(args.dump_config)
        if dump_parent:
            ensure_dir(dump_parent)
        write_json_atomic(effective_config, args.dump_config)
        logger.info("Wrote requested config snapshot to %s", args.dump_config)

    write_json_atomic(effective_config, effective_config_path)
    logger.info("Saved effective config to %s", effective_config_path)

    manifest_path = os.path.join(args.save_dir, "manifest.json")
    stats_path = os.path.join(args.save_dir, "extract_stats.json")
    failed_path = os.path.join(args.save_dir, "failed_samples.json")

    logger.info("Loading tokenizer and model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    load_kwargs = {"low_cpu_mem_usage": True}
    try:
        load_kwargs["torch_dtype"] = torch.float16
    except Exception:
        pass
    if args.use_8bit:
        load_kwargs["load_in_8bit"] = True

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            device_map="auto",
            **load_kwargs,
        )
        logger.info("Loaded model via AutoModelForCausalLM (device_map=auto).")
    except Exception as exc:
        logger.warning("AutoModelForCausalLM load failed: %s. Falling back to AutoModel.", exc)
        model = AutoModel.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            device_map="auto",
            **load_kwargs,
        )
        logger.info("Loaded model via AutoModel (device_map=auto).")

    model.eval()

    if args.segment_max_tokens is None:
        seg_max = getattr(tokenizer, "model_max_length", 2048)
    else:
        seg_max = args.segment_max_tokens

    stats = []
    failed_samples = []
    manifest = {}

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            logger.info("Loaded existing manifest with %d chunk entries.", len(manifest))
        except Exception:
            manifest = {}

    for i, chunk_rows in enumerate(load_jsonl_rows(args.jsonl_path, args.chunk_size)):
        if i < args.start_chunk:
            logger.info("Skipping chunk %d (start_chunk=%d)", i, args.start_chunk)
            continue
        if args.chunks_limit > 0 and (i - args.start_chunk) >= args.chunks_limit:
            logger.info("Reached chunks_limit, stopping.")
            break

        chunk_file = os.path.join(args.save_dir, f"chunk_{i:04d}.pt")
        if os.path.exists(chunk_file):
            logger.info("Chunk file exists: %s -> skipping (resume).", chunk_file)
            continue

        logger.info("Processing chunk %d with %d rows", i, len(chunk_rows))
        t0 = time.time()
        dlset = JSONLChunkDataset(chunk_rows, tokenizer, max_length=seg_max)

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
                max_chars_per_segment=args.max_chars_per_segment,
            )
        except Exception as exc:
            logger.exception("Extraction failed for chunk %d: %s", i, exc)
            failed_samples.append({"chunk_idx": i, "error": str(exc)})
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
            "timestamp": time.time(),
        }
        manifest[os.path.basename(chunk_file)] = manifest_entry

        try:
            safe_save_torch(samples, chunk_file, compress=args.compress)
        except Exception as exc:
            logger.exception("Failed to save chunk file %s: %s", chunk_file, exc)
            try:
                torch.save(samples, chunk_file)
            except Exception as exc2:
                logger.exception("Fallback save also failed: %s", exc2)
                failed_samples.append({"chunk_idx": i, "error": f"save_failed: {str(exc2)}"})
                continue

        t_chunk = time.time() - t0
        peak = 0.0
        try:
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            peak = 0.0

        stats.append(
            {
                "chunk_idx": i,
                "rows_in_chunk": int(len(chunk_rows)),
                "samples_saved": int(num_saved),
                "saved_path": os.path.basename(chunk_file),
                "time_s": round(t_chunk, 3),
                "gpu_peak_GB": round(peak, 3),
            }
        )
        logger.info(
            "Saved chunk_%04d.pt (%d samples) | time %.2fs | peak GPU %.2fGB",
            i,
            num_saved,
            t_chunk,
            peak,
        )

        try:
            write_json_atomic(manifest, manifest_path)
        except Exception as exc:
            logger.warning("Failed to write manifest: %s", exc)
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
    except Exception as exc:
        logger.warning("Failed to write final manifest: %s", exc)
    try:
        write_json_atomic(stats, stats_path)
    except Exception as exc:
        logger.warning("Failed to write extract stats: %s", exc)
    if failed_samples:
        try:
            write_json_atomic(failed_samples, failed_path)
            logger.info("Failed samples (if any) written to %s", failed_path)
        except Exception:
            pass

    logger.info("Extraction complete.")


if __name__ == "__main__":
    main()
