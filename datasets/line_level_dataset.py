# ==============================================
# datasets/line_level_dataset.py
# Line-level dataset with safe truncation & logging
# ==============================================
import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import logging

logger = logging.getLogger(__name__)

class LineLevelDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=None):
        super().__init__()
        df = pd.read_csv(csv_path)

        # --- parse bugLineNum safely ---
        def parse_list(x):
            if pd.isna(x):
                return []
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    import re
                    nums = re.findall(r"\d+", x)
                    return [int(n) for n in nums]
            if isinstance(x, (list, tuple)):
                return list(x)
            return []

        df["bugLineNum"] = df["bugLineNum"].apply(parse_list)
        self.rows = df.to_dict(orient="records")
        self.tokenizer = tokenizer

        # use tokenizer max length if not given
        self.max_length = max_length or min(
            getattr(tokenizer, "model_max_length", 16384),
            16384
        )

        logger.info(f"[Dataset] Loaded {len(self.rows)} samples from {csv_path}")
        logger.info(f"[Dataset] Using max_length={self.max_length}")

    def __len__(self):
        return len(self.rows)

    # ======================================================
    # internal helper: map lines to token spans
    # ======================================================
    def _map_lines_to_token_spans(self, code_str):
        lines = code_str.splitlines(keepends=True)
        char_positions = []
        pos = 0
        for ln in lines:
            start = pos
            end = pos + len(ln)
            char_positions.append((start, end))
            pos = end

        # Safe tokenize (return_offsets_mapping)
        enc = self.tokenizer(
            code_str,
            return_offsets_mapping=True,
            return_attention_mask=True,
            truncation=True,  # ✅ 强制截断
            max_length=self.max_length,
        )

        offsets = enc["offset_mapping"]
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        num_tokens = len(input_ids)

        # map token midpoint to line idx
        token_midpoints = [(a + b) / 2 for (a, b) in offsets]
        token_to_line = []
        for mid in token_midpoints:
            li = None
            for idx, (ls, le) in enumerate(char_positions):
                if ls <= mid < le:
                    li = idx
                    break
            if li is None:
                li = len(lines) - 1
            token_to_line.append(li)

        # group tokens by line
        line_token_indices = {}
        for tidx, lidx in enumerate(token_to_line):
            line_token_indices.setdefault(lidx, []).append(tidx)

        # build spans
        line_spans = []
        for i in range(len(lines)):
            toks = line_token_indices.get(i, [])
            if not toks:
                line_spans.append(None)
            else:
                s = toks[0]
                e = toks[-1] + 1
                line_spans.append((s, e))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "line_spans": line_spans,
            "num_tokens": num_tokens,
            "num_lines": len(lines)
        }

    # ======================================================
    # main: get item
    # ======================================================
    def __getitem__(self, idx):
        row = self.rows[idx]
        code = row.get("buggyCode", "")
        bug_lines = set(row.get("bugLineNum", []))  # 1-based indices

        mapped = self._map_lines_to_token_spans(code)
        num_tokens = mapped["num_tokens"]
        if num_tokens > self.max_length:
            logger.warning(
                f"[Truncate] Sample {idx} too long ({num_tokens} > {self.max_length}), truncated."
            )

        input_ids = mapped["input_ids"]
        attention_mask = mapped["attention_mask"]
        line_spans = mapped["line_spans"]
        num_lines = mapped["num_lines"]

        # Ensure each line has at least one token span
        final_spans = []
        token_count = len(input_ids)
        for i, span in enumerate(line_spans):
            if span is None:
                found = False
                for j in range(i - 1, -1, -1):
                    if line_spans[j] is not None:
                        s = line_spans[j][1] - 1
                        final_spans.append((s, s + 1))
                        found = True
                        break
                if not found:
                    for j in range(i + 1, len(line_spans)):
                        if line_spans[j] is not None:
                            s = line_spans[j][0]
                            final_spans.append((s, s + 1))
                            found = True
                            break
                if not found:
                    final_spans.append((0, 1))
            else:
                s, e = span
                s = max(0, min(s, token_count - 1))
                e = max(s + 1, min(e, token_count))
                final_spans.append((s, e))

        # build per-line labels
        labels = [
            float(1 if (i + 1) in bug_lines else 0)
            for i in range(num_lines)
        ]

        # convert to tensors
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.float32)

        item = {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "line_spans": final_spans,
            "line_labels": labels_t,
            "num_lines": len(final_spans),
        }
        return item
