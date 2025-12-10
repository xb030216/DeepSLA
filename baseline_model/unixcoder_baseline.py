# 修改版：codebert_finetune_linevul.py
# 在你的原始 codebert_finetune.py 基础上增加 LineVul attention-based 行级定位实现
import os
import json
import argparse
import bisect
import logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, fbeta_score,
    average_precision_score
)

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

# -------------------- Logging --------------------
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("CodeBERT_LineVul")

# --------------------
# （1）原有：Token -> Line 标签对齐（训练用）
# --------------------
def encode_token_labels(code, line_labels, offsets):
    line_starts = []
    pos = 0
    split_lines = code.split("\n")
    for line in split_lines:
        line_starts.append(pos)
        pos += len(line) + 1

    token_labels = []

    for (start, end) in offsets:
        if start == 0 and end == 0:
            token_labels.append(-100)
            continue
        line_idx = bisect.bisect_right(line_starts, start) - 1
        if 0 <= line_idx < len(line_labels):
            token_labels.append(int(line_labels[line_idx]))
        else:
            token_labels.append(0)
    return token_labels

# --------------------
# （2）保留原来基于 token 分类概率的 token->line 聚合（作为 baseline 兼容）
# --------------------
def gather_line_probs_from_token_probs(code, offsets, token_probs):
    split_lines = code.split("\n")
    line_starts = []
    pos = 0
    for line in split_lines:
        line_starts.append(pos)
        pos += len(line) + 1

    line_prob_dict = {i: [] for i in range(len(split_lines))}
    for (start, end), p in zip(offsets, token_probs):
        if start == 0 and end == 0:
            continue
        line_idx = bisect.bisect_right(line_starts, start) - 1
        if 0 <= line_idx < len(split_lines):
            line_prob_dict[line_idx].append(p)

    line_final_probs = []
    for i in range(len(split_lines)):
        probs_in_line = line_prob_dict[i]
        if len(probs_in_line) == 0:
            line_final_probs.append(0.0)
        else:
            line_final_probs.append(float(np.max(probs_in_line)))
    return line_final_probs

# --------------------
# （3）LineVul: 从 attention -> token_scores -> 行分数（核心实现）
# 说明：
# - 我们按“token 被接收的注意力总和（received attention）”作为 token 的贡献信号：
#   对于每一层每一头，attn 的 shape = (batch, heads, seq_len, seq_len)，attn[b,h,i,j] 表示 token i 对 token j 的注意力权重。
#   我们对 j（target token）沿 i 求和，再对 heads 与 layers 求和，得到每个 token 的贡献分数。
# - 在滑动窗口场景下，我们在“全序列 token 索引”层面维护累积和与计数，最后做除法平均，避免窗口重复计入。
# --------------------
# def compute_linevul_token_scores_for_code(model, tokenizer, code, max_length, stride, device):
#     """
#     返回： line_scores (按代码行的求和聚合), 以及 token-level (offsets, token_scores)
#     注意：这里我们基于 tokenizer(code, truncation=False, return_offsets_mapping=True) 得到完整 token 列表，
#     再按 window slice 执行前向并收集 attention。
#     """
#     # 1) 全序列 encoding（不截断，得到完整 offset mapping）
#     full_enc = tokenizer(code, return_offsets_mapping=True, truncation=False)
#     full_input_ids = full_enc["input_ids"]
#     full_offsets = full_enc["offset_mapping"]  # list of (start,end)
#     total_tokens = len(full_input_ids)
#
#     # 初始化 per-token 累积与计数（用于滑窗时合并）
#     token_score_sum = np.zeros(total_tokens, dtype=np.float64)
#     token_count = np.zeros(total_tokens, dtype=np.int32)
#
#     # 如果 total_tokens 小于等于 max_length，则只需一次窗口
#     if total_tokens <= max_length:
#         windows = [(0, total_tokens)]
#     else:
#         windows = []
#         start_idx = 0
#         while start_idx < total_tokens:
#             end_idx = min(start_idx + max_length, total_tokens)
#             windows.append((start_idx, end_idx))
#             if end_idx >= total_tokens:
#                 break
#             start_idx += stride
#
#     model.eval()
#     with torch.no_grad():
#         for (w_start, w_end) in windows:
#             # slice input ids and offsets for this window
#             window_input_ids = full_input_ids[w_start:w_end]
#             window_offsets = full_offsets[w_start:w_end]
#
#             # create attention mask (1s for tokens present)
#             window_attention_mask = [1] * len(window_input_ids)
#
#             # NOTE: we pass raw slices directly; tokenizer added special tokens? full_input_ids includes special tokens
#             # Some tokenizers include special tokens — offsets for them are (0,0). That's fine.
#             input_ids_tensor = torch.tensor([window_input_ids]).to(device)
#             attention_mask_tensor = torch.tensor([window_attention_mask]).to(device)
#
#             # forward with attentions
#             outputs = model(input_ids=input_ids_tensor,
#                             attention_mask=attention_mask_tensor,
#                             output_attentions=True,
#                             return_dict=True)
#
#             # attentions: tuple(len=num_layers) of tensors (batch, heads, seq_len, seq_len)
#             attentions = outputs.attentions  # tuple of tensors
#             # Compute token-level "received attention" per token in this window:
#             # for each layer l, each head h: sum over source tokens i of attn[i,j] => column-sum
#             # then sum across heads & layers.
#             seq_len = attentions[0].shape[-1]  # window seq len
#             # accumulate over layers and heads:
#             accum = np.zeros(seq_len, dtype=np.float64)  # per-token score in window
#             for layer_attn in attentions:
#                 # layer_attn shape (batch, heads, seq_len, seq_len)
#                 layer_arr = layer_attn[0].cpu().numpy()  # (heads, seq_len, seq_len)
#                 # sum over heads: gives (seq_len, seq_len) if we sum axis 0 then do col-sum
#                 # But easier: for each head, compute column sums and add
#                 # compute column sums for each head then sum across heads
#                 # column_sum_head = np.sum(head_arr, axis=1) -> shape (seq_len,)
#                 # So:
#                 col_sums = np.sum(layer_arr, axis=2)  # shape (heads, seq_len) summing over source axis gives row-sum; careful
#                 # Explanation: layer_arr[head,i,j]; np.sum(layer_arr, axis=1) sums over i => gives (heads, seq_len) with index j
#                 # But axis numbering: axis=2 or axis=1? To get sum_i attn[i,j], we want sum over axis=1 (source index) if layer_arr shape is (heads, seq_len, seq_len)
#                 col_sums = np.sum(layer_arr, axis=1)  # sum over source i -> shape (heads, seq_len)
#                 # sum across heads
#                 head_sum = np.sum(col_sums, axis=0)  # shape (seq_len,)
#                 accum += head_sum
#
#             # accum is per token in window (length seq_len). Map to global token indices [w_start, w_end)
#             token_score_sum[w_start:w_end] += accum
#             token_count[w_start:w_end] += 1
#
#     # finalize per-token scores by averaging over counts where >0
#     final_token_scores = np.zeros_like(token_score_sum)
#     mask = token_count > 0
#     final_token_scores[mask] = token_score_sum[mask] / token_count[mask]
#     final_token_scores[~mask] = 0.0
#
#     # Now map token scores to code lines (按你的规则：token 分数求和得到行分数)
#     split_lines = code.split("\n")
#     line_starts = []
#     pos = 0
#     for line in split_lines:
#         line_starts.append(pos)
#         pos += len(line) + 1
#
#     # line_scores init
#     line_scores = [0.0 for _ in range(len(split_lines))]
#     for idx, ((start, end), score) in enumerate(zip(full_offsets, final_token_scores)):
#         if start == 0 and end == 0:
#             continue
#         line_idx = bisect.bisect_right(line_starts, start) - 1
#         if 0 <= line_idx < len(split_lines):
#             # 按 LineVul 描述：行分数为该行所有 token 分数之和（求和聚合）
#             line_scores[line_idx] += float(score)
#
#     return line_scores, full_offsets, final_token_scores
def compute_linevul_token_scores_for_code(model, tokenizer, code, max_length, stride, device):
    full_enc = tokenizer(code, return_offsets_mapping=True, truncation=False)
    full_input_ids = full_enc["input_ids"]
    full_offsets = full_enc["offset_mapping"]
    total_tokens = len(full_input_ids)

    token_score_sum = np.zeros(total_tokens, dtype=np.float64)
    token_count = np.zeros(total_tokens, dtype=np.int32)

    if total_tokens <= max_length:
        windows = [(0, total_tokens)]
    else:
        windows = []
        start_idx = 0
        while start_idx < total_tokens:
            end_idx = min(start_idx + max_length, total_tokens)
            windows.append((start_idx, end_idx))
            if end_idx >= total_tokens:
                break
            start_idx += stride

    model.eval()
    with torch.no_grad():
        for (w_start, w_end) in windows:
            window_input_ids = full_input_ids[w_start:w_end]

            # 修正：处理 window 长度可能小于 max_length 的情况，无需 padding，直接由 tensor 转换
            # 但要小心 CLS token 位置。通常 window_input_ids[0] 就是 CLS (如果 tokenizer 没截断特殊符号)
            # 如果是滑动窗口中间段，可能 input_ids[0] 不是 CLS。
            # **关键点**：LineVul 强依赖 CLS。如果滑动窗口切片后没有 CLS，LineVul 逻辑会失效。
            # CodeBERT 的 Tokenizer 通常在 encode 时自动加 Special Token。
            # 你的 full_enc 是整体 encode 的。
            # -> 第一个窗口 [0:max] 有 CLS (index 0)。
            # -> 第二个窗口 [stride:stride+max] **没有 CLS** 作为 index 0。

            # *** 紧急修正 ***
            # 为了让 LineVul 在滑动窗口生效，必须强制每个窗口前加 CLS，或者只接受第一个窗口的 CLS 关注？
            # 实际上，LineVul 并不适合纯滑动窗口（因为它依赖全局 CLS）。
            # 既然你是 Token Classification，其实每个 Token 都能看到全局（在 Attention 范围内）。

            # 退一步方案（最稳妥的 Baseline）：
            # 仍然计算 index 0 (Window 的第一个 token) 对其他的关注。
            # 虽然在后续窗口中 index 0 不是全局 CLS，但在局部窗口中它充当了"主要上下文"的起始点。
            # 或者，坚持你原来的"All-to-All Attention Sum" (Graph Centrality)。

            # 鉴于实现难度和 Baseline 的合理性，我建议：
            # 方案 A (严格 LineVul): 仅对包含 CLS 的第一个窗口计算 Attention，后面忽略（但这不公平）。
            # 方案 B (你的原版 - Graph Centrality): 计算所有 Token 对 Token j 的关注。这在滑动窗口下更鲁棒。

            # *** 最终建议 ***
            # 鉴于你是滑动窗口，且中间窗口没有 CLS token，**你原来的代码（Sum over axis=1）反而是更合理的 "Attention-based Baseline"**。
            # 因为它衡量的是 "Local Context Importance"。

            # 如果你想稍微改进，可以只计算 "Last Layer" 的 attention sum，而不是所有层。
            # LineVul 论文中指出 Last Layer 包含最强的语义定位信息。

            input_ids_tensor = torch.tensor([window_input_ids]).to(device)
            attention_mask_tensor = torch.tensor([[1] * len(window_input_ids)]).to(device)

            outputs = model(input_ids=input_ids_tensor,
                            attention_mask=attention_mask_tensor,
                            output_attentions=True,
                            return_dict=True)

            attentions = outputs.attentions
            # 取最后一层 (Last Layer Only) - LineVul 推荐
            last_layer_attn = attentions[-1][0].cpu().numpy()  # (heads, seq_len, seq_len)

            # 使用你原来的 All-to-All Sum (Graph Centrality)，因为滑动窗口丢失了 CLS
            col_sums = np.sum(last_layer_attn, axis=1)  # (heads, seq_len)
            head_sum = np.sum(col_sums, axis=0)  # (seq_len,)

            token_score_sum[w_start:w_end] += head_sum
            token_count[w_start:w_end] += 1

    final_token_scores = np.zeros_like(token_score_sum)
    mask = token_count > 0
    final_token_scores[mask] = token_score_sum[mask] / token_count[mask]

    # Map to lines
    split_lines = code.split("\n")
    line_starts = []
    pos = 0
    for line in split_lines:
        line_starts.append(pos)
        pos += len(line) + 1

    line_scores = [0.0] * len(split_lines)
    for idx, ((start, end), score) in enumerate(zip(full_offsets, final_token_scores)):
        if start == 0 and end == 0: continue
        line_idx = bisect.bisect_right(line_starts, start) - 1
        if 0 <= line_idx < len(split_lines):
            line_scores[line_idx] += float(score)

    return line_scores, full_offsets, final_token_scores

# --------------------
# (4) 兼容原有的 sliding / single token-prob inference (保留)
# --------------------
def sliding_window_inference(model, tokenizer, code, max_length, stride, device):
    encoding = tokenizer(code, return_offsets_mapping=True, truncation=False)
    total_tokens = len(encoding["input_ids"])

    if total_tokens <= max_length:
        return single_inference(model, tokenizer, code, max_length, device)

    logger.debug(f"Handling long code: {total_tokens} tokens, sliding window (window={max_length}, stride={stride})")

    window_size = max_length
    stride_size = stride

    char_probs = np.zeros(len(code), dtype=np.float32)
    char_counts = np.zeros(len(code), dtype=np.int32)

    start_idx = 0
    while start_idx < total_tokens:
        end_idx = min(start_idx + window_size, total_tokens)
        window_offset_mapping = encoding["offset_mapping"][start_idx:end_idx]
        window_input_ids = encoding["input_ids"][start_idx:end_idx]
        window_attention_mask = [1] * len(window_input_ids)

        input_ids = torch.tensor([window_input_ids]).to(device)
        attention_mask = torch.tensor([window_attention_mask]).to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            token_probs = torch.softmax(logits[0], dim=-1)[:, 1].cpu().numpy()

        for (start, end), prob in zip(window_offset_mapping, token_probs):
            if start == 0 and end == 0:
                continue
            for char_idx in range(start, end):
                char_probs[char_idx] += prob
                char_counts[char_idx] += 1

        start_idx += stride_size
        if start_idx >= total_tokens - window_size // 2:
            break

    char_probs_final = np.zeros(len(code), dtype=np.float32)
    for i in range(len(char_probs)):
        if char_counts[i] > 0:
            char_probs_final[i] = char_probs[i] / char_counts[i]
    return char_probs_final

def single_inference(model, tokenizer, code, max_length, device):
    enc = tokenizer(
        code,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].cpu().numpy()

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        token_probs = torch.softmax(logits[0], dim=-1)[:, 1].cpu().numpy()

    char_probs = np.zeros(len(code), dtype=np.float32)
    char_counts = np.zeros(len(code), dtype=np.int32)

    for (start, end), prob in zip(offsets, token_probs):
        if start == 0 and end == 0:
            continue
        for char_idx in range(start, end):
            char_probs[char_idx] += prob
            char_counts[char_idx] += 1

    for i in range(len(char_probs)):
        if char_counts[i] > 0:
            char_probs[i] = char_probs[i] / char_counts[i]

    return char_probs

def char_probs_to_line_probs(code, char_probs):
    split_lines = code.split("\n")
    line_starts = []
    pos = 0
    for line in split_lines:
        line_starts.append(pos)
        pos += len(line) + 1

    line_final_probs = []
    for i in range(len(split_lines)):
        start_char = line_starts[i]
        end_char = start_char + len(split_lines[i])
        line_char_probs = char_probs[start_char:end_char]
        if len(line_char_probs) == 0:
            line_final_probs.append(0.0)
        else:
            line_final_probs.append(float(np.max(line_char_probs)))
    return line_final_probs

# --------------------
# (5) metrics (保留你的实现，略作小调整以支持两种预测来源)
# --------------------
def calculate_metrics(all_true_labels, all_pred_probs):
    flat_true = []
    flat_pred_binary = []
    flat_pred_prob = []

    for t, p in zip(all_true_labels, all_pred_probs):
        length = min(len(t), len(p))
        flat_true.extend(t[:length])
        flat_pred_prob.extend(p[:length])
        flat_pred_binary.extend([1 if x > 0.5 else 0 for x in p[:length]])

    top_hits = {1: 0, 3: 0, 5: 0}
    total_buggy_files = 0
    first_ranks = []
    all_aps = []

    for t, p in zip(all_true_labels, all_pred_probs):
        t = np.array(t)
        p = np.array(p)
        bug_indices = np.where(t == 1)[0]
        if len(bug_indices) == 0:
            continue
        total_buggy_files += 1
        sorted_indices = np.argsort(-p)
        for k in top_hits:
            if np.any(np.isin(sorted_indices[:k], bug_indices)):
                top_hits[k] += 1
        rank_list = []
        for bug_idx in bug_indices:
            rank = np.where(sorted_indices == bug_idx)[0][0] + 1
            rank_list.append(rank)
        first_rank = min(rank_list) if rank_list else len(p) + 1
        first_ranks.append(first_rank)

        precision_at_k = []
        relevant_count = 0
        for k in range(1, len(sorted_indices) + 1):
            current_idx = sorted_indices[k - 1]
            if t[current_idx] == 1:
                relevant_count += 1
                precision_at_k.append(relevant_count / k)
        if len(precision_at_k) > 0:
            ap = np.mean(precision_at_k)
            all_aps.append(ap)
        else:
            all_aps.append(0.0)

    metrics = {}
    metrics["F1"] = f1_score(flat_true, flat_pred_binary, zero_division=0)
    metrics["Precision"] = precision_score(flat_true, flat_pred_binary, zero_division=0)
    metrics["Recall"] = recall_score(flat_true, flat_pred_binary, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(flat_true, flat_pred_binary)
    try:
        metrics["AUC"] = roc_auc_score(flat_true, flat_pred_prob)
    except:
        metrics["AUC"] = 0.0
    try:
        metrics["AP"] = average_precision_score(flat_true, flat_pred_prob)
    except:
        metrics["AP"] = 0.0
    for k in top_hits:
        metrics[f"Top{k}"] = top_hits[k] / max(total_buggy_files, 1)

    if first_ranks:
        metrics["MFR"] = float(np.mean(first_ranks))
        metrics["MEdFR"] = float(np.median(first_ranks))
        metrics["MAP"] = float(np.mean(all_aps))
        metrics["Num_Buggy_Files"] = total_buggy_files
        metrics["Total_Files"] = len(all_true_labels)
        metrics["First_Ranks_Min"] = float(np.min(first_ranks))
        metrics["First_Ranks_Max"] = float(np.max(first_ranks))
        metrics["First_Ranks_Std"] = float(np.std(first_ranks))
    else:
        metrics["MFR"] = 0.0
        metrics["MEdFR"] = 0.0
        metrics["MAP"] = 0.0
        metrics["Num_Buggy_Files"] = 0
        metrics["Total_Files"] = len(all_true_labels)

    return metrics

# --------------------
# (6) evaluate_dataset：增加 use_linevul 参数以切换两种方法
# --------------------
# def evaluate_dataset(model, tokenizer, json_path, max_length, stride, device, use_linevul=False, topk=10):
#     logger.info(f"Evaluating on {json_path} (use_linevul={use_linevul}) with sliding window (stride={stride})...")
#     items = []
#     with open(json_path, "r", encoding="utf8") as f:
#         for line in f:
#             items.append(json.loads(line))
#
#     model.eval()
#
#     all_true_labels = []
#     all_pred_probs = []
#
#     with torch.no_grad():
#         for idx, item in enumerate(tqdm(items, desc="Inference")):
#             code = item["code"]
#             line_labels = item["line_labels"]
#
#             if use_linevul:
#                 # attention-based -> line scores
#                 line_scores, _, _ = compute_linevul_token_scores_for_code(model, tokenizer, code, max_length, stride, device)
#                 # line_scores already is "risk score" per-line (sum of token contributions)
#                 # 为了评估上的概率语义，我们可以对 line_scores 做归一化（softmax / min-max）。
#                 # 这里使用 min-max scaling -> [0,1] 以便与原概率指标兼容（也可以直接使用原分数排序）
#                 arr = np.array(line_scores)
#                 if np.max(arr) - np.min(arr) > 0:
#                     norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
#                 else:
#                     norm = np.zeros_like(arr)
#                 pred_line_probs = norm.tolist()
#             else:
#                 # 原始 token-classifier 概率路径：先得到字符级概率再聚合到行
#                 char_probs = sliding_window_inference(model, tokenizer, code, max_length, stride, device)
#                 pred_line_probs = char_probs_to_line_probs(code, char_probs)
#
#             min_len = min(len(line_labels), len(pred_line_probs))
#             all_true_labels.append(line_labels[:min_len])
#             all_pred_probs.append(pred_line_probs[:min_len])
#
#     return calculate_metrics(all_true_labels, all_pred_probs)
def evaluate_dataset(model, tokenizer, json_path, max_length, stride, device,
                     use_linevul=False, topk=10, save_perfile=True, output_dir=None):
    """
    Evaluate dataset and optionally save per-file metrics for significance testing.
    Returns: summary_metrics (same as original calculate_metrics output)
    """
    import os
    import json
    from sklearn.metrics import average_precision_score  # ensure available in this scope
    import numpy as np

    logger.info(f"Evaluating on {json_path} (use_linevul={use_linevul}) with sliding window (stride={stride})...")

    # ---- load data ----
    items = []
    with open(json_path, "r", encoding="utf8") as f:
        for line in f:
            items.append(json.loads(line))

    model.eval()

    all_true_labels = []
    all_pred_probs = []
    per_file_records = []   # <-- 用于统计显著性检验

    with torch.no_grad():
        for file_id, item in enumerate(tqdm(items, desc="Inference")):

            code = item["code"]
            line_labels = item["line_labels"]

            # 最终要预测的行级概率
            if use_linevul:
                line_scores, _, _ = compute_linevul_token_scores_for_code(
                    model, tokenizer, code, max_length, stride, device
                )
                # normalize to [0,1]
                arr = np.array(line_scores)
                if np.max(arr) - np.min(arr) > 0:
                    pred_line_probs = ((arr - np.min(arr)) / (np.max(arr) - np.min(arr))).tolist()
                else:
                    pred_line_probs = np.zeros_like(arr).tolist()

            else:
                # sliding window
                char_probs = sliding_window_inference(
                    model, tokenizer, code, max_length, stride, device
                )
                pred_line_probs = char_probs_to_line_probs(code, char_probs)

            # 对齐长度
            min_len = min(len(line_labels), len(pred_line_probs))
            labels = np.array(line_labels[:min_len])
            probs = np.array(pred_line_probs[:min_len])

            # 保存全局评估所需数据
            all_true_labels.append(labels)
            all_pred_probs.append(probs)

            # ====== per-file metrics for statistical test ======
            bug_lines = np.where(labels == 1)[0].tolist()
            if len(bug_lines) > 0:
                # 排序 index
                order = np.argsort(-probs)

                # top-k
                top1 = int(any(b in order[:1] for b in bug_lines))
                top3 = int(any(b in order[:3] for b in bug_lines))
                top5 = int(any(b in order[:5] for b in bug_lines))

                # rank (0-based index; keep consistent with your other code - currently returns 0-based)
                ranks = [int(np.where(order == b)[0][0]) for b in bug_lines]
                rank = min(ranks)+1

                # average precision
                try:
                    ap = float(average_precision_score(labels, probs))
                except:
                    ap = 0.0

                per_file_records.append({
                    "file_id": file_id,
                    "top1": top1,
                    "top3": top3,
                    "top5": top5,
                    "rank": rank,
                    "ap": ap
                })
            # 如果该文件无缺陷行（bug_lines 为空）则不加入 per_file_records，
            # 与之前设计保持一致（你的主模型同样忽略无缺陷文件）

    # ---- 保存 per_file_metrics.json ----
    if save_perfile:
        if output_dir is None:
            output_dir = os.path.dirname(json_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, "per_file_metrics.json")
        with open(save_path, "w", encoding="utf8") as f:
            json.dump(per_file_records, f, indent=2)
        logger.info(f"Saved per-file metrics to: {save_path}")

    # ---- 原始 summary 指标 ----
    summary_metrics = calculate_metrics(all_true_labels, all_pred_probs)
    return summary_metrics


# --------------------
# (7) Dataset & training 主逻辑（保留你原来的 Trainer 流程）
# --------------------
class LineTokenDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        logger.info(f"Loading dataset from {path}")
        self.items = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                self.items.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.items[idx]
        code = item["code"]
        line_labels = item["line_labels"]

        enc = self.tokenizer(
            code,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
        )

        token_labels = encode_token_labels(code, line_labels, enc["offset_mapping"])
        enc["labels"] = token_labels
        output = {k: torch.tensor(v) for k, v in enc.items() if k != "offset_mapping"}
        return output

    def __len__(self):
        return len(self.items)

# --------------------
# (8) main: 加入 --use_linevul 参数
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="folder containing train/val/test.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256, help="sliding window stride")
    parser.add_argument("--use_linevul", action="store_true", help="If set, evaluate using attention-based LineVul method")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=2
    ).to(device)

    train_path = os.path.join(args.dataset_dir, "train.jsonl")
    val_path = os.path.join(args.dataset_dir, "val.jsonl")
    test_path = os.path.join(args.dataset_dir, "test.jsonl")

    train_set = LineTokenDataset(train_path, tokenizer, args.max_len)
    val_set = LineTokenDataset(val_path, tokenizer, args.max_len)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting Training...")
    trainer.train()

    best_model_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    logger.info(f"Best model saved to {best_model_path}")

    logger.info("Starting Test Set Evaluation...")
    model = AutoModelForTokenClassification.from_pretrained(best_model_path, num_labels=2).to(device)

    # test_metrics = evaluate_dataset(model, tokenizer, test_path, args.max_len, args.stride, device, use_linevul=args.use_linevul)
    test_metrics = evaluate_dataset(
        model, tokenizer,
        json_path=test_path,
        max_length=args.max_len,
        stride=args.stride,
        device=device,
        use_linevul=args.use_linevul,
        save_perfile=True,
        output_dir=args.output_dir
    )

    logger.info("==== Test Set Results ====")
    print(json.dumps(test_metrics, indent=4))

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

if __name__ == "__main__":
    main()
