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
logger = logging.getLogger("CodeBERT_Baseline")


# ---------------------------------------------------------
#  1. Token → Line 标签对齐 (用于训练数据准备)
# ---------------------------------------------------------
def encode_token_labels(code, line_labels, offsets):
    """
    将行级标签 (line_labels) 扩展为 Token 级标签 (token_labels)。
    用于训练阶段计算 Loss。
    """
    # 1. 计算每一行在字符串中的 (start, end)
    line_starts = []
    pos = 0
    # 注意：split('\n') 会丢失换行符，位置计算需 +1
    # 如果代码以 \n 结尾，split 会产生空串，需注意处理
    split_lines = code.split("\n")
    for line in split_lines:
        line_starts.append(pos)
        pos += len(line) + 1

    token_labels = []

    for (start, end) in offsets:
        # 特殊 token (CLS, SEP, PAD) 的 offset 通常是 (0,0)
        if start == 0 and end == 0:
            token_labels.append(-100)  # PyTorch CrossEntropyLoss 忽略 -100
            continue

        # 二分查找当前 token 属于哪一行
        # bisect_right 返回插入点，减 1 得到该数值所属的区间索引
        line_idx = bisect.bisect_right(line_starts, start) - 1

        if 0 <= line_idx < len(line_labels):
            token_labels.append(int(line_labels[line_idx]))
        else:
            # 异常情况（如分词越界），标为 0 (正常) 或 -100
            token_labels.append(0)

    return token_labels


# ---------------------------------------------------------
#  2. Token → Line 概率聚合 (用于预测/评估)
# ---------------------------------------------------------
def gather_line_probs(code, offsets, token_probs):
    """
    将模型预测的 Token 概率聚合回行级概率。
    策略：Max-Pooling (Baseline 标准做法: 只要行内有一个 token 是 bug，该行就是 bug)
    """
    split_lines = code.split("\n")
    line_starts = []
    pos = 0
    for line in split_lines:
        line_starts.append(pos)
        pos += len(line) + 1

    # 初始化每行的概率列表
    line_prob_dict = {i: [] for i in range(len(split_lines))}

    # 遍历 token 概率
    # 注意：CodeBERT 有 512 限制，offsets 和 token_probs 长度可能远小于代码总行数
    for (start, end), p in zip(offsets, token_probs):
        if start == 0 and end == 0:
            continue

        line_idx = bisect.bisect_right(line_starts, start) - 1

        if 0 <= line_idx < len(split_lines):
            line_prob_dict[line_idx].append(p)

    # 聚合
    line_final_probs = []
    for i in range(len(split_lines)):
        probs_in_line = line_prob_dict[i]
        if len(probs_in_line) == 0:
            # CodeBERT 截断导致该行没被看到 -> 预测概率为 0
            line_final_probs.append(0.0)
        else:
            # Max Pooling
            line_final_probs.append(float(np.max(probs_in_line)))
            # 也可以尝试 Mean Pooling: np.mean(probs_in_line)

    return line_final_probs


# ---------------------------------------------------------
#  3. 数据集类
# ---------------------------------------------------------
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
        line_labels = item["line_labels"]  # [0, 1, 0, ...]

        # Tokenize
        enc = self.tokenizer(
            code,
            max_length=self.max_length,
            truncation=True,  # 关键：CodeBERT 必须截断
            padding="max_length",
            return_offsets_mapping=True,
        )

        # 生成 Token 级标签
        token_labels = encode_token_labels(code, line_labels, enc["offset_mapping"])
        enc["labels"] = token_labels

        # 训练时不需要 offset_mapping，但验证时需要重新 tokenize，所以这里只返回 tensor
        output = {k: torch.tensor(v) for k, v in enc.items() if k != "offset_mapping"}
        return output

    def __len__(self):
        return len(self.items)


# ---------------------------------------------------------
#  4. 滑动窗口推理函数
# ---------------------------------------------------------
def sliding_window_inference(model, tokenizer, code, max_length, stride, device):
    """
    使用滑动窗口进行推理，处理超过max_length的长代码
    """
    # 计算代码的总token数
    encoding = tokenizer(code, return_offsets_mapping=True, truncation=False)
    total_tokens = len(encoding["input_ids"])

    if total_tokens <= max_length:
        # 如果代码不长，直接使用单次推理
        return single_inference(model, tokenizer, code, max_length, device)

    logger.debug(f"长代码处理: {total_tokens} tokens, 使用滑动窗口 (window={max_length}, stride={stride})")

    # 滑动窗口参数
    window_size = max_length
    stride_size = stride

    # 存储每个字符位置的概率（初始化为0）
    char_probs = np.zeros(len(code), dtype=np.float32)
    char_counts = np.zeros(len(code), dtype=np.int32)

    # 遍历每个窗口
    start_idx = 0
    while start_idx < total_tokens:
        # 获取当前窗口的token范围
        end_idx = min(start_idx + window_size, total_tokens)

        # 获取窗口对应的字符范围
        window_offset_mapping = encoding["offset_mapping"][start_idx:end_idx]

        # 提取窗口内的token
        window_input_ids = encoding["input_ids"][start_idx:end_idx]
        window_attention_mask = [1] * len(window_input_ids)

        # 转换为tensor
        input_ids = torch.tensor([window_input_ids]).to(device)
        attention_mask = torch.tensor([window_attention_mask]).to(device)

        # 推理
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            token_probs = torch.softmax(logits[0], dim=-1)[:, 1].cpu().numpy()

        # 将token概率映射到字符位置
        for (start, end), prob in zip(window_offset_mapping, token_probs):
            if start == 0 and end == 0:
                continue

            # 将概率累加到字符位置
            for char_idx in range(start, end):
                char_probs[char_idx] += prob
                char_counts[char_idx] += 1

        # 移动窗口
        start_idx += stride_size

        # 如果已经覆盖了所有token，提前结束
        if start_idx >= total_tokens - window_size // 2:
            break

    # 计算平均概率（对于被多个token覆盖的字符位置）
    char_probs_final = np.zeros(len(code), dtype=np.float32)
    for i in range(len(char_probs)):
        if char_counts[i] > 0:
            char_probs_final[i] = char_probs[i] / char_counts[i]

    return char_probs_final


def single_inference(model, tokenizer, code, max_length, device):
    """单次推理（用于短代码）"""
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

    # 将token概率映射到字符位置
    char_probs = np.zeros(len(code), dtype=np.float32)
    char_counts = np.zeros(len(code), dtype=np.int32)

    for (start, end), prob in zip(offsets, token_probs):
        if start == 0 and end == 0:
            continue

        for char_idx in range(start, end):
            char_probs[char_idx] += prob
            char_counts[char_idx] += 1

    # 计算平均概率
    for i in range(len(char_probs)):
        if char_counts[i] > 0:
            char_probs[i] = char_probs[i] / char_counts[i]

    return char_probs


def char_probs_to_line_probs(code, char_probs):
    """将字符级概率转换为行级概率"""
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

        # 获取该行所有字符的概率
        line_char_probs = char_probs[start_char:end_char]

        if len(line_char_probs) == 0:
            line_final_probs.append(0.0)
        else:
            # 使用该行字符的最大概率作为行概率
            line_final_probs.append(float(np.max(line_char_probs)))

    return line_final_probs


# ---------------------------------------------------------
#  5. 全面评估函数 (Top-K, MCC, AUC, MFR, MEdFR, MAP)
# ---------------------------------------------------------
def calculate_metrics(all_true_labels, all_pred_probs):
    """
    all_true_labels: List[List[int]] (文件级 ground truth)
    all_pred_probs: List[List[float]] (文件级预测概率)
    """
    # 1. 展平用于计算全局指标 (F1, MCC, AUC)
    flat_true = []
    flat_pred_binary = []
    flat_pred_prob = []

    for t, p in zip(all_true_labels, all_pred_probs):
        # 确保长度一致（防御性编程）
        length = min(len(t), len(p))
        flat_true.extend(t[:length])
        flat_pred_prob.extend(p[:length])
        flat_pred_binary.extend([1 if x > 0.5 else 0 for x in p[:length]])

    # 2. 计算 Top-K
    top_hits = {1: 0, 3: 0, 5: 0}
    total_buggy_files = 0

    # 3. 新增：用于计算 MFR, MEdFR, MAP 的列表
    first_ranks = []  # 每个有缺陷文件的第一个缺陷行排名
    all_aps = []  # 每个有缺陷文件的平均精度

    for t, p in zip(all_true_labels, all_pred_probs):
        t = np.array(t)
        p = np.array(p)

        bug_indices = np.where(t == 1)[0]
        if len(bug_indices) == 0:
            continue  # 没有 bug 的文件不参与 Top-K, MFR, MEdFR, MAP 计算

        total_buggy_files += 1

        # 按概率降序排列的索引
        sorted_indices = np.argsort(-p)

        # 计算 Top-K
        for k in top_hits:
            # 检查前 k 个预测中是否包含任意一个 bug 行
            if np.any(np.isin(sorted_indices[:k], bug_indices)):
                top_hits[k] += 1

        # ========== 计算 MFR (Mean First Rank) ==========
        # 找到第一个缺陷行的排名
        rank_list = []
        for bug_idx in bug_indices:
            # 找到该缺陷行在排序中的位置（从1开始计数）
            rank = np.where(sorted_indices == bug_idx)[0][0] + 1
            rank_list.append(rank)

        # 取第一个缺陷行的排名（最小的排名）
        first_rank = min(rank_list) if rank_list else len(p) + 1
        first_ranks.append(first_rank)

        # ========== 计算 MAP (Mean Average Precision) ==========
        # 计算每个缺陷文件的平均精度
        # 按概率排序后，计算每个位置的精度
        precision_at_k = []
        relevant_count = 0

        for k in range(1, len(sorted_indices) + 1):
            current_idx = sorted_indices[k - 1]
            if t[current_idx] == 1:  # 如果当前行是缺陷行
                relevant_count += 1
                precision_at_k.append(relevant_count / k)

        if len(precision_at_k) > 0:
            # 计算平均精度 (AP)
            ap = np.mean(precision_at_k)
            all_aps.append(ap)
        else:
            all_aps.append(0.0)

    metrics = {}

    # 基础指标
    metrics["F1"] = f1_score(flat_true, flat_pred_binary, zero_division=0)
    metrics["Precision"] = precision_score(flat_true, flat_pred_binary, zero_division=0)
    metrics["Recall"] = recall_score(flat_true, flat_pred_binary, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(flat_true, flat_pred_binary)

    # AUC 相关指标
    try:
        metrics["AUC"] = roc_auc_score(flat_true, flat_pred_prob)
    except:
        metrics["AUC"] = 0.0

    try:
        metrics["AP"] = average_precision_score(flat_true, flat_pred_prob)
    except:
        metrics["AP"] = 0.0

    # Top-K 指标
    for k in top_hits:
        metrics[f"Top{k}"] = top_hits[k] / max(total_buggy_files, 1)

    # ========== 新增指标 ==========
    if first_ranks:
        # MFR (Mean First Rank) - 平均首次排名
        metrics["MFR"] = float(np.mean(first_ranks))

        # MEdFR (Median First Rank) - 中位数首次排名
        metrics["MEdFR"] = float(np.median(first_ranks))

        # MAP (Mean Average Precision) - 平均精度均值
        metrics["MAP"] = float(np.mean(all_aps))

        # 添加统计信息
        metrics["Num_Buggy_Files"] = total_buggy_files
        metrics["Total_Files"] = len(all_true_labels)
        metrics["First_Ranks_Min"] = float(np.min(first_ranks))
        metrics["First_Ranks_Max"] = float(np.max(first_ranks))
        metrics["First_Ranks_Std"] = float(np.std(first_ranks))
    else:
        # 如果没有缺陷文件，设置默认值
        metrics["MFR"] = 0.0
        metrics["MEdFR"] = 0.0
        metrics["MAP"] = 0.0
        metrics["Num_Buggy_Files"] = 0
        metrics["Total_Files"] = len(all_true_labels)

    return metrics


def evaluate_dataset(model, tokenizer, json_path, max_length, stride, device):
    """ 对整个数据集进行推理并评估，使用滑动窗口处理长代码 """
    logger.info(f"Evaluating on {json_path} with sliding window (stride={stride})...")
    items = []
    with open(json_path, "r", encoding="utf8") as f:
        for line in f:
            items.append(json.loads(line))

    model.eval()

    all_true_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for idx, item in enumerate(tqdm(items, desc="Inference")):
            code = item["code"]
            line_labels = item["line_labels"]

            # 使用滑动窗口推理
            char_probs = sliding_window_inference(model, tokenizer, code, max_length, stride, device)

            # 将字符概率转换为行概率
            line_probs = char_probs_to_line_probs(code, char_probs)

            # 对齐长度（防止split逻辑微小差异）
            min_len = min(len(line_labels), len(line_probs))
            all_true_labels.append(line_labels[:min_len])
            all_pred_probs.append(line_probs[:min_len])

    return calculate_metrics(all_true_labels, all_pred_probs)


# ---------------------------------------------------------
#  Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="folder containing train/val/test.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256, help="滑动窗口的步长，默认256 (max_len的一半)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=2
    ).to(device)

    # 训练集和验证集
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
        fp16=True,  # 开启混合精度加速
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

    # 保存最佳模型
    best_model_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    logger.info(f"Best model saved to {best_model_path}")

    # ---------------------------------------------------------
    #  Test Set Evaluation (Crucial for Paper)
    # ---------------------------------------------------------
    logger.info("Starting Test Set Evaluation...")
    # 使用最佳模型进行推理
    # 为了保险，重新加载一下最佳权重
    model = AutoModelForTokenClassification.from_pretrained(best_model_path, num_labels=2).to(device)

    test_metrics = evaluate_dataset(model, tokenizer, test_path, args.max_len, args.stride, device)

    logger.info("==== Test Set Results ====")
    print(json.dumps(test_metrics, indent=4))

    # 保存结果到 json
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    main()