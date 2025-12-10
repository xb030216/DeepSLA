import json
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import argparse


# ==========================
# 特征工程：浅层行特征
# ==========================
def extract_shallow_features(line):
    """
    行级浅层特征（扩展版本）
    返回一个特征向量 list
    """
    line_stripped = line.strip()
    line_lower = line.lower()

    return [
        # 基础长度特征
        len(line),
        len(line_stripped),  # 去掉空格的长度
        len(line.split()),  # 单词数量

        # 字符类型统计
        sum(c.isdigit() for c in line),  # 数字数量
        sum(c.isalpha() for c in line),  # 字母数量
        sum(c.isspace() for c in line),  # 空格数量
        sum(not c.isalnum() and not c.isspace() for c in line),  # 特殊字符数量

        # 符号统计
        line.count("="),
        line.count("("), line.count(")"),
        line.count("{"), line.count("}"),
        line.count("["), line.count("]"),
        line.count(";"),
        line.count(","),
        line.count("."),
        line.count(":"),
        line.count("!"),

        # 关键字特征
        int("null" in line),
        int("throw" in line),
        int("catch" in line),
        int("try" in line),
        int("finally" in line),
        int("error" in line_lower),
        int("exception" in line_lower),
        int("fail" in line_lower),
        int("debug" in line_lower),
        int("warn" in line_lower),
        int("if" in line),  # 控制流相关
        int("for" in line),
        int("while" in line),
        int("switch" in line),
        int("case" in line),

        # 代码结构特征
        int(line_stripped.startswith("//")),  # 注释行
        int(line_stripped.startswith("/*")),
        int(line_stripped.startswith("*")),
        int(line_stripped.endswith(";")),  # 以分号结束
        int('"' in line),  # 字符串相关
        int("'" in line),
        int("//" in line),  # 行内注释

        # 复杂性特征
        len(line_stripped) > 100,  # 是否超长行
        len(line_stripped) < 10,  # 是否超短行
    ]


# ==========================
#  加载 jsonl → 行级样本
# ==========================
def load_dataset(path):
    X = []
    Y = []

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            lines = obj["lines"]
            labels = obj["line_labels"]

            for t, y in zip(lines, labels):
                X.append(extract_shallow_features(t))
                Y.append(y)

    return np.array(X), np.array(Y)


# ==========================
#  寻找最优阈值
# ==========================
def find_optimal_threshold(model, X_val, Y_val):
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.01, 0.5, 0.01)  # 在较低概率范围搜索
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        pred = (probs >= threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(Y_val, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

# ==========================
#    MFR, MEdFR, MAP
# ==========================
def evaluate_rank_metrics(model, path):
    first_ranks = []
    ap_list = []

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            lines = obj["lines"]
            labels = obj["line_labels"]

            gold = {i for i, v in enumerate(labels) if v == 1}
            if not gold:
                continue   # 跳过没有 bug 的文件

            # 预测概率
            X = np.array([extract_shallow_features(t) for t in lines])
            probs = model.predict_proba(X)[:, 1]

            # 排名（大→小）
            ranked = np.argsort(-probs)

            # ------- MFR / MEdFR -------
            first_rank = None
            for idx, line_id in enumerate(ranked, start=1):
                if line_id in gold:
                    first_rank = idx
                    break
            first_ranks.append(first_rank)

            # ------- MAP -------
            # AP = sum over ranks: Precision@k * rel(k)
            num_correct = 0
            ap_sum = 0.0
            for k, line_id in enumerate(ranked, start=1):
                if line_id in gold:
                    num_correct += 1
                    precision_at_k = num_correct / k
                    ap_sum += precision_at_k

            ap = ap_sum / len(gold)
            ap_list.append(ap)

    MFR = np.mean(first_ranks) if first_ranks else 0
    MEdFR = np.median(first_ranks) if first_ranks else 0
    MAP = np.mean(ap_list) if ap_list else 0

    return {
        "MFR": MFR,
        "MEdFR": MEdFR,
        "MAP": MAP,
        "num_files": len(first_ranks)
    }

# ==========================
#   Top-K 评估（按文件）
# ==========================
def evaluate_topk(model, path):
    top1 = top3 = top5 = 0
    total = 0

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            lines = obj["lines"]
            labels = obj["line_labels"]

            X = np.array([extract_shallow_features(t) for t in lines])
            probs = model.predict_proba(X)[:, 1]

            ranked = np.argsort(-probs)
            gold = {i for i, v in enumerate(labels) if v == 1}

            if not gold: continue
            total += 1

            top1 += 1 if ranked[0] in gold else 0
            top3 += 1 if set(ranked[:3]) & gold else 0
            top5 += 1 if set(ranked[:5]) & gold else 0

    return {
        "Top1": top1 / total,
        "Top3": top3 / total,
        "Top5": top5 / total
    }


# ==========================
#           MAIN
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    # 加载数据
    print("加载训练数据...")
    X_train, Y_train = load_dataset(args.train)
    print("加载测试数据...")
    X_test, Y_test = load_dataset(args.test)

    print(f"训练集: {X_train.shape[0]} 样本, 正样本比例: {Y_train.mean():.4f}")
    print(f"测试集: {X_test.shape[0]} 样本, 正样本比例: {Y_test.mean():.4f}")

    # 分割训练集用于阈值调优
    X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train
    )

    # 训练模型（使用类别平衡）
    print("训练模型...")
    model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',  # 处理类别不平衡
        C=0.1,  # 更强的正则化
        random_state=42
    )
    model.fit(X_train_split, Y_train_split)

    # 默认阈值评估
    print("\n=== 默认阈值 (0.5) ===")
    pred_default = model.predict(X_test)
    prec_default, rec_default, f1_default, _ = precision_recall_fscore_support(
        Y_test, pred_default, average="binary", zero_division=0
    )
    print(f"F1: {f1_default:.4f}")
    print(f"Precision: {prec_default:.4f}")
    print(f"Recall: {rec_default:.4f}")

    # 寻找最优阈值
    print("\n寻找最优阈值...")
    optimal_threshold, val_f1 = find_optimal_threshold(model, X_val_split, Y_val_split)
    print(f"验证集最优阈值: {optimal_threshold:.4f} (F1: {val_f1:.4f})")

    # 最优阈值评估
    print("\n=== 最优阈值评估 ===")
    test_probs = model.predict_proba(X_test)[:, 1]
    pred_optimal = (test_probs >= optimal_threshold).astype(int)
    prec_opt, rec_opt, f1_opt, _ = precision_recall_fscore_support(
        Y_test, pred_optimal, average="binary", zero_division=0
    )
    print(f"F1: {f1_opt:.4f}")
    print(f"Precision: {prec_opt:.4f}")
    print(f"Recall: {rec_opt:.4f}")

    # Top-K 评估
    print("\n=== Top-K 评估 ===")
    topk_results = evaluate_topk(model, args.test)
    for metric, value in topk_results.items():
        print(f"{metric}: {value:.4f}")
    # Rank-based metrics
    print("\n=== Rank-based Metrics (MFR / MEdFR / MAP) ===")
    rank_results = evaluate_rank_metrics(model, args.test)
    for metric, value in rank_results.items():
        if metric != "num_files":
            print(f"{metric}: {value:.4f}")
    print(f"评估文件数: {rank_results['num_files']}")


    # 额外统计信息
    print(f"\n=== 统计信息 ===")
    print(f"测试集正样本预测数 (默认阈值): {pred_default.sum()}")
    print(f"测试集正样本预测数 (最优阈值): {pred_optimal.sum()}")
    print(f"测试集实际正样本数: {Y_test.sum()}")