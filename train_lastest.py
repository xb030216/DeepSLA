import os
import gc
import time
import json
import math
import random
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score, matthews_corrcoef,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, fbeta_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

# -------------------- Logging --------------------
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("train_adapter")

# -------------------- 忽略 sklearn 的 warning --------------------
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ====================================================
# Dataset
# ====================================================
class TokenLineDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = sorted([os.path.join(feature_dir, f)
                             for f in os.listdir(feature_dir)
                             if f.endswith(".pt")])
        self.samples = []
        for f in self.files:
            chunk = torch.load(f, map_location="cpu")
            self.samples.extend(chunk)
        logger.info(f"[Dataset] Loaded {len(self.samples)} samples from {feature_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        token_hidden = item.get("token_hidden", None)
        line_spans = item.get("line_spans", [])
        labels = item.get("line_labels", [])
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().clone().float()
        else:
            labels = torch.tensor(labels, dtype=torch.float32)
        if token_hidden is None:
            raise ValueError("Missing token_hidden in sample")
        line_vecs = []
        for s, e in line_spans:
            s = max(0, int(s))
            e = max(s + 1, int(e))
            e = min(e, token_hidden.size(0))
            line_vecs.append(token_hidden[s:e].mean(dim=0))
        feats = torch.stack(line_vecs)
        return feats, labels


def collate_fn(batch):
    max_len = max(f.size(0) for f, _ in batch)
    feat_dim = batch[0][0].size(1)
    feats = torch.zeros(len(batch), max_len, feat_dim)
    labels = torch.zeros(len(batch), max_len)
    masks = torch.zeros(len(batch), max_len)
    for i, (f, l) in enumerate(batch):
        L = f.size(0)
        feats[i, :L] = f
        labels[i, :len(l)] = l
        masks[i, :L] = 1
    return feats, labels, masks


# ====================================================
# Focal Loss
# ====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer，不随梯度更新，但随 state_dict 保存
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        # 截取对应长度的位置编码相加
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
# ====================================================
# Model
# ====================================================
class LineAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_layers=2,
                 num_heads=8, dropout=0.3):
        super().__init__()
        self.proj_down = nn.Linear(input_dim, hidden_dim)

        # 【新增 1】 定义位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=5000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # x: (B, L, D_in)
        x = self.proj_down(x)  # (B, L, D_hidden)

        # 【新增 2】 加入位置编码
        # 注意：位置编码必须在 Transformer 之前加
        x = self.pos_encoder(x)

        key_padding_mask = (mask == 0) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        logits = self.out(x).squeeze(-1)
        return logits

class MLPAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # x: (B, L, D)
        B, L, D = x.shape
        h = self.act(self.fc1(x))  # (B, L, hidden)
        h = self.drop(h)
        logits = self.fc2(h).squeeze(-1)  # (B, L)
        return logits

def evaluate_metrics(all_logits, all_labels, all_masks):
    from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
    flat_probs, flat_preds, flat_labels = [], [], []
    top_hits = {1: 0, 3: 0, 5: 0}
    total_files = 0

    for logits, labels, masks in zip(all_logits, all_labels, all_masks):
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # 展平有效行用于全局指标
        flat_probs.append(probs[masks.bool()].cpu())
        flat_preds.append(preds[masks.bool()].cpu())
        flat_labels.append(labels[masks.bool()].cpu())

        # 文件级 Top@N
        B = logits.size(0)
        for i in range(B):
            valid_len = int(masks[i].sum().item())
            if valid_len == 0:
                continue
            l = labels[i, :valid_len].cpu().numpy()
            p_line = probs[i, :valid_len].cpu().numpy()
            bug_lines = [j for j, v in enumerate(l) if v == 1]
            if not bug_lines:
                continue
            total_files += 1
            top_idx = p_line.argsort()[::-1]
            for n in top_hits:
                if any(b in top_idx[:n] for b in bug_lines):
                    top_hits[n] += 1

    y_true = torch.cat(flat_labels).numpy()
    y_pred = torch.cat(flat_preds).numpy()
    y_prob = torch.cat(flat_probs).numpy()

    if len(set(y_true)) < 2:
        return {m: 0.0 for m in ["F2", "Precision", "Recall", "MCC", "AUC", "Top1", "Top3", "Top5"]}

    metrics = dict()
    # metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["F2"] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    except:
        metrics["AUC"] = 0.0

    for n in top_hits:
        metrics[f"Top{n}"] = top_hits[n] / max(total_files, 1)

    return metrics



# ====================================================
# Full Evaluation Helper（科研增强版）
# ====================================================
@torch.no_grad()
def evaluate_full(model, loader, device, save_prefix=None):
    model.eval()
    all_logits, all_labels, all_masks = [], [], []
    for feats, labels, masks in loader:
        feats, labels, masks = feats.to(device), labels.to(device), masks.to(device)
        logits = model(feats, masks)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_masks.append(masks.cpu())

    # ⚙️ 不拼接不等长 batch，逐 batch 评估
    metrics = evaluate_metrics(all_logits, all_labels, all_masks)

    # ---------- 绘图 ----------
    # 展平所有有效行用于 ROC/PR
    flat_probs, flat_labels = [], []
    for logits, labels, masks in zip(all_logits, all_labels, all_masks):
        flat_probs.append(torch.sigmoid(logits[masks.bool()]).cpu())
        flat_labels.append(labels[masks.bool()].cpu())
    probs = torch.cat(flat_probs).numpy()
    y_true = torch.cat(flat_labels).numpy()

    if save_prefix:
        import numpy as np
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
        os.makedirs(save_prefix, exist_ok=True)

        # ROC 曲线
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(save_prefix, "roc_curve.png"), dpi=300)
        plt.close()

        # PR 曲线
        pr, rc, _ = precision_recall_curve(y_true, probs)
        plt.figure()
        plt.plot(rc, pr)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.savefig(os.path.join(save_prefix, "pr_curve.png"), dpi=300)
        plt.close()

        # 混淆矩阵
        preds = (probs > 0.5).astype(int)
        cm = confusion_matrix(y_true, preds)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.savefig(os.path.join(save_prefix, "confusion_matrix.png"), dpi=300)
        plt.close()

        # TopN 柱状图
        plt.figure()
        plt.bar(["Top@1", "Top@3", "Top@5"], [metrics["Top1"], metrics["Top3"], metrics["Top5"]])
        plt.ylim(0, 1)
        plt.title("Top-N Accuracy")
        plt.savefig(os.path.join(save_prefix, "topn.png"), dpi=300)
        plt.close()
        # ========= Per-file Evaluation for Significance Tests ==========
        per_file = []  # list of dicts: [{"file_id": xx, "top1": xx, "rank": xx, "ap": xx}, ...]

        from sklearn.metrics import average_precision_score

        file_counter = 0
        for logits, labels, masks in zip(all_logits, all_labels, all_masks):
            B = logits.size(0)
            for i in range(B):
                valid_len = int(masks[i].sum().item())
                if valid_len == 0:
                    continue

                # 取当前文件的数据
                prob = torch.sigmoid(logits[i, :valid_len]).cpu().numpy()
                lab = labels[i, :valid_len].cpu().numpy()

                bug_lines = [j for j, v in enumerate(lab) if v == 1]
                if not bug_lines:
                    continue  # 无缺陷文件可忽略，或记录为 None

                # ===== per-file Top-k =====
                order = prob.argsort()[::-1]
                top1 = int(any(b in order[:1] for b in bug_lines))
                top3 = int(any(b in order[:3] for b in bug_lines))
                top5 = int(any(b in order[:5] for b in bug_lines))
                top10 = int(any(b in order[:10] for b in bug_lines))

                # ===== rank =====
                # 取第一个真实缺陷的 rank
                ranks = [int(np.where(order == b)[0][0]) for b in bug_lines]
                rank = min(ranks)+1

                # ===== AP =====
                ap = average_precision_score(lab, prob)

                per_file.append({
                    "file_id": file_counter,
                    "top1": top1,
                    "top3": top3,
                    "top5": top5,
                    "top10": top10,
                    "rank": rank,
                    "ap": float(ap),
                })
                file_counter += 1

        # 保存：用于统计显著性检验
        with open(os.path.join(save_prefix, "per_file_metrics.json"), "w") as f:
            json.dump(per_file, f, indent=2)

        np.savez_compressed(os.path.join(save_prefix, "probs_labels.npz"), probs=probs, labels=y_true)
        with open(os.path.join(save_prefix, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved evaluation results at {save_prefix}")

    return metrics

def train(args):
    # ---------- Setup ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Output Structure
    os.makedirs(args.save_dir, exist_ok=True)
    best_dir = os.path.join(args.save_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # ---------- Data ----------
    train_set = TokenLineDataset(args.train_feature_dir)
    valid_set = TokenLineDataset(args.valid_feature_dir)
    test_set = None
    if getattr(args, "test_feature_dir", None):
        test_set = TokenLineDataset(args.test_feature_dir)
        logger.info(f"Loaded test set from {args.test_feature_dir}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if test_set else None

    # ---------- Model ----------
    if args.model_type == "transformer":
        model = LineAdapter(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout
        ).to(device)
    else:
        model = MLPAdapter(input_dim=args.input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with open(os.path.join(args.save_dir, "model_stats.json"), "w") as f:
        json.dump({"total_params": total_params, "trainable_params": trainable_params}, f, indent=2)

    logger.info(f"Model params: {trainable_params/1e6:.2f}M trainable / {total_params/1e6:.2f}M total")

    # ---------- Optimizer ----------
    criterion = FocalLoss(alpha=0.9, gamma=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    # ---------- Multi-metric Best Save Initialization ----------
    save_metrics = ["F2", "Precision", "Recall", "AUC", "Top1", "Top3", "Top5"]
    best_ckpts = {m: {"score": -1e9, "path": None} for m in save_metrics}

    TOP_N = 5
    topN_queue = []   # (score, epoch, path)

    # ---------- Logs ----------
    metrics_log = []
    start_time = time.time()

    # ==========================================================
    #                         Training Loop
    # ==========================================================
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---- Train ----
        for feats, labels, masks in tqdm(train_loader, desc=f"Epoch {epoch}"):
            feats, labels, masks = feats.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                logits = model(feats, masks)
                loss = criterion(logits[masks.bool()], labels[masks.bool()])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - t0
        gpu_peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

        logger.info(f"Epoch {epoch} | train_loss={avg_loss:.4f} | time={epoch_time:.2f}s | GPU={gpu_peak:.2f}GB")

        # ---- Validation ----
        val_metrics = evaluate_full(model, valid_loader, device, save_prefix=None)
        val_metrics["train_loss"] = avg_loss

        # ---- Logging ----
        epoch_log = {"epoch": epoch, "epoch_time_s": epoch_time, "gpu_peak_gb": gpu_peak, **val_metrics}
        metrics_log.append(epoch_log)
        logger.info(f"[Valid] {val_metrics}")

        # ======================================================
        #            Multi-metric Best Model Saving
        # ======================================================
        for m in save_metrics:
            if m in val_metrics:
                current = val_metrics[m]
                if current > best_ckpts[m]["score"]:
                    best_ckpts[m]["score"] = current
                    save_path = os.path.join(best_dir, f"best_{m}.pt")
                    torch.save(model.state_dict(), save_path)
                    best_ckpts[m]["path"] = save_path
                    logger.info(f"[BEST-{m}] Updated: {current:.4f} @ epoch {epoch}")

        # ---- Top-N checkpoint saving ----
        ref_metric = "AUC" if "AUC" in val_metrics else "F2"
        ref_score = val_metrics[ref_metric]
        ckpt_path = os.path.join(best_dir, f"ckpt_epoch{epoch:03d}_{ref_metric}{ref_score:.4f}.pt")
        torch.save(model.state_dict(), ckpt_path)
        topN_queue.append((ref_score, epoch, ckpt_path))
        topN_queue = sorted(topN_queue, key=lambda x: x[0], reverse=True)[:TOP_N]

        # Remove outdated ckpts
        for _, _, path in topN_queue[TOP_N:]:
            if os.path.exists(path):
                os.remove(path)

        # Save best summary
        with open(os.path.join(best_dir, "best_ckpts.json"), "w") as f:
            json.dump({k: v["score"] for k, v in best_ckpts.items()}, f, indent=2)

        torch.cuda.empty_cache()
        gc.collect()

    # ==========================================================
    #                  Multi-Checkpoint Testing
    # ==========================================================
    if test_loader:
        logger.info("===== Testing All Best Checkpoints =====")

        best_model_files = [
            f for f in os.listdir(best_dir)
            if f.startswith("best_") and f.endswith(".pt")
        ]

        all_tests = {}
        for ckpt in best_model_files:
            ckpt_path = os.path.join(best_dir, ckpt)
            logger.info(f"[TEST] Evaluating {ckpt}")

            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            save_prefix = os.path.join(best_dir, f"test_{ckpt.replace('.pt','')}")
            test_metrics = evaluate_full(model, test_loader, device, save_prefix)

            all_tests[ckpt] = test_metrics
            with open(save_prefix + "_metrics.json", "w") as f:
                json.dump(test_metrics, f, indent=4)

        with open(os.path.join(best_dir, "all_test_results.json"), "w") as f:
            json.dump(all_tests, f, indent=4)

        logger.info("===== Multi-checkpoint Testing Done =====")

    # ==========================================================
    #                           END
    # ==========================================================
    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time/60:.2f} min")

    with open(os.path.join(log_dir, "metrics_log.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    with open(os.path.join(log_dir, "resource_usage.json"), "w") as f:
        json.dump({"total_train_time_s": total_time}, f, indent=2)

    logger.info(f"Logs saved under: {args.save_dir}")


# ====================================================
# CLI
# ====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train line-level Adapter (科研版)")
    parser.add_argument("--train_feature_dir", type=str, required=True)
    parser.add_argument("--valid_feature_dir", type=str, required=True)
    parser.add_argument("--test_feature_dir", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=["transformer", "mlp"], default="transformer")
    parser.add_argument("--save_dir", type=str, default="results/adapter")
    parser.add_argument("--input_dim", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    # parser.add_argument("--use_lora", action="store_true")
    # parser.add_argument("--lora_rank", type=int, default=8)
    # parser.add_argument("--early_stop", action="store_true")  # 保留兼容参数
    args = parser.parse_args()
    train(args)
