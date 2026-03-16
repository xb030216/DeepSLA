***

```markdown
# DeepSLA: Long-Context Aware Line-Level Vulnerability Detection

> **Note for Reviewers:** This repository is completely anonymized for the double-blind review process. It contains the core implementation of **DeepSLA**, a novel "Freeze-Align-Adapt" paradigm for line-level fault localization using Large Language Models (LLMs).

DeepSLA bridges the sequence-structure granularity mismatch between generative LLMs and discriminative line-level fault localization tasks. It achieves highly accurate and extremely fast (batched) inference on arbitrarily long code files without suffering from catastrophic forgetting.

## 1. Environment Setup

It is recommended to use a Linux environment with CUDA support.
We require Python 3.8+ and PyTorch 2.0+.

```bash
# Create a virtual environment
conda create -n deepsla python=3.10
conda activate deepsla

# Install PyTorch (Please adjust the CUDA version according to your hardware)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers accelerate datasets pandas scikit-learn matplotlib tqdm
```

*(Optional)* To load models in 8-bit for GPU memory reduction, install `bitsandbytes`:
```bash
pip install bitsandbytes
```

## 2. Data Preparation

DeepSLA expects the dataset to be in `.csv` format. 
Your CSV files (e.g., `train.csv`, `valid.csv`, `test.csv`) must contain at least the following two columns:
*   `buggyCode`: The full source code of the buggy file as a plain string.
*   `bugLineNum`: The line numbers containing the bug. It should be a stringified list or comma-separated integers (e.g., `"[12, 13]"` or `12, 13`). Note that line numbers are 1-indexed.

## 3. Pipeline Execution

DeepSLA follows a two-stage "Freeze-Align-Adapt" pipeline:
1. **Feature Extraction (Freeze & Align):** Extracting deep semantic features from the frozen LLM using our offset-mapping based long-context segmenter.
2. **Adapter Training (Adapt):** Training a lightweight context-aware adapter to predict vulnerability probabilities per line.

### Stage 1: Token & Line-Level Feature Extraction

The `token_line_features_longcode.py` script efficiently processes long files by character-aligned segmentation, extracting hidden states, and mapping them to source code lines.

```bash
python token_line_features_longcode.py \
    --model_name "deepseek-ai/deepseek-coder-6.7b-base" \
    --csv_path data/train.csv \
    --save_dir features/train_features \
    --chunk_size 400 \
    --num_last_layers 8 \
    --segment_max_tokens 4096 \
    --segment_stride_tokens 256 \
    --device cuda \
    --progress
```

**Key Arguments:**
*   `--model_name`: The HuggingFace model ID (e.g., DeepSeek-Coder, CodeLlama).
*   `--csv_path`: Path to your input CSV file.
*   `--save_dir`: Output directory to save the extracted `.pt` feature tensors.
*   `--num_last_layers`: Number of middle/last hidden layers to fuse. By default, it extracts syntactic-rich middle layers (e.g., layers 12-19).
*   `--segment_max_tokens` & `--segment_stride_tokens`: Controls the sliding window for extremely long files, ensuring OOM errors are avoided while maintaining cross-segment semantic context.

*(Note: Repeat Stage 1 for your validation and test datasets, saving them to `features/valid_features` and `features/test_features` respectively).*

### Stage 2: Context-Aware Adapter Training

Once features are extracted, train the lightweight line-level adapter using `train_lastest.py`. This script handles class imbalance using Focal Loss and evaluates the model dynamically.

```bash
python train_lastest.py \
    --train_feature_dir features/train_features \
    --valid_feature_dir features/valid_features \
    --test_feature_dir features/test_features \
    --model_type transformer \
    --input_dim 4096 \
    --hidden_dim 1024 \
    --num_layers 2 \
    --batch_size 4 \
    --epochs 20 \
    --lr 5e-5 \
    --save_dir results/deepsla_adapter
```

**Key Arguments:**
*   `--input_dim`: The hidden size of your base LLM (e.g., `4096` for 7B models, `768` for CodeBERT).
*   `--model_type`: Choose between `transformer` (default, with Positional Encoding for cross-line dependencies) or `mlp`.
*   `--batch_size`: Batch size of code files (not lines).

## 4. Evaluation & Results Output

During and after training, the script automatically evaluates the model on multiple metrics crucial for fault localization:
*   **Ranking Metrics:** Top@1, Top@3, Top@5, Rank, and Average Precision (AP).
*   **Classification Metrics:** F2-Score, Precision, Recall, AUC, and MCC.

All evaluation artifacts are automatically saved in the `--save_dir` (e.g., `results/deepsla_adapter/best_model/`), including:
1.  **Saved Checkpoints:** `.pt` files for the best models across different metrics.
2.  **Visualizations:** 
    *   `roc_curve.png` (ROC curve)
    *   `pr_curve.png` (Precision-Recall curve)
    *   `confusion_matrix.png`
    *   `topn.png` (Top-N Accuracy bar chart)
3.  **Statistical Files:** `metrics.json` and `per_file_metrics.json` for significance testing.
```
