# DeepSLA

DeepSLA is a two-stage pipeline for long-context, line-level code localization.
This repository currently contains:

- long-context feature extraction for `csv` and `jsonl` inputs
- adapter training on extracted `.pt` features
- ablation and baseline scripts used in the study

## 1. Repository Overview

Main scripts:

- `token_line_features_longcode.py`
  CSV-based feature extractor for Stage 1.
- `token_line_features_longcode_jsonl.py`
  JSONL-based feature extractor for Stage 1.
- `token_no_offset.py`
  no-offset ablation extractor with the same feature output style.
- `train_lastest.py`
  Stage 2 adapter training and evaluation.
- `baseline_model/codebert_finetune.py`
  CodeBERT baseline on JSONL datasets.
- `baseline_model/unixcoder_baseline.py`
  UniXcoder-style baseline on JSONL datasets.
- `baseline_model/logistic_baseline.py`
  shallow-feature logistic regression baseline on JSONL datasets.
- `baseline_model/zero_shot_topk.py`
  zero-shot prompting baseline.

Example data and config files:

- `data/train.csv`, `data/val.csv`, `data/test.csv`
- `baseline_model/dataset_hf/train.jsonl`, `val.jsonl`, `test.jsonl`
- `configs/train.json`
- `configs/val_jsonl.json`

## 2. Environment Setup

Recommended:

- Python 3.9+
- CUDA-enabled PyTorch
- Linux or Windows with a working GPU environment

Example setup:

```bash
conda create -n deepsla python=3.10
conda activate deepsla

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pandas scikit-learn matplotlib tqdm numpy
```

Optional packages:

- `bitsandbytes`
  for `--use_8bit` model loading
- `peft`
  not required by the current default training flow, but imported optionally in `train_lastest.py`

```bash
pip install bitsandbytes peft
```

## 3. Supported Data Formats

### 3.1 CSV format

Used by:

- `token_line_features_longcode.py`
- `token_no_offset.py`

Required columns:

- `buggyCode`
  full source code as one string
- `bugLineNum`
  buggy line numbers, 1-based

Accepted `bugLineNum` styles:

- `"[12, 13]"`
- `"12,13"`
- `"12, 13"`

### 3.2 JSONL format

Used by:

- `token_line_features_longcode_jsonl.py`
- scripts under `baseline_model/`

Recommended fields per line:

- `id`
- `code`
- `lines`
- `line_labels`
- `num_lines`

Practical requirements for the extractor:

- `code` is preferred
- if `code` is missing, the extractor can rebuild code from `lines`
- `line_labels` can be a binary per-line list such as `[0, 1, 0, ...]`
- `line_labels` can also be a direct list of buggy line numbers

The JSONL examples in `baseline_model/dataset_hf/` already match this layout.

## 4. Stage 1: Feature Extraction

Stage 1 uses a frozen code LLM to produce token-level hidden states and aligned line-level metadata for each source file.

### 4.1 CSV extractor

Direct CLI example:

```bash
python token_line_features_longcode.py \
  --model_name deepseek-ai/deepseek-coder-6.7b-instruct \
  --csv_path data/train.csv \
  --save_dir features/train_features \
  --chunk_size 400 \
  --num_last_layers 8 \
  --segment_max_tokens 4096 \
  --segment_stride_tokens 256 \
  --progress
```

Config-driven example:

```bash
python token_line_features_longcode.py --config configs/train.json
```

Override config values from the command line:

```bash
python token_line_features_longcode.py \
  --config configs/train.json \
  --csv_path data/val.csv \
  --save_dir features/val_features
```

### 4.2 JSONL extractor

Direct CLI example:

```bash
python token_line_features_longcode_jsonl.py \
  --model_name deepseek-ai/deepseek-coder-6.7b-instruct \
  --jsonl_path baseline_model/dataset_hf/val.jsonl \
  --save_dir features_jsonl/val_features \
  --chunk_size 400 \
  --num_last_layers 8 \
  --segment_max_tokens 4096 \
  --segment_stride_tokens 256 \
  --progress
```

Config-driven example:

```bash
python token_line_features_longcode_jsonl.py --config configs/val_jsonl.json
```

### 4.3 Important extractor arguments

- `--model_name`
  Hugging Face model name or local model path
- `--csv_path` / `--jsonl_path`
  input dataset path
- `--save_dir`
  output directory for chunked feature files
- `--chunk_size`
  number of source files per saved chunk
- `--num_last_layers`
  number of hidden layers fused in the extractor
- `--segment_max_tokens`
  token limit of each long-context segment
- `--segment_stride_tokens`
  overlap between neighboring segments
- `--max_chars_per_segment`
  safety cap during segment search
- `--use_8bit`
  enable 8-bit model loading when available
- `--start_chunk`
  resume from a specific chunk index
- `--chunks_limit`
  process only a fixed number of chunks
- `--print_config`
  print the effective configuration
- `--dump_config`
  write the effective configuration to a JSON file

### 4.4 Extractor outputs

Each `save_dir` contains:

- `chunk_0000.pt`, `chunk_0001.pt`, ...
- `manifest.json`
- `extract_stats.json`
- `effective_config.json`
- `failed_samples.json`
  only present if some samples fail

Each saved sample record may contain:

- `token_hidden`
- `line_spans`
- `token_offsets`
- `line_labels`
- `num_lines`
- `meta`
- `line_features`
  optional

Important compatibility note:

- `train_lastest.py` currently loads `token_hidden` and recomputes per-line vectors from `line_spans`
- if you enable `--quantize_int8`, the extractor may save `token_hidden_int8` instead of `token_hidden`
- the current training script does not read `token_hidden_int8`

So for the default DeepSLA training pipeline, keep:

- `--keep_token_hidden`
- `--quantize_int8` disabled

## 5. Stage 2: Adapter Training

After extracting features for train, validation, and optionally test sets, train the lightweight adapter with `train_lastest.py`.

Example:

```bash
python train_lastest.py \
  --train_feature_dir features/train_features \
  --valid_feature_dir features/val_features \
  --test_feature_dir features/test_features \
  --model_type transformer \
  --input_dim 4096 \
  --hidden_dim 1024 \
  --num_layers 2 \
  --num_heads 8 \
  --dropout 0.3 \
  --batch_size 4 \
  --epochs 20 \
  --lr 5e-5 \
  --save_dir results/deepsla_adapter
```

Main training arguments:

- `--train_feature_dir`
- `--valid_feature_dir`
- `--test_feature_dir`
  optional but recommended
- `--model_type`
  `transformer` or `mlp`
- `--input_dim`
  hidden size of the extracted base model
- `--hidden_dim`
- `--num_layers`
- `--num_heads`
- `--dropout`
- `--batch_size`
- `--epochs`
- `--lr`
- `--save_dir`

Expected `input_dim` examples:

- `4096` for DeepSeek-Coder 6.7B style hidden states
- use the hidden size of the actual backbone if you change models

### 5.1 Training outputs

`train_lastest.py` creates:

- `results/.../best_model/`
  best checkpoints and test-time evaluation folders
- `results/.../logs/metrics_log.json`
- `results/.../logs/resource_usage.json`

During evaluation, the script also saves:

- `metrics.json`
- `per_file_metrics.json`
- `probs_labels.npz`
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `topn.png`

## 6. Typical End-to-End Workflow

### 6.1 CSV pipeline

```bash
python token_line_features_longcode.py --config configs/train.json

python token_line_features_longcode.py \
  --config configs/train.json \
  --csv_path data/val.csv \
  --save_dir features/val_features

python token_line_features_longcode.py \
  --config configs/train.json \
  --csv_path data/test.csv \
  --save_dir features/test_features

python train_lastest.py \
  --train_feature_dir features/train_features \
  --valid_feature_dir features/val_features \
  --test_feature_dir features/test_features \
  --model_type transformer \
  --input_dim 4096 \
  --save_dir results/deepsla_adapter
```

### 6.2 JSONL pipeline

```bash
python token_line_features_longcode_jsonl.py \
  --config configs/val_jsonl.json \
  --jsonl_path baseline_model/dataset_hf/train.jsonl \
  --save_dir features_jsonl/train_features

python token_line_features_longcode_jsonl.py \
  --config configs/val_jsonl.json \
  --jsonl_path baseline_model/dataset_hf/val.jsonl \
  --save_dir features_jsonl/val_features

python token_line_features_longcode_jsonl.py \
  --config configs/val_jsonl.json \
  --jsonl_path baseline_model/dataset_hf/test.jsonl \
  --save_dir features_jsonl/test_features

python train_lastest.py \
  --train_feature_dir features_jsonl/train_features \
  --valid_feature_dir features_jsonl/val_features \
  --test_feature_dir features_jsonl/test_features \
  --model_type transformer \
  --input_dim 4096 \
  --save_dir results/deepsla_adapter_jsonl
```

## 7. Ablation and Baseline Scripts

### 7.1 No-offset ablation

`token_no_offset.py` is the no-offset extractor used for ablation.
Its output format matches the Stage 1 feature format and can be fed into `train_lastest.py` in the same way as the main extractor.

### 7.2 CodeBERT baseline

Dataset format:

- `baseline_model/dataset_hf/train.jsonl`
- `baseline_model/dataset_hf/val.jsonl`
- `baseline_model/dataset_hf/test.jsonl`

Run:

```bash
python baseline_model/codebert_finetune.py \
  --dataset_dir baseline_model/dataset_hf \
  --output_dir results/codebert \
  --model_name microsoft/codebert-base \
  --epochs 5 \
  --batch 8 \
  --max_len 512 \
  --stride 256
```

### 7.3 UniXcoder baseline

Run:

```bash
python baseline_model/unixcoder_baseline.py \
  --dataset_dir baseline_model/dataset_hf \
  --output_dir results/unixcoder \
  --epochs 5 \
  --batch 8 \
  --max_len 512 \
  --stride 256 \
  --use_linevul
```

### 7.4 Logistic regression baseline

Run:

```bash
python baseline_model/logistic_baseline.py \
  --train baseline_model/dataset_hf/train.jsonl \
  --test baseline_model/dataset_hf/test.jsonl
```

### 7.5 Zero-shot baseline

`baseline_model/zero_shot_topk.py` currently uses a hard-coded entry point in `__main__`.
Before running it, update the dataset path and model path inside the file.
Also note that the script currently expects a `vuln_lines` field, so it is not plug-and-play with `baseline_model/dataset_hf/*.jsonl` unless you adapt the field mapping.

## 8. Notes and Practical Tips

- Use the same backbone and hidden size across train, validation, and test feature extraction.
- If you change the backbone, update `--input_dim` in `train_lastest.py`.
- Use `--start_chunk` to resume interrupted extraction jobs.
- Check `effective_config.json` in every feature directory before training.
- If you are extracting features only for archival or inspection, `line_features` can be saved, but the current training script does not require it.
- The JSONL extractor is meant for datasets shaped like `baseline_model/dataset_hf/*.jsonl`.

## 9. Minimal Sanity Check

Before launching a full experiment, it is a good idea to run:

```bash
python token_line_features_longcode.py \
  --config configs/train.json \
  --max_samples 2 \
  --save_dir tmp_features_csv

python token_line_features_longcode_jsonl.py \
  --config configs/val_jsonl.json \
  --max_samples 2 \
  --save_dir tmp_features_jsonl
```

Then verify that the output directories contain:

- at least one `chunk_*.pt`
- `manifest.json`
- `extract_stats.json`
- `effective_config.json`
