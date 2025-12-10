import json
import re
from tqdm import tqdm
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================
# 1. 加行号
# ============================================

def add_line_numbers(code: str) -> str:
    lines = code.split("\n")
    return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))


# ============================================
# 2. Prompt
# ============================================

def build_zero_shot_prompt(code_with_lines: str) -> str:
    return f"""
You are a professional security auditor specializing in vulnerability localization.
Given the source code with line numbers, identify the Top-5 most suspicious lines.

Rules:
1. Think briefly (one or two short sentences).
2. Then output EXACTLY one JSON object:
   {{"buggy_lines": [line1, line2, line3, line4, line5]}}
3. The JSON must be the last thing in your response.
4. Always output 5 integers.

---CODE_START---
{code_with_lines}
---CODE_END---

Now analyze the code and output the JSON:
""".strip()


# ============================================
# 3. 解析模型输出（提取 JSON）
# ============================================

def extract_json_from_text(text: str):
    m = re.search(r'(\{.*\})\s*$', text, flags=re.DOTALL)
    if m:
        cand = m.group(1)
    else:
        m = re.search(r'(\{.*\})', text, flags=re.DOTALL)
        if m:
            cand = m.group(1)
        else:
            return None

    try:
        return json.loads(cand)
    except:
        s = cand
        s = re.sub(r',\s*}', '}', s)
        s = re.sub(r',\s*\]', ']', s)
        s = s.replace("'", '"')
        try:
            return json.loads(s)
        except:
            return None


def parse_buggy_lines(j: Dict[str, Any]):
    if not j or "buggy_lines" not in j:
        return []
    try:
        return [int(x) for x in j["buggy_lines"]][:5]
    except:
        return []


# ============================================
# 4. 调用本地 DeepSeek 模型
# ============================================

def local_generate(model, tokenizer, prompt: str, max_new_tokens=512):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()  # 只取生成部分


# ============================================
# 5. top-k 评估
# ============================================

def top_k_hit(pred: List[int], gt: List[int], k: int):
    pred_k = pred[:k]
    return int(any(t in pred_k for t in gt))


# ============================================
# 6. 主入口
# ============================================

def zero_shot_eval_local(jsonl_path: str, model_dir: str):
    # 加载模型
    print("Loading model:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Model loaded.")

    # 加载数据
    samples = []
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            samples.append(json.loads(line))

    total = len(samples)
    top1 = top3 = top5 = 0
    parse_fail = 0

    for sample in tqdm(samples):
        code = sample["code"]
        gt = sample["vuln_lines"]

        code_with_lines = add_line_numbers(code)
        prompt = build_zero_shot_prompt(code_with_lines)

        raw_out = local_generate(model, tokenizer, prompt)

        parsed = extract_json_from_text(raw_out)
        pred_lines = parse_buggy_lines(parsed)

        if len(pred_lines) == 0:
            parse_fail += 1

        top1 += top_k_hit(pred_lines, gt, 1)
        top3 += top_k_hit(pred_lines, gt, 3)
        top5 += top_k_hit(pred_lines, gt, 5)

    print("=======================================")
    print("Zero-shot results:")
    print("Total samples:", total)
    print("Parse fail:", parse_fail, f"({parse_fail/total*100:.2f}%)")
    print("Top-1:", top1 / total)
    print("Top-3:", top3 / total)
    print("Top-5:", top5 / total)
    print("=======================================")


if __name__ == "__main__":
    #  Example:
    #  python zero_shot_eval_local.py
    zero_shot_eval_local(
        jsonl_path="test_zero_shot.jsonl",                 # 你的数据集
        model_dir="deepseek-ai/deepseek-coder-6.7b-instruct"  # 你本地模型的路径
    )
