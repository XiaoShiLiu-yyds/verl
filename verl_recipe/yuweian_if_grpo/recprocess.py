#!/usr/bin/env python3
# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""把 yuweian 指令遵循(IF)数据集 JSON 转成 verl 使用的 parquet 格式。

对 GRPO 只有 3 个字段有用（其余忽略）：
  - conversations[0].content  -> prompt（提示词）
  - code_checker              -> 硬约束：一组 `def check(response_text)` 源码（可执行）
  - llm_checker               -> 软约束：一组自然语言约束描述（需 LLM 判定）

输出 parquet schema（参考 verl/examples/data_preprocess/geo3k.py）：
  data_source : str
  prompt      : list[{role, content}]
  reward_model: {style, ground_truth}      # ground_truth 不用，约束都在 extra_info
  extra_info  : {id, code_checker[], llm_checker[], num_hard, num_soft}

用法：
  python preprocess.py \
      --input /path/to/data_english_train_gemini3_add_soft_clean_1_cleaned.json \
      --output_dir ./data \
      --val_size 1000
"""

import argparse
import json
import os

from datasets import Dataset


def build_record(item, data_source):
    """从原始 JSON 条目构造一条 parquet 记录；不合法返回 None。"""
    conversations = item.get("conversations") or []
    if not conversations or not isinstance(conversations, list):
        return None
    first = conversations[0]
    if not isinstance(first, dict) or first.get("role") != "user":
        return None
    content = first.get("content")
    if not isinstance(content, str) or not content.strip():
        return None

    code_checker = item.get("code_checker") or []
    llm_checker = item.get("llm_checker") or []
    # 保证是 list[str]
    code_checker = [c for c in code_checker if isinstance(c, str) and c.strip()]
    llm_checker = [c for c in llm_checker if isinstance(c, str) and c.strip()]
    # 两类约束都为空的样本对 GRPO 无信号，跳过
    if not code_checker and not llm_checker:
        return None

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": content}],
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "id": item.get("id"),
            "code_checker": code_checker,
            "llm_checker": llm_checker,
            "num_hard": len(code_checker),
            "num_soft": len(llm_checker),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="原始 JSON 文件路径（顶层数组）。",
    )
    parser.add_argument(
        "--output_dir",
        default="./data",
        help="parquet 输出目录（生成 train.parquet / val.parquet）。",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=1000,
        help="验证集大小（取末尾若干条）；设为 0 则 train==val。",
    )
    parser.add_argument(
        "--data_source",
        default="yuweian_if",
        help="写入 parquet 的 data_source 字段。",
    )
    args = parser.parse_args()

    print(f"[preprocess] loading {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"[preprocess] raw items: {len(raw)}")

    records = []
    skipped = 0
    for item in raw:
        rec = build_record(item, args.data_source)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
    print(f"[preprocess] valid records: {len(records)}, skipped: {skipped}")

    if not records:
        raise RuntimeError("No valid records after filtering. Check the input JSON.")

    val_size = min(args.val_size, max(0, len(records) - 1))
    if val_size <= 0:
        train_records, val_records = records, records
    else:
        train_records, val_records = records[:-val_size], records[-val_size:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")

    Dataset.from_list(train_records).to_parquet(train_path)
    Dataset.from_list(val_records).to_parquet(val_path)
    print(f"[preprocess] wrote train: {train_path} ({len(train_records)} rows)")
    print(f"[preprocess] wrote val  : {val_path} ({len(val_records)} rows)")

    # 抽查 round-trip
    import datasets as _ds

    chk = _ds.load_dataset("parquet", data_files=train_path, split="train")
    ex = chk[0]["extra_info"]
    print(
        "[preprocess] sanity-check extra_info: "
        f"id={ex.get('id')} num_hard={ex.get('num_hard')} num_soft={ex.get('num_soft')} "
        f"code_checker[0][:60]={ex['code_checker'][0][:60]!r}"
    )


if __name__ == "__main__":
    main()
