"""
从 HuggingFace 下载数据集并转换为 LLaMA-Factory 可用的 Alpaca JSON 格式。

用法:
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --dataset shibing624/alpaca-zh --subset 5000
  python scripts/prepare_dataset.py --source modelscope  # 国内 ModelScope 下载

输出:
  data/train.json  +  data/eval.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.utils import (
    convert_hf_to_alpaca,
    load_hf_dataset,
    save_json,
    train_test_split,
    validate_alpaca_format,
)

DATASET_NAME = "shibing624/alpaca-zh"


def download_from_hf(dataset_name: str, max_samples: int | None = None):
    """从 HuggingFace 下载数据集（国内需要 HF_ENDPOINT=https://hf-mirror.com）"""
    print(f"从 HuggingFace 下载: {dataset_name}")
    ds = load_hf_dataset(dataset_name, max_samples=max_samples)
    return convert_hf_to_alpaca(ds)


def download_from_modelscope(dataset_name: str, max_samples: int | None = None):
    """从 ModelScope 下载数据集（国内更快，不需要代理）"""
    from modelscope.msdatasets import MsDataset

    print(f"从 ModelScope 下载: {dataset_name}")
    ds = MsDataset.load(dataset_name, subset_name="default")
    data = []
    for item in ds["train"]:
        data.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
        })
        if max_samples and len(data) >= max_samples:
            break
    return data


def main():
    parser = argparse.ArgumentParser(description="准备 LoRA 微调数据集")
    parser.add_argument("--dataset", default=DATASET_NAME, help="HuggingFace 数据集名称")
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface",
                        help="下载源（国内建议 modelscope）")
    parser.add_argument("--subset", type=int, default=None, help="限制样本数量（用于快速实验）")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), "..", args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.source == "modelscope":
        data = download_from_modelscope(args.dataset, max_samples=args.subset)
    else:
        data = download_from_hf(args.dataset, max_samples=args.subset)

    print(f"共获取 {len(data)} 条数据")

    # 验证格式
    errors = validate_alpaca_format(data)
    if errors:
        print(f"格式验证发现 {len(errors)} 个问题（前10个）：")
        for e in errors[:10]:
            print(f"  - {e}")
    else:
        print("格式验证通过")

    # 切分
    train_data, eval_data = train_test_split(data, test_ratio=args.test_ratio, seed=args.seed)
    print(f"训练集: {len(train_data)} 条, 验证集: {len(eval_data)} 条")

    # 保存
    train_path = os.path.join(output_dir, "train.json")
    eval_path = os.path.join(output_dir, "eval.json")
    save_json(train_data, train_path)
    save_json(eval_data, eval_path)

    print("完成！数据已保存到 data/ 目录。")


if __name__ == "__main__":
    main()
