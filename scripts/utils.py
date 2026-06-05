"""
工具函数：数据集加载、格式转换、文本处理
"""
import json
import os
import random
from pathlib import Path


def load_json(filepath: str | Path) -> list[dict]:
    """加载 JSON 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict], filepath: str | Path) -> None:
    """保存为 JSON 文件"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(data)} 条数据到 {filepath}")


def format_alpaca(instruction: str, input_text: str, output: str) -> dict:
    """构造一条 Alpaca 格式的训练样本"""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def format_sharegpt(messages: list[dict]) -> dict:
    """
    构造一条 ShareGPT 格式的训练样本
    messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    return {"messages": messages}


def convert_hf_to_alpaca(
    hf_dataset,
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
) -> list[dict]:
    """将 HuggingFace 数据集转换为 Alpaca 格式"""
    result = []
    for item in hf_dataset:
        result.append({
            "instruction": item.get(instruction_key, ""),
            "input": item.get(input_key, ""),
            "output": item.get(output_key, ""),
        })
    return result


def train_test_split(
    data: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """将数据随机切分为训练集和测试集"""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def load_hf_dataset(dataset_name: str, split: str = "train", max_samples: int | None = None):
    """从 HuggingFace 加载数据集"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请先安装 datasets: pip install datasets")

    ds = load_dataset(dataset_name, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def load_dataset_modelscope(dataset_name: str, max_samples: int | None = None):
    """从 ModelScope 加载数据集（国内更快）"""
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        raise ImportError("请先安装 modelscope: pip install modelscope")

    ds = MsDataset.load(dataset_name)
    if max_samples:
        ds = ds["train"].take(max_samples)
    return ds


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """将长文本按段落切分"""
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 对过长的段落按句子进一步切分
            sentences = para.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n")
            current_chunk = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(current_chunk) + len(sent) <= chunk_size:
                    current_chunk += sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
    return chunks


def validate_alpaca_format(data: list[dict]) -> list[str]:
    """验证 Alpaca 格式数据，返回错误列表"""
    errors = []
    for i, item in enumerate(data):
        for key in ["instruction", "input", "output"]:
            if key not in item:
                errors.append(f"第 {i} 条缺少字段 '{key}'")
            elif not isinstance(item.get(key), str):
                errors.append(f"第 {i} 条字段 '{key}' 不是字符串")
        if not item.get("instruction", "").strip() and not item.get("input", "").strip():
            errors.append(f"第 {i} 条 instruction 和 input 都为空")
        if not item.get("output", "").strip():
            errors.append(f"第 {i} 条 output 为空")
    return errors
