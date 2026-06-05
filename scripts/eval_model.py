"""
模型评估脚本 — 在微调完成后测试知识掌握程度

用法：
    python eval_model.py --base-model Qwen/Qwen2.5-1.5B-Instruct \
                         --lora-path ./output/checkpoint-best \
                         --test-data ../data/eval.json

支持的评估方式：
    - 精确匹配: answer in response
    - 语义相似: 用 embedding 计算相似度
    - 人工判断: 逐条打印让人工打分
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_json


def load_model_and_tokenizer(base_model: str, lora_path: str | None = None):
    """加载基础模型和可选的 LoRA 权重"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        raise ImportError("请安装依赖: pip install torch transformers peft")

    print(f"加载基础模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("加载模型权重...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path:
        print(f"加载 LoRA 权重: {lora_path}")
        model = PeftModel.from_pretrained(base, lora_path)
        model = model.merge_and_unload()
    else:
        model = base

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, input_text: str, max_new_tokens: int = 256) -> str:
    """生成模型回复"""
    prompt = f"{instruction}\n{input_text}" if instruction else input_text

    # Qwen 风格 prompt
    messages = [
        {"role": "system", "content": "你是一个专业的知识助手，请准确、简洁地回答用户的问题。如果你不确定答案，请诚实说明。"},
        {"role": "user", "content": prompt},
    ]

    # 尝试使用 chat_template
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # fallback: 简单拼接
        formatted = f"<|im_start|>system\n你是一个专业的知识助手，请准确、简洁地回答用户的问题。如果你不确定答案，请诚实说明。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(formatted, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate_precision(test_data: list[dict], model, tokenizer, output_file: str | None = None) -> dict:
    """评估：基于规则检查答案是否包含在回复中"""
    results = []
    correct = 0
    total = len(test_data)

    for i, item in enumerate(test_data):
        response = generate_response(model, tokenizer, item["instruction"], item.get("input", ""))
        expected = item["output"].strip()

        # 简单判断：期望答案的关键内容是否在回复中
        # 对期望答案取关键子串
        keywords = _extract_keywords(expected)
        matched = any(kw in response for kw in keywords) if keywords else expected[:20] in response

        results.append({
            "index": i,
            "question": item.get("input", item["instruction"]),
            "expected": expected,
            "response": response,
            "correct": matched,
        })

        if matched:
            correct += 1

        if i < 3 or i % 20 == 0:  # 打印前 3 条和每 20 条进度
            print(f"[{i+1}/{total}] {'✓' if matched else '✗'} Q: {item.get('input', '')[:60]}...")
            if not matched:
                print(f"  期望: {expected[:100]}...")
                print(f"  回复: {response[:100]}...")

    accuracy = correct / total if total > 0 else 0

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {output_file}")

    return summary


def _extract_keywords(text: str, max_len: int = 10) -> list[str]:
    """从期望答案中提取关键短语"""
    import re
    # 移除标点后取较长的词或短语
    cleaned = re.sub(r"[，。！？、；：""''（）【】《》\s]", " ", text)
    words = [w for w in cleaned.split() if len(w) >= 4]
    # 取最长的几个关键词
    words.sort(key=len, reverse=True)
    return words[:5]


def evaluate_semantic_similarity(test_data: list[dict], model, tokenizer, threshold: float = 0.6) -> dict:
    """
    评估：用语义相似度判断（更准确但需要额外模型）
    需要: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("语义评估需要 sentence-transformers: pip install sentence-transformers")

    print("加载语义相似度模型...")
    st_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    correct = 0
    total = len(test_data)

    for i, item in enumerate(test_data):
        response = generate_response(model, tokenizer, item["instruction"], item.get("input", ""))
        expected = item["output"]

        emb1 = st_model.encode(response, normalize_embeddings=True)
        emb2 = st_model.encode(expected, normalize_embeddings=True)
        similarity = (emb1 @ emb2).item()

        is_correct = similarity >= threshold
        if is_correct:
            correct += 1

        if i < 3 or i % 20 == 0:
            print(f"[{i+1}/{total}] sim={similarity:.4f} {'✓' if is_correct else '✗'}")

    accuracy = correct / total if total > 0 else 0
    return {"total": total, "correct": correct, "accuracy": accuracy, "method": "semantic_similarity"}


def evaluate_interactive(test_data: list[dict], model, tokenizer) -> dict:
    """
    交互式评估：逐条显示给用户，人工打分
    """
    correct = 0
    total = len(test_data)

    print(f"共 {total} 条测试数据，逐条评估...")
    print("输入 y=正确, n=错误, s=跳过, q=退出\n")

    for i, item in enumerate(test_data):
        response = generate_response(model, tokenizer, item["instruction"], item.get("input", ""))
        expected = item["output"]

        print(f"--- [{i+1}/{total}] ---")
        print(f"问题: {item.get('input', item['instruction'])}")
        print(f"期望: {expected}")
        print(f"回复: {response}")
        choice = input("判定 (y/n/s/q): ").strip().lower()

        if choice == "q":
            total = i
            break
        elif choice == "y":
            correct += 1
        elif choice == "s":
            total -= 1

    accuracy = correct / total if total > 0 else 0
    return {"total": total, "correct": correct, "accuracy": accuracy, "method": "human"}


def main():
    parser = argparse.ArgumentParser(description="评估微调后模型的知识掌握程度")
    parser.add_argument("--base-model", required=True,
                        help="基础模型名称或路径 (如 Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--lora-path", default=None,
                        help="LoRA checkpoint 路径 (不指定则评估原始模型)")
    parser.add_argument("--test-data", required=True,
                        help="测试数据 JSON 文件路径")
    parser.add_argument("--method", choices=["precision", "semantic", "interactive"], default="precision",
                        help="评估方式: precision(关键词匹配), semantic(语义相似度), interactive(人工打分)")
    parser.add_argument("--output", default=None,
                        help="详细结果输出路径 (JSON)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="语义相似度阈值 (默认: 0.6)")
    parser.add_argument("--device", default="auto",
                        help="设备 (默认: auto)")
    args = parser.parse_args()

    # 加载测试数据
    test_data = load_json(args.test_data)
    print(f"加载测试数据: {len(test_data)} 条")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.lora_path)

    # 评估
    if args.method == "semantic":
        result = evaluate_semantic_similarity(test_data, model, tokenizer, args.threshold)
    elif args.method == "interactive":
        result = evaluate_interactive(test_data, model, tokenizer)
    else:
        result = evaluate_precision(test_data, model, tokenizer, args.output)

    # 输出结果
    print("\n" + "=" * 50)
    print("评估结果:")
    print(f"  方法: {result.get('method', 'precision')}")
    print(f"  总题数: {result['total']}")
    print(f"  正确: {result['correct']}")
    print(f"  准确率: {result['accuracy']:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
