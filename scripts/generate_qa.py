"""
从文档自动生成 Q&A 训练数据（知识注入专用）

使用方法：
    1. 将文档放到 ../data/raw/ 目录下
    2. 设置 API_KEY（Claude 或 OpenAI）
    3. 运行: python generate_qa.py

支持的模型 API:
    - Anthropic Claude (推荐，质量最好)
    - OpenAI GPT (需要 openai 包)
    - 本地 Ollama 模型 (免费但质量一般)

输出: ../data/train.json + ../data/eval.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

# 添加脚本目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from utils import chunk_text, load_json, save_json, train_test_split, validate_alpaca_format


# ============================================================
# 配置区 — 根据你的情况修改
# ============================================================
QA_GENERATION_PROMPT = """请根据以下段落内容，生成 {num_questions} 个中文问答对。

要求：
1. 每个问答对应一个确定的知识点
2. 答案必须严格基于原文，不得虚构
3. 覆盖不同角度：定义、原理、应用、对比等
4. instruction 字段写 "请根据专业知识回答以下问题："
5. input 字段写问题，output 字段写答案

请严格按以下 JSON 格式输出，不要输出其他内容：
[
  {{"instruction": "请根据专业知识回答以下问题：", "input": "问题内容", "output": "答案内容"}},
  ...
]

段落内容：
{text}"""

ADD_UNKNOWN_PROMPT = """以下是一些模型不掌握的知识点问答，请生成 {num} 个这类样本。这些问题的答案为"我不确定"或"我没有这方面的信息"，用于教模型诚实面对不知道的问题。

格式：
[
  {{"instruction": "请根据专业知识回答以下问题：", "input": "一个可能被问到但模型不知道的问题", "output": "抱歉，我目前没有这方面的确切信息。建议您查阅相关资料或咨询专业人士。"}},
  ...
]

请生成 {num} 个不同类型的问题。"""


def generate_with_claude(
    text: str,
    api_key: str,
    num_questions: int = 5,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict] | None:
    """使用 Claude API 生成 Q&A"""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("请先安装 anthropic: pip install anthropic")
        return None

    client = Anthropic(api_key=api_key)
    prompt = QA_GENERATION_PROMPT.format(num_questions=num_questions, text=text)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text
        # 提取 JSON 部分
        return _parse_json_response(response_text)
    except Exception as e:
        print(f"Claude API 调用失败: {e}")
        return None


def generate_with_openai(
    text: str,
    api_key: str,
    num_questions: int = 5,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
) -> list[dict] | None:
    """使用 OpenAI 兼容 API 生成 Q&A（也支持国内中转）"""
    try:
        from openai import OpenAI
    except ImportError:
        print("请先安装 openai: pip install openai")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = QA_GENERATION_PROMPT.format(num_questions=num_questions, text=text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.3,
        )
        response_text = response.choices[0].message.content or ""
        return _parse_json_response(response_text)
    except Exception as e:
        print(f"OpenAI API 调用失败: {e}")
        return None


def generate_with_ollama(
    text: str,
    num_questions: int = 5,
    model: str = "qwen2.5:7b",
) -> list[dict] | None:
    """使用本地 Ollama 模型生成 Q&A（免费，但质量较低且速度慢）"""
    try:
        import requests
    except ImportError:
        print("请先安装 requests: pip install requests")
        return None

    prompt = QA_GENERATION_PROMPT.format(num_questions=num_questions, text=text)

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response_text = resp.json()["response"]
        return _parse_json_response(response_text)
    except Exception as e:
        print(f"Ollama 调用失败: {e}")
        return None


def _parse_json_response(text: str) -> list[dict] | None:
    """从 LLM 回复中提取 JSON 数组"""
    # 尝试直接解析
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 代码块
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # 尝试提取第一个 [ ... ] 数组
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    print(f"无法解析 API 返回的 JSON: {text[:200]}...")
    return None


def generate_from_file(
    filepath: str,
    generator_func,
    num_per_chunk: int = 5,
    chunk_size: int = 500,
) -> list[dict]:
    """从单个文件生成 Q&A"""
    print(f"正在处理: {filepath}")

    # 读取文件
    path = Path(filepath)
    if path.suffix.lower() == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            text = "\n\n".join(page.get_text() for page in doc)
        except ImportError:
            print("PDF 需要 PyMuPDF: pip install PyMuPDF")
            return []
    elif path.suffix.lower() in (".txt", ".md", ".markdown"):
        text = path.read_text(encoding="utf-8")
    else:
        print(f"不支持的文件格式: {path.suffix}")
        return []

    # 切分为段落
    chunks = chunk_text(text, chunk_size=chunk_size)
    print(f"  切分为 {len(chunks)} 个段落")

    # 逐段生成
    all_qa = []
    for i, chunk in enumerate(chunks):
        print(f"  处理段落 {i+1}/{len(chunks)} ({len(chunk)} 字)...")
        qa_pairs = generator_func(chunk, num_questions=num_per_chunk)
        if qa_pairs:
            # 补充默认 instruction（如果 API 没返回）
            for qa in qa_pairs:
                if "instruction" not in qa or not qa["instruction"]:
                    qa["instruction"] = "请根据专业知识回答以下问题："
                if "input" not in qa:
                    qa["input"] = ""
                if "output" not in qa:
                    qa["output"] = ""
            all_qa.extend(qa_pairs)

    return all_qa


def generate_unknown_samples(
    generator_func,
    num_samples: int = 20,
) -> list[dict]:
    """生成'不知道'类型的样本"""
    print(f"正在生成 {num_samples} 条'不知道'样本...")
    prompt = ADD_UNKNOWN_PROMPT.format(num=num_samples)

    # 用一个特殊的调用方式 — 直接用 API 的内核
    result = generator_func(prompt, num_questions=num_samples)
    if result:
        for item in result:
            if "instruction" not in item or not item["instruction"]:
                item["instruction"] = "请根据专业知识回答以下问题："
        return result
    return []


def main():
    parser = argparse.ArgumentParser(description="从文档生成 Q&A 训练数据")
    parser.add_argument("--api", choices=["claude", "openai", "ollama"], default="claude",
                        help="使用的 API (默认: claude)")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                        help="API Key (或设置环境变量 ANTHROPIC_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--model", default=None,
                        help="模型名称 (默认: claude-haiku, openai=gpt-4o-mini, ollama=qwen2.5:7b)")
    parser.add_argument("--input-dir", default="../data/raw",
                        help="原始文档目录 (默认: ../data/raw)")
    parser.add_argument("--output-dir", default="../data",
                        help="输出目录 (默认: ../data)")
    parser.add_argument("--num-per-chunk", type=int, default=5,
                        help="每个段落生成的 Q&A 数量 (默认: 5)")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="段落最大字数 (默认: 500)")
    parser.add_argument("--unknown-samples", type=int, default=0,
                        help="额外生成的'不知道'样本数量 (默认: 0)")
    args = parser.parse_args()

    # 设置默认模型
    if args.model is None:
        if args.api == "claude":
            args.model = "claude-haiku-4-5-20251001"
        elif args.api == "openai":
            args.model = "gpt-4o-mini"
        else:
            args.model = "qwen2.5:7b"

    # 检查 API Key
    if args.api != "ollama" and not args.api_key:
        print(f"错误: 使用 {args.api} API 需要提供 --api-key 或设置环境变量")
        sys.exit(1)

    # 选择生成函数
    if args.api == "claude":
        generator = lambda text, num_q: generate_with_claude(
            text, args.api_key, num_q, args.model
        )
    elif args.api == "openai":
        generator = lambda text, num_q: generate_with_openai(
            text, args.api_key, num_q, args.model
        )
    else:
        generator = lambda text, num_q: generate_with_ollama(
            text, num_q, args.model
        )

    # 查找所有文档
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    files = list(input_dir.glob("*"))
    files = [f for f in files if f.suffix.lower() in (".pdf", ".txt", ".md", ".markdown")]
    if not files:
        print(f"在 {input_dir} 下未找到任何文档文件 (.pdf/.txt/.md)")
        sys.exit(1)

    print(f"找到 {len(files)} 个文档文件")
    print(f"使用 API: {args.api}, 模型: {args.model}")

    # 逐文件生成
    all_data = []
    for filepath in files:
        qa = generate_from_file(
            str(filepath),
            generator,
            num_per_chunk=args.num_per_chunk,
            chunk_size=args.chunk_size,
        )
        all_data.extend(qa)
        print(f"  生成了 {len(qa)} 个 Q&A")

    # 添加"不知道"样本
    if args.unknown_samples > 0:
        unknown = generate_unknown_samples(generator, args.unknown_samples)
        all_data.extend(unknown)
        print(f"  生成了 {len(unknown)} 条'不知道'样本")

    print(f"\n总共生成 {len(all_data)} 条数据")

    # 验证格式
    errors = validate_alpaca_format(all_data)
    if errors:
        print(f"\n⚠ 发现 {len(errors)} 个格式问题:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个问题")
        print("建议修正后重新运行")
        return

    # 切分训练/测试集
    train, eval_set = train_test_split(all_data, test_ratio=0.2)

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(train, output_dir / "train.json")
    save_json(eval_set, output_dir / "eval.json")

    print(f"\n✅ 完成！")
    print(f"  训练集: {len(train)} 条 -> {output_dir / 'train.json'}")
    print(f"  测试集: {len(eval_set)} 条 -> {output_dir / 'eval.json'}")
    print(f"\n下一步: 将 train.json 和 eval.json 上传到云端 AutoDL 实例进行训练")


if __name__ == "__main__":
    main()
