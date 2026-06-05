"""
Gradio 演示界面 — 测试微调后的知识注入模型

使用方法:
    python demo/app.py --base-model Qwen/Qwen2.5-1.5B-Instruct \
                       --lora-path ./output/lora_checkpoint \
                       --share

参数:
    --share  创建公网链接 (通过 Gradio 中转)
"""
import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# 全局模型和 tokenizer
_model = None
_tokenizer = None


def load_model(base_model: str, lora_path: str | None = None):
    """加载模型（只执行一次）"""
    global _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"加载模型: {base_model}")
    _tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    _model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path:
        print(f"加载 LoRA: {lora_path}")
        _model = PeftModel.from_pretrained(_model, lora_path)
        _model = _model.merge_and_unload()

    _model.eval()
    print("模型加载完成！")


def chat(message: str, history: list, temperature: float, max_tokens: int):
    """处理对话"""
    global _model, _tokenizer

    if _model is None:
        yield "模型尚未加载，请先启动。"
        return

    # 构建 prompt
    messages = [
        {
            "role": "system",
            "content": "你是一个专业知识助手，经过特定领域知识的微调训练。请准确、简洁地回答用户问题。如果你不确定答案，请诚实说明。",
        },
    ]

    # 添加历史对话
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})

    messages.append({"role": "user", "content": message})

    try:
        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # fallback
        parts = []
        for m in messages:
            role = m["role"]
            parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        formatted = "\n".join(parts)

    inputs = _tokenizer(formatted, return_tensors="pt")
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.1,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=_tokenizer.eos_token_id,
        )

    response = _tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    yield response.strip()


def create_demo():
    """创建 Gradio 界面"""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    )

    with gr.Blocks(theme=theme, title="知识注入模型演示") as demo:
        gr.Markdown(
            """
            # 📚 领域知识问答助手
            这是一个经过 **LoRA 微调** 的专业知识模型。
            请提问目标领域的相关问题来测试模型的知识掌握程度。
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=500,
                    bubble_full_width=False,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="输入问题",
                        placeholder="输入你的问题...",
                        scale=4,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("清空对话", size="sm")

            with gr.Column(scale=1):
                gr.Markdown("### 参数设置")
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.1,
                    label="Temperature",
                    info="越低越准确，越高越有创意",
                )
                max_tokens = gr.Slider(
                    minimum=64, maximum=1024, value=256, step=64,
                    label="最大回复长度",
                )

        gr.Markdown(
            """
            ---
            ### 💡 使用提示
            - 提问**目标领域**内的专业知识问题，测试微调效果
            - 也可以问**领域外**的问题，观察模型是否诚实承认不知道
            - 对比**微调前后**的回答差异，直观感受知识注入效果
            """
        )

        # 事件绑定
        msg.submit(chat, [msg, chatbot, temperature, max_tokens], [chatbot]).then(
            lambda: "", None, [msg]
        )
        submit_btn.click(chat, [msg, chatbot, temperature, max_tokens], [chatbot]).then(
            lambda: "", None, [msg]
        )
        clear_btn.click(lambda: [], None, [chatbot])

    return demo


def main():
    parser = argparse.ArgumentParser(description="知识注入模型 Gradio 演示")
    parser.add_argument("--base-model", required=True,
                        help="基础模型名称或路径")
    parser.add_argument("--lora-path", default=None,
                        help="LoRA checkpoint 路径")
    parser.add_argument("--share", action="store_true",
                        help="创建 Gradio 公网分享链接")
    parser.add_argument("--port", type=int, default=7860,
                        help="服务端口 (默认: 7860)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="服务地址 (默认: 127.0.0.1)")
    args = parser.parse_args()

    # 加载模型
    load_model(args.base_model, args.lora_path)

    # 启动
    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
