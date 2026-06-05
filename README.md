# LoRA 微调 — 知识注入项目

使用 LoRA/QLoRA 对小型语言模型进行领域知识注入微调。

## 项目结构

```
d:\lora\
├── data/
│   ├── raw/                  # 原始文档（PDF/TXT/Markdown）
│   ├── train.json            # 训练数据
│   └── eval.json             # 测试数据
├── scripts/
│   ├── generate_qa.py        # 从文档自动生成 Q&A 训练数据
│   ├── eval_model.py         # 模型评估脚本
│   ├── utils.py              # 工具函数
│   ├── cloud_setup.sh        # 云端环境一键安装
│   ├── train_config.yaml     # LLaMA-Factory 训练配置
│   └── dataset_info.json     # 数据集注册文件
├── demo/
│   └── app.py                # Gradio 演示界面
├── output/                   # 微调产物下载到这里
└── README.md
```

## 快速开始（完整流程）

### 第一步：准备数据（本地）

**方式 A — 使用 HuggingFace 开源数据集：**
```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('shibing624/alpaca-zh', split='train')
ds.to_json('data/train.json', force_ascii=False)
"
```

**方式 B — 从文档自动生成：**
```bash
# 1. 把 PDF/文档放到 data/raw/ 目录下
# 2. 设置 API Key
set ANTHROPIC_API_KEY=sk-ant-xxx

# 3. 运行生成脚本
pip install anthropic PyMuPDF
python scripts/generate_qa.py --api claude --unknown-samples 20
```

**方式 C — 手工编写或从外部获取：**
直接把 JSON 文件放到 `data/` 目录。格式要求：
```json
[
  {"instruction": "请回答以下专业问题：", "input": "问题", "output": "答案"},
  ...
]
```

### 第二步：云端训练

1. 注册 [AutoDL](https://autodl.com) → 充值 50 元
2. 创建实例：**RTX 3090 / 4090** + PyTorch 镜像
3. 上传文件：
   - `data/train.json` → `/root/autodl-tmp/data/`
   - `data/eval.json` → `/root/autodl-tmp/data/`
   - `scripts/` 下所有文件 → `/root/autodl-tmp/scripts/`
4. SSH/JupyterLab 连接实例，运行：
```bash
bash /root/autodl-tmp/scripts/cloud_setup.sh

# 注册数据集
cp /root/autodl-tmp/scripts/dataset_info.json \
   /root/autodl-tmp/LLaMA-Factory/data/

# 开始训练
cd /root/autodl-tmp/LLaMA-Factory
cp /root/autodl-tmp/scripts/train_config.yaml examples/train_lora/
llamafactory-cli train examples/train_lora/train_config.yaml
```

5. 训练完成后下载 checkpoint 到本地 `output/`

### 第三步：评估模型

```bash
pip install torch transformers peft

python scripts/eval_model.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --lora-path ./output/qwen2.5_1.5b_lora/checkpoint-600 \
  --test-data ./data/eval.json
```

### 第四步：本地演示

```bash
pip install gradio

python demo/app.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --lora-path ./output/checkpoint-best \
  --share
```

## 云平台备选

| 平台 | 特点 |
|------|------|
| [AutoDL](https://autodl.com) | 国内首选，按小时计费 |
| [恒源云](https://gpushare.com) | 国内备选 |
| [Google Colab](https://colab.research.google.com) | 免费但限制多 |
| [Vast.ai](https://vast.ai) | 海外低价 |

## 常见问题

**Q: 3050 Ti 能跑吗？**  
A: 4GB 显存只能跑 Qwen2.5-0.5B + QLoRA，效果很差。推荐用云端。

**Q: 多少数据够用？**  
A: 最少 500 条，推荐 2000-5000 条。质量远比数量重要。

**Q: 训练要多久？**  
A: 1.5B 模型，3000 条数据，RTX 3090 约 1-2 小时。

**Q: 成本多少？**  
A: 云端约 20-50 元人民币（包含 3-5 轮实验）。

**Q: 训练完怎么在本地用？**  
A: 合并 LoRA 权重后用 Ollama 或 transformers 加载，4GB 推理 1.5B 模型足够。
