# LLM Training Handbook

> 从监督学习到 LLM 全链路训练：一份面向 AI 算法工程师的实战教程

## 目标读者

你是一个有深度学习经验的 AI 算法工程师：
- 熟悉 CNN、RNN、Transformer 架构
- 有 supervised learning 训练模型的经验
- 想要上手 LLM 的 pretrain、SFT、RLHF、DPO 等新范式

本教程**不重复**你已经掌握的基础知识（backprop、optimizer 原理等），而是聚焦于 LLM 训练的**新概念和实战技巧**。

## 教程特点

- **实战为主**：每个章节 70%+ 是可执行代码
- **概念精简**：每个概念 1-2 段话 + 原论文链接，不啰嗦
- **Colab 友好**：所有实验在 Google Colab T4 上可运行
- **循序渐进**：10 章覆盖 pretrain → SFT → RLHF → DPO → 推理优化

## 环境搭建

### 方式一：Google Colab（推荐）

无需本地配置。每个 notebook 的第一个 cell 会自动检测 Colab 环境并安装依赖。

### 方式二：本地环境

```bash
# 1. 创建 conda 环境
conda create -n llm-handbook python=3.10 -y
conda activate llm-handbook

# 2. 安装 PyTorch（根据你的 CUDA 版本调整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python scripts/check_env.py
```

### 方式三：本地 CPU（仅限 Ch1-2）

Ch1 和 Ch2 不需要 GPU，纯 CPU 即可运行。Ch3 开始需要 GPU（本地或 Colab）。

## 硬件需求

| 章节 | 最低要求 | 推荐 |
|------|---------|------|
| Ch1-2 | CPU | 任意 GPU |
| Ch3 | RTX 2060S (8GB) / Colab T4 | Colab T4 |
| Ch4 | Colab T4（单卡模拟） | 多卡环境 |
| Ch5-8 | Colab T4（LoRA） | Colab T4 |
| Ch9 | Colab T4 | Colab T4 |
| Ch10 | Colab T4 | Colab T4 |

## 参考项目

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy 的极简 GPT 训练框架（Ch1-4 参考）
- [trl](https://github.com/huggingface/trl) — HuggingFace 的 Transformer Reinforcement Learning（Ch5-8 参考）
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF 格式推理引擎（Ch9 参考）
