# LLM Training Handbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a JupyterBook-based learning website with 10 hands-on chapters covering the full LLM training pipeline (pretrain → SFT → RLHF → DPO → inference), targeting AI engineers transitioning from supervised learning to LLM paradigms.

**Architecture:** Independent repository with JupyterBook scaffolding. Ch1-4 reference nanoGPT source code directly. Ch5-10 use HuggingFace ecosystem (trl, transformers, peft). Every notebook targets Colab T4 (16GB VRAM) as the primary runtime.

**Tech Stack:** JupyterBook, PyTorch 2.x, nanoGPT, tiktoken, trl, transformers, peft, bitsandbytes, llama.cpp

---

## File Map

| File | Responsibility |
|------|---------------|
| `_config.yml` | JupyterBook site configuration (title, author, theme) |
| `_toc.yml` | Chapter navigation and numbering |
| `intro.md` | Preface, target audience, environment setup (local + Colab) |
| `requirements.txt` | All Python dependencies with pinned versions |
| `notebooks/ch01_gpt_architecture.ipynb` | GPT architecture from scratch |
| `notebooks/ch02_tokenization.ipynb` | BPE tokenization and data engineering |
| `notebooks/ch03_pretraining.ipynb` | Pretraining on Shakespeare/OpenWebText |
| `notebooks/ch04_distributed_training.ipynb` | DDP, mixed precision, Flash Attention |
| `notebooks/ch05_sft.ipynb` | Instruction tuning with SFTTrainer |
| `notebooks/ch06_reward_modeling.ipynb` | Reward model training |
| `notebooks/ch07_rlhf_ppo.ipynb` | RLHF with PPO |
| `notebooks/ch08_dpo.ipynb` | Direct preference optimization |
| `notebooks/ch09_inference_optimization.ipynb` | Quantization and inference speed |
| `notebooks/ch10_full_pipeline.ipynb` | End-to-end training recipe |
| `scripts/check_env.py` | Runtime environment detection (Colab vs local) |
| `scripts/download_data.sh` | Download all datasets |
| `.github/workflows/build.yml` | JupyterBook auto-build on push |

---

## Phase 1: Project Infrastructure

### Task 1: Initialize Repository and JupyterBook Scaffolding

**Files:**
- Create: `_config.yml`
- Create: `_toc.yml`
- Create: `requirements.txt`

- [ ] **Step 1: Create `_config.yml`**

```yaml
title: LLM Training Handbook
author: LLM Training Handbook Contributors
copyright: "2026"
logo: assets/logo.png
execute:
  execute_notebooks: "off"
  timeout: 600
repository:
  url: https://github.com/YOUR_USERNAME/llm-training-handbook
  path_to_book: ""
  branch: main
html:
  use_issues_button: true
  use_repository_button: true
  announcement: "本教程以实战为主，概念为辅。目标读者：有深度学习经验的 AI 算法工程师。"
parse:
  myst_enable_extensions:
    - dollarmath
    - linkify
    - substitution
```

- [ ] **Step 2: Create `_toc.yml`**

```yaml
format: jb-book
root: intro
parts:
  - caption: Part I - 基础架构与预训练
    chapters:
      - file: notebooks/ch01_gpt_architecture
        title: "第1章：GPT 架构拆解"
      - file: notebooks/ch02_tokenization
        title: "第2章：Tokenizer 与数据工程"
      - file: notebooks/ch03_pretraining
        title: "第3章：预训练实战"
      - file: notebooks/ch04_distributed_training
        title: "第4章：分布式训练与效率优化"
  - caption: Part II - 对齐训练
    chapters:
      - file: notebooks/ch05_sft
        title: "第5章：SFT 指令微调"
      - file: notebooks/ch06_reward_modeling
        title: "第6章：Reward Modeling"
      - file: notebooks/ch07_rlhf_ppo
        title: "第7章：RLHF (PPO)"
      - file: notebooks/ch08_dpo
        title: "第8章：DPO 直接偏好优化"
  - caption: Part III - 生产部署
    chapters:
      - file: notebooks/ch09_inference_optimization
        title: "第9章：推理优化"
      - file: notebooks/ch10_full_pipeline
        title: "第10章：全流程实战"
```

- [ ] **Step 3: Create `requirements.txt`**

```
torch>=2.1.0
tiktoken>=0.6.0
transformers>=4.38.0
trl>=0.8.0
peft>=0.9.0
datasets>=2.18.0
accelerate>=0.27.0
bitsandbytes>=0.42.0
wandb>=0.16.0
matplotlib>=3.8.0
pandas>=2.2.0
jupyter-book>=1.0.0
numpy>=1.26.0
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p notebooks data scripts assets .github/workflows
```

- [ ] **Step 5: Commit scaffolding**

```bash
git init
git add _config.yml _toc.yml requirements.txt
git commit -m "feat: initialize JupyterBook project scaffolding"
```

---

### Task 2: Write `intro.md` — Preface and Environment Setup

**Files:**
- Create: `intro.md`

- [ ] **Step 1: Write `intro.md`**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add intro.md
git commit -m "docs: add preface and environment setup guide"
```

---

### Task 3: Write Helper Scripts

**Files:**
- Create: `scripts/check_env.py`
- Create: `scripts/download_data.sh`

- [ ] **Step 1: Write `scripts/check_env.py`**

```python
"""
环境检查脚本 - 验证所有依赖是否正确安装
"""
import sys

def check_python():
    print(f"Python: {sys.version}")
    assert sys.version_info >= (3, 10), "需要 Python >= 3.10"

def check_torch():
    import torch
    print(f"PyTorch: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

def check_packages():
    packages = {
        "tiktoken": "tiktoken",
        "transformers": "transformers",
        "trl": "trl",
        "peft": "peft",
        "datasets": "datasets",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
    }
    for name, module in packages.items():
        try:
            mod = __import__(module)
            print(f"  {name}: {mod.__version__}")
        except ImportError:
            print(f"  {name}: NOT INSTALLED")

def check_colab():
    return "google.colab" in sys.modules

if __name__ == "__main__":
    is_colab = check_colab()
    print(f"Running on: {'Colab' if is_colab else 'Local'}")
    print()
    check_python()
    print()
    check_torch()
    print()
    print("Dependencies:")
    check_packages()
    print()
    print("Environment check complete.")
```

- [ ] **Step 2: Write `scripts/download_data.sh`**

```bash
#!/bin/bash
# 下载本教程使用的所有数据集
set -e

echo "Downloading datasets..."

# tiny-shakespeare
mkdir -p data/shakespeare_char
wget -q -O data/shakespeare_char/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
echo "  tiny-shakespeare: done"

# Alpaca dataset (SFT)
wget -q -O data/alpaca_data.json https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
echo "  alpaca: done"

echo "All datasets downloaded to data/"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/check_env.py scripts/download_data.sh
git commit -m "feat: add environment check and data download scripts"
```

---

### Task 4: Verify JupyterBook Build

**Files:**
- None (verification only)

- [ ] **Step 1: Install JupyterBook and build**

```bash
pip install jupyter-book
jupyter-book create --help  # verify installation
```

- [ ] **Step 2: Create placeholder notebooks so _toc.yml resolves**

Create minimal notebooks (just a title cell) for all 10 chapters so the TOC doesn't break during build verification.

- [ ] **Step 3: Build and verify**

```bash
jupyter-book build .
# Expected: build succeeds with no errors
# HTML output in _build/html/
```

- [ ] **Step 4: Commit placeholder notebooks**

```bash
git add notebooks/
git commit -m "chore: add placeholder notebooks for build verification"
```

---

## Phase 2: Foundation Chapters (Ch1-4) — Based on nanoGPT

### Task 5: Ch1 — GPT 架构拆解

**Files:**
- Modify: `notebooks/ch01_gpt_architecture.ipynb`

**Notebook cell outline:**

Cell 1 [Markdown]: 标题 + 本章目标 + 前置知识清单
```
# 第1章：GPT 架构拆解

## 本章目标
- 理解 GPT (Generative Pre-trained Transformer) 的完整架构
- 从零手写每个组件：CausalSelfAttention、MLP、Block、GPT
- 理解 GPT-2 的参数配置和计算量分析

## 前置知识
- Transformer 的 Self-Attention 机制（你知道 Q/K/V 是什么）
- PyTorch nn.Module 的基本用法
- 矩阵乘法和 softmax 运算
```

Cell 2 [Code]: 环境检测 + 安装
```python
import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !pip install torch tiktoken matplotlib
    !wget -q -O input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
else:
    print("本地环境运行，请确保已按 intro.md 配置好环境")
```

Cell 3 [Markdown]: 核心概念速览
```
## GPT 架构速览

GPT = Token Embedding + Positional Embedding + N × Transformer Block + LayerNorm + Linear Head

与 BERT 的区别：GPT 是 **decoder-only**，使用 causal mask（下三角矩阵）确保只能看到当前和之前的 token。

参考论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762), [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
```

Cell 4 [Code]: 手写 CausalSelfAttention
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    与标准 Self-Attention 的区别：
    1. 多头并行计算 (n_head 个独立的 attention head)
    2. Causal mask：只关注当前位置及之前的 token
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # Q, K, V 合并计算
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # causal mask: 下三角矩阵
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```

Cell 5 [Markdown]: 分析 attention 的计算过程 — 打印中间变量 shape

Cell 6 [Code]: 手写 MLP
```python
class MLP(nn.Module):
    """Feed-forward network with GELU activation.

    标准做法：d_model → 4 × d_model → d_model
    GPT-2 使用 GELU 而不是 ReLU。
    """
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

Cell 7 [Code]: 手写 Block（Transformer Block）
```python
class Block(nn.Module):
    """Pre-norm Transformer Block.

    GPT 使用 Pre-LN（LayerNorm 在 attention/MLP 之前），
    而原始 Transformer 使用 Post-LN。
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # residual connection
        x = x + self.mlp(self.ln_2(x))     # residual connection
        return x
```

Cell 8 [Markdown]: 解释 Pre-LN vs Post-LN 的区别，为什么 GPT 选择 Pre-LN

Cell 9 [Code]: 手写完整 GPT 模型
```python
class GPTConfig:
    """GPT-2 的配置参数"""
    def __init__(self, vocab_size=50304, block_size=1024,
                 n_layer=12, n_head=12, n_embd=768, dropout=0.1):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)    # token embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)    # position embedding
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config.n_embd, config.n_head,
                                       config.block_size, config.dropout)
                                 for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: embedding 和 output head 共享权重
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

Cell 10 [Code]: 实例化模型，验证 forward pass
```python
# 用 baby-GPT 配置测试（小到可以 CPU 跑）
config = GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128)
model = GPT(config)
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# forward pass 测试
x = torch.randint(0, config.vocab_size, (2, 64))  # batch=2, seq_len=64
logits, loss = model(x, targets=x)
print(f"logits shape: {logits.shape}")
print(f"loss: {loss.item():.4f}")
```

Cell 11 [Markdown]: GPT-2 参数量分析表

Cell 12 [Code]: 各模型配置对比
```python
configs = {
    "GPT-2 Small":  GPTConfig(n_layer=12, n_head=12, n_embd=768),
    "GPT-2 Medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024),
    "GPT-2 Large":  GPTConfig(n_layer=36, n_head=20, n_embd=1280),
    "GPT-2 XL":     GPTConfig(n_layer=48, n_head=25, n_embd=1600),
}
for name, cfg in configs.items():
    model = GPT(cfg)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{name:15s}: {params:.0f}M params")
```

Cell 13 [Markdown]: 练习 — 修改模型配置、尝试 Flash Attention

Cell 14 [Markdown]: 延伸阅读

- [ ] **Step 1: Create the complete notebook** — implement all cells above as a `.ipynb` file

- [ ] **Step 2: Run `Run All` locally** — verify every cell executes on CPU without GPU

- [ ] **Step 3: Commit**

```bash
git add notebooks/ch01_gpt_architecture.ipynb
git commit -m "feat: ch01 GPT architecture deep dive notebook"
```

---

### Task 6: Ch2 — Tokenization 与数据工程

**Files:**
- Modify: `notebooks/ch02_tokenization.ipynb`

**Notebook cell outline:**

Cell 1 [Markdown]: 标题 + 本章目标
```
# 第2章：Tokenizer 与数据工程

## 本章目标
- 理解 BPE (Byte Pair Encoding) 的原理和实现
- 使用 tiktoken 进行 GPT-2 BPE tokenization
- 从零手写一个 mini BPE tokenizer
- 构建训练数据的完整 pipeline
```

Cell 2 [Code]: 环境检测 + 安装（同 Ch1 模式）

Cell 3 [Markdown]: 核心概念速览
```
## Tokenization 速览

BPE 的核心思想：从字符级别开始，反复合并最高频的 token pair，直到达到目标词表大小。

GPT-2 的 BPE：基于 byte-level，词表大小 50,257。
tiktoken 是 OpenAI 开源的高性能 BPE 实现（Rust 后端）。
```

Cell 4 [Code]: tiktoken 基本使用
```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
text = "Hello, LLM Training Handbook!"
tokens = enc.encode(text)
print(f"原文: {text}")
print(f"Tokens: {tokens}")
print(f"解码: {enc.decode(tokens)}")
print(f"词表大小: {enc.n_vocab}")
```

Cell 5 [Code]: 从零手写 mini BPE
```python
def get_pair_counts(ids, counts=None):
    """统计相邻 token pair 的频率"""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """将 ids 中所有 pair 替换为 idx"""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class MiniTokenizer:
    """Minimal BPE tokenizer for educational purposes."""
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.merges = {}
        self.vocab = {}

    def train(self, text):
        tokens = list(text.encode("utf-8"))
        vocab_size = 256 + self.num_merges
        for i in range(self.num_merges):
            counts = get_pair_counts(tokens)
            if not counts:
                break
            top_pair = max(counts, key=counts.get)
            idx = 256 + i
            tokens = merge(tokens, top_pair, idx)
            self.merges[top_pair] = idx
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            counts = get_pair_counts(tokens)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = merge(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

# 训练并测试
with open("input.txt", "r") as f:
    text = f.read()
tokenizer = MiniTokenizer(num_merges=200)
tokenizer.train(text[:10000])
test = "Hello world"
encoded = tokenizer.encode(test)
print(f"Encoded: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")
```

Cell 6 [Markdown]: 分析 BPE 合并过程 — 打印前 20 次合并的 pair

Cell 7 [Code]: 字符级 tokenizer（nanoGPT shakespeare_char 方式）
```python
# Character-level tokenization (最简单的方式)
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print(f"词表大小: {vocab_size} (character-level)")
print(f"示例编码: {encode('hello')}")
```

Cell 8 [Code]: 构建 train/val 数据集
```python
import torch

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"总 token 数: {len(data):,}")
print(f"训练集: {len(train_data):,} tokens")
print(f"验证集: {len(val_data):,} tokens")

def get_batch(split, block_size=128, batch_size=4):
    data_source = train_data if split == "train" else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print(f"batch x shape: {xb.shape}, batch y shape: {yb.shape}")
```

Cell 9 [Markdown]: 练习 — 对比不同 tokenizer 的压缩率

Cell 10 [Markdown]: 延伸阅读

- [ ] **Step 1: Create the complete notebook**

- [ ] **Step 2: Run `Run All` locally on CPU**

- [ ] **Step 3: Commit**

```bash
git add notebooks/ch02_tokenization.ipynb
git commit -m "feat: ch02 tokenization and data engineering notebook"
```

---

### Task 7: Ch3 — 预训练实战

**Files:**
- Modify: `notebooks/ch03_pretraining.ipynb`

**Notebook cell outline:**

Cell 1 [Markdown]: 标题 + 本章目标
```
# 第3章：预训练实战

## 本章目标
- 在 Shakespeare 数据集上完整训练一个 GPT
- 分析 loss 曲线，理解训练 dynamics
- 实验 learning rate、batch size 对训练的影响
- 理解 Scaling Laws 的基本规律
```

Cell 2 [Code]: 环境检测

Cell 3 [Markdown]: 核心概念 — pretraining 就是 next-token prediction

Cell 4 [Code]: 复用 Ch1 的 GPT 模型代码（内联 copy，确保 notebook 自包含）

Cell 5 [Code]: 数据加载
```python
# Shakespeare character-level data
with open("input.txt", "r") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
```

Cell 6 [Code]: 训练循环
```python
@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=100):
    model.eval()
    losses = {}
    for split, data_source in [("train", train_data), ("val", val_data)]:
        losses_split = torch.zeros(eval_iters)
        for i in range(eval_iters):
            ix = torch.randint(len(data_source) - block_size, (batch_size,))
            x = torch.stack([data_source[j:j+block_size] for j in ix])
            y = torch.stack([data_source[j+1:j+block_size+1] for j in ix])
            _, loss = model(x, y)
            losses_split[i] = loss.item()
        losses[split] = losses_split.mean()
    model.train()
    return losses

def train_gpt(config, train_data, val_data, max_iters=2000, eval_interval=200,
              learning_rate=3e-4, batch_size=64, block_size=256):
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_losses, val_losses, steps = [], [], []

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, block_size, batch_size)
            train_losses.append(losses["train"].item())
            val_losses.append(losses["val"].item())
            steps.append(iter)
            print(f"Step {iter:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        xb = torch.stack([train_data[i:i+block_size] for i in ix])
        yb = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model, train_losses, val_losses, steps
```

Cell 7 [Code]: 运行训练 + 绘制 loss 曲线
```python
config = GPTConfig(vocab_size=vocab_size, block_size=256,
                   n_layer=6, n_head=6, n_embd=384, dropout=0.2)
model, train_losses, val_losses, steps = train_gpt(
    config, train_data, val_data,
    max_iters=3000, eval_interval=300,
    learning_rate=1e-3, batch_size=64
)
import matplotlib.pyplot as plt
plt.plot(steps, train_losses, label="train")
plt.plot(steps, val_losses, label="val")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss Curve")
plt.show()
```

Cell 8 [Code]: 文本生成测试
```python
@torch.no_grad()
def generate(model, idx, max_new_tokens=500, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

context = torch.zeros((1, 1), dtype=torch.long)
output = generate(model, context, max_new_tokens=500, temperature=0.8, top_k=40)
print("".join([itos[i.item()] for i in output[0]]))
```

Cell 9 [Markdown]: 分析生成结果 — 观察模型学到了什么模式

Cell 10 [Code]: Learning Rate 实验 — 对比不同 LR
```python
# 实验：不同 learning rate 的训练曲线
results = {}
for lr in [1e-2, 1e-3, 1e-4]:
    config = GPTConfig(vocab_size=vocab_size, block_size=128,
                       n_layer=4, n_head=4, n_embd=128, dropout=0.1)
    _, tl, vl, s = train_gpt(config, train_data, val_data,
                              max_iters=1000, eval_interval=200,
                              learning_rate=lr, batch_size=32, block_size=128)
    results[lr] = (s, vl)
# 绘制对比图
```

Cell 11 [Markdown]: Scaling Laws 简述 — Chinchilla 的核心结论

Cell 12 [Markdown]: 练习 + 延伸阅读

- [ ] **Step 1: Create the complete notebook**

- [ ] **Step 2: Run on GPU (local 2060S or Colab T4)**

- [ ] **Step 3: Commit**

```bash
git add notebooks/ch03_pretraining.ipynb
git commit -m "feat: ch03 pretraining notebook with loss analysis and scaling laws"
```

---

### Task 8: Ch4 — 分布式训练与效率优化

**Files:**
- Modify: `notebooks/ch04_distributed_training.ipynb`

**Notebook cell outline:**

Cell 1 [Markdown]: 标题 + 本章目标
```
# 第4章：分布式训练与效率优化

## 本章目标
- 理解 DDP (Distributed Data Parallel) 的原理
- 掌握混合精度训练 (FP16/BF16) 的实现
- 理解 Flash Attention 的加速原理
- 学会使用 gradient accumulation 模拟大 batch
- 计算 MFU (Model FLOPs Utilization)
```

Cell 2 [Code]: 环境检测

Cell 3 [Markdown]: 核心概念速览

Cell 4 [Code]: Mixed Precision Training
```python
# 对比 FP32 vs FP16/BF16 的显存占用和训练速度
model_fp32 = GPT(GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128))
model_fp16 = GPT(GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128)).half()

def measure_memory(model, label):
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 65, (4, 128))
    if next(model.parameters()).dtype != torch.float32:
        x = x  # inputs stay as long
    _, loss = model(x, x)
    loss.backward()
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    print(f"{label}: peak memory = {peak_mem:.1f} MB")

if torch.cuda.is_available():
    model_fp32 = model_fp32.cuda()
    model_fp16 = model_fp16.cuda()
    measure_memory(model_fp32, "FP32")
    measure_memory(model_fp16, "FP16")
```

Cell 5 [Code]: 使用 torch.cuda.amp 的自动混合精度
```python
from torch.cuda.amp import autocast, GradScaler

model = GPT(GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128)).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler()

x = torch.randint(0, 65, (4, 128)).cuda()
with autocast():
    logits, loss = model(x, x)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
print("Mixed precision training step complete")
```

Cell 6 [Code]: Gradient Accumulation
```python
# 模拟大 batch：accumulation_steps 次前向后向再一步 optimizer
accumulation_steps = 4
micro_batch_size = 8
effective_batch_size = micro_batch_size * accumulation_steps
print(f"micro batch: {micro_batch_size}, accumulation: {accumulation_steps}")
print(f"effective batch size: {effective_batch_size}")

optimizer.zero_grad()
for micro_step in range(accumulation_steps):
    x = torch.randint(0, 65, (micro_batch_size, 128)).cuda()
    _, loss = model(x, x)
    loss = loss / accumulation_steps  # scale loss
    loss.backward()
optimizer.step()
```

Cell 7 [Markdown]: Flash Attention 原理 — IO-aware attention，减少 HBM 访问

Cell 8 [Code]: Flash Attention 使用（PyTorch 2.0+）
```python
# PyTorch 2.0+ 的 scaled_dot_product_attention
from torch.nn.functional import scaled_dot_product_attention

q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)

# Flash Attention (is_causal=True)
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    out_flash = scaled_dot_product_attention(q, k, v, is_causal=True)

# 标准 attention（对比）
out_standard = scaled_dot_product_attention(q, k, v, is_causal=True, attn_mask=torch.tril(torch.ones(128, 128, device="cuda")))

print(f"Flash output shape: {out_flash.shape}")
print(f"Max diff: {(out_flash - out_standard).abs().max().item():.6f}")
```

Cell 9 [Code]: DDP 概念演示（单卡模拟）
```python
# DDP 原理：每个 GPU 持有模型副本，数据分片，梯度 all-reduce
# 单卡模拟 DDP 的 gradient sync 概念
print("DDP 核心流程：")
print("1. 每个 GPU 独立前向 + 反向，得到梯度")
print("2. All-Reduce 同步所有 GPU 的梯度（取平均）")
print("3. 每个 GPU 独立更新参数（结果一致）")
print()
print("实际使用：")
print("  torchrun --nproc_per_node=4 train.py  # 4卡 DDP")
```

Cell 10 [Code]: MFU 计算
```python
def estimate_mfu(model, config, batch_size, seq_len, dt):
    """估算 Model FLOPs Utilization"""
    # 每个 token 的前向 FLOPs ≈ 6 * n_params
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_token = 6 * n_params
    tokens_per_sec = batch_size * seq_len / dt
    achieved_tflops = flops_per_token * tokens_per_sec / 1e12
    # A100 理论算力 ~312 TFLOPS (FP16)
    # T4 理论算力 ~65 TFLOPS (FP16)
    hardware_tflops = 65.0  # T4
    mfu = achieved_tflops / hardware_tflops * 100
    return mfu

# 实测 MFU
import time
model = GPT(config).cuda()
x = torch.randint(0, 65, (16, 256), device="cuda")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    _, loss = model(x, x)
    loss.backward()
torch.cuda.synchronize()
t1 = time.time()
mfu = estimate_mfu(model, config, 16, 256, (t1-t0)/10)
print(f"MFU: {mfu:.1f}%")
```

Cell 11 [Markdown]: 练习 + 延伸阅读

- [ ] **Step 1: Create the complete notebook**

- [ ] **Step 2: Run on Colab T4 (Flash Attention 需要GPU)**

- [ ] **Step 3: Commit**

```bash
git add notebooks/ch04_distributed_training.ipynb
git commit -m "feat: ch04 distributed training and efficiency optimization"
```

---

## Phase 3: Alignment Chapters (Ch5-8) — Based on trl

### Task 9: Ch5 — SFT 指令微调

**Files:**
- Modify: `notebooks/ch05_sft.ipynb`

**Notebook cell outline:**

Cell 1 [Markdown]: 标题 + 本章目标
```
# 第5章：SFT 指令微调 (Supervised Fine-Tuning)

## 本章目标
- 理解 instruction tuning 的数据格式（Alpaca format）
- 理解 loss masking：只在 response 部分计算 loss
- 使用 LoRA 高效微调大模型
- 使用 trl 的 SFTTrainer 完成完整的 SFT 流程
```

Cell 2 [Code]: 环境检测 + 安装
```python
import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !pip install torch transformers trl peft datasets accelerate bitsandbytes
```

Cell 3 [Markdown]: SFT 概念速览

Cell 4 [Code]: 构建 instruction dataset
```python
from datasets import Dataset

# Alpaca format: instruction + input + output
alpaca_data = [
    {"instruction": "翻译以下句子为英文", "input": "今天天气很好",
     "output": "The weather is nice today."},
    {"instruction": "计算", "input": "25 * 37", "output": "925"},
    # ... 更多样本
]

# 下载 Alpaca 数据集
from datasets import load_dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

# 格式化为对话模板
def format_alpaca(example):
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    return {"text": prompt + example["output"]}

dataset = dataset.map(format_alpaca)
print(f"数据集大小: {len(dataset)}")
print(f"样本示例:\n{dataset[0]['text'][:200]}")
```

Cell 5 [Code]: 加载 base model + LoRA 配置
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_name = "Qwen/Qwen2.5-0.5B"  # 小到 Colab T4 可跑
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Cell 6 [Code]: 使用 trl SFTTrainer 训练
```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    max_seq_length=512,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
print("SFT 训练完成")
```

Cell 7 [Code]: 测试 SFT 后的模型
```python
def test_sft_model(model, tokenizer, instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

test_cases = [
    "Explain what a transformer is in simple terms.",
    "Write a haiku about programming.",
    "What is the capital of France?",
]
for tc in test_cases:
    print(f"Q: {tc}")
    print(f"A: {test_sft_model(model, tokenizer, tc)}")
    print()
```

Cell 8 [Markdown]: Loss masking 原理详解 — 为什么只在 response 部分计算 loss

Cell 9 [Code]: 手动实现 loss masking（教育目的）
```python
# 展示如何在 instruction + response 中只对 response 计算 loss
def compute_loss_with_mask(logits, labels, response_start_positions):
    """只在 response 部分计算 loss"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # 创建 mask：只有 response 部分为 True
    mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    for i, start in enumerate(response_start_positions):
        mask[i, start:] = True
    masked_loss = loss.view(shift_labels.shape) * mask
    return masked_loss.sum() / mask.sum()
```

Cell 10 [Markdown]: 练习 + 延伸阅读

- [ ] **Step 1: Create the complete notebook**

- [ ] **Step 2: Run on Colab T4**

- [ ] **Step 3: Commit**

```bash
git add notebooks/ch05_sft.ipynb
git commit -m "feat: ch05 SFT instruction tuning with trl"
```

---

### Task 10: Ch6 — Reward Modeling

**Files:**
- Modify: `notebooks/ch06_reward_modeling.ipynb`

**Notebook cell outline:**

Cell 1 [Markdown]: 标题 + 本章目标
```
# 第6章：Reward Modeling

## 本章目标
- 理解偏好数据 (preference data) 的格式和收集方式
- 理解 Bradley-Terry 模型（Reward Model 的理论基础）
- 使用 trl 训练一个 Reward Model
```

Cell 2 [Code]: 环境检测 + 安装

Cell 3 [Markdown]: Bradley-Terry 模型速览

Cell 4 [Code]: 加载偏好数据集
```python
from datasets import load_dataset

# HH-RLHF 数据集（Anthropic 出品）
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:3000]")
print(f"数据集大小: {len(dataset)}")
print(f"样本格式: {list(dataset[0].keys())}")
# chosen 和 rejected 的对话格式
print(f"Chosen: {dataset[0]['chosen'][:200]}...")
print(f"Rejected: {dataset[0]['rejected'][:200]}...")
```

Cell 5 [Code]: 数据预处理
```python
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_preference(example):
    return {
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

dataset = dataset.map(preprocess_preference)
```

Cell 6 [Code]: 使用 trl 训练 Reward Model
```python
from transformers import AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

# Reward Model 用 sequence classification head
reward_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, torch_dtype=torch.float16, device_map="auto"
)

training_args = RewardConfig(
    output_dir="./reward_model_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    max_length=512,
)

trainer = RewardTrainer(
    model=reward_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
print("Reward Model 训练完成")
```

Cell 7 [Code]: 测试 Reward Model
```python
def score_response(reward_model, tokenizer, prompt, response):
    text = prompt + " " + response
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(reward_model.device)
    with torch.no_grad():
        score = reward_model(**inputs).logits[0, 0].item()
    return score

# 对比 chosen vs rejected 的分数
sample = dataset[0]
score_chosen = score_response(reward_model, tokenizer, "", sample["chosen"])
score_rejected = score_response(reward_model, tokenizer, "", sample["rejected"])
print(f"Chosen score: {score_chosen:.4f}")
print(f"Rejected score: {score_rejected:.4f}")
print(f"Chosen > Rejected: {score_chosen > score_rejected}")
```

Cell 8 [Markdown]: 练习 + 延伸阅读

- [ ] **Step 1: Create the complete notebook**

- [ ] **Step 2: Run on Colab T4**

- [ ] **Step 3: Commit**

```bash
git add notebooks/ch06_reward_modeling.ipynb
git commit -m "feat: ch06 reward modeling with trl"
```

---

### Task 11: Ch7 — RLHF (PPO)

**Files:**
- Modify: `notebooks/ch07_rlhf_ppo.ipynb`

**Notebook cell outline:**

Cell 1 [Markdown]: 标题 + 本章目标
```
# 第7章：RLHF (PPO)

## 本章目标
- 理解 PPO (Proximal Policy Optimization) 在 RLHF 中的角色
- 理解 actor-critic 框架在 LLM 训练中的应用
- 使用 trl 的 PPOTrainer 完成 RLHF 训练
```

Cell 2 [Code]: 环境检测 + 安装

Cell 3 [Markdown]: PPO in RLHF 速览 — policy gradient + clipping + KL penalty

Cell 4 [Code]: 准备模型和数据
```python
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import torch

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Policy model (SFT model)
policy_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

# Reference model (frozen, for KL penalty)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
ref_model.eval()

# Reward model (from Ch6)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, torch_dtype=torch.float16, device_map="auto"
)
```

Cell 5 [Code]: PPO 训练
```python
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=2,
    kl_coef=0.1,   # KL penalty coefficient
)

# 数据集：用于生成 prompt
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")

def extract_prompt(example):
    # 提取对话中的 prompt 部分
    text = example["chosen"]
    if "Human:" in text:
        prompt = text.split("Human:")[1].split("Assistant:")[0].strip()
    else:
        prompt = text[:100]
    return {"query": prompt}

dataset = dataset.map(extract_prompt)

# PPOTrainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
)
```

Cell 6 [Code]: PPO 训练循环
```python
from tqdm import tqdm

generation_kwargs = {
    "max_new_tokens": 64,
    "temperature": 0.7,
    "do_sample": True,
}

for epoch in range(1):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        # 1. Generate responses
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

        # 2. Compute rewards
        texts = [tokenizer.decode(q + r) for q, r in zip(query_tensors, response_tensors)]
        reward_inputs = tokenizer(texts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=512).to(reward_model.device)
        with torch.no_grad():
            rewards = reward_model(**reward_inputs).logits[:, 0]
        rewards = [r.item() for r in rewards]

        # 3. PPO update
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
```

Cell 7 [Markdown]: 分析 PPO 训练指标 — reward 变化、KL divergence

Cell 8 [Markdown]: 练习 + 延伸阅读

- [ ] **Step 1: Create the complete notebook**

- [ ] **Step 2: Run on Colab T4**

- [ ] **Step 3: Commit**

```bash
git add notebooks/ch07_rlhf_ppo.ipynb
git commit -m "feat: ch07 RLHF with PPO using trl"
```

---

### Task 12: Ch8 — DPO 直接偏好优化

**Files:**
- Modify: `notebooks/ch08_dpo.ipynb`

**Notebook cell outline:**

Cell 1 [Markdown]: 标题 + 本章目标
```
# 第8章：DPO 直接偏好优化

## 本章目标
- 理解 DPO 相比 RLHF 的优势（无需 Reward Model）
- 理解 DPO loss 的推导思路
- 使用 trl 的 DPOTrainer 完成 DPO 训练
- 对比 DPO vs RLHF 的效果和成本
```

Cell 2 [Code]: 环境检测 + 安装

Cell 3 [Markdown]: DPO 速览 — 直接用偏好数据优化策略，跳过 reward modeling

Cell 4 [Code]: 准备数据和模型
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Policy model
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

# Reference model (frozen)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

# 偏好数据
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:3000]")
```

Cell 5 [Code]: DPO 训练
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)

training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    max_length=512,
    beta=0.1,  # DPO temperature parameter
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)
trainer.train()
print("DPO 训练完成")
```

Cell 6 [Code]: 手动实现 DPO loss（教育目的）
```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    DPO loss: -log σ(β * (log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss

# 验证实现
policy_chosen = torch.tensor([-1.0, -2.0, -1.5])
policy_rejected = torch.tensor([-3.0, -2.5, -4.0])
ref_chosen = torch.tensor([-1.0, -2.0, -1.5])
ref_rejected = torch.tensor([-3.0, -2.5, -4.0])
loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
print(f"DPO loss example: {loss.item():.4f}")
```

Cell 7 [Markdown]: DPO vs RLHF 对比表

Cell 8 [Markdown]: 练习 + 延伸阅读

- [ ] **Step 1: Create the complete notebook**

- [ ] **Step 2: Run on Colab T4**

- [ ] **Step 3: Commit**

```bash
git add notebooks/ch08_dpo.ipynb
git commit -m "feat: ch08 DPO direct preference optimization"
```

---

## Phase 4: Production Chapters (Ch9-10)

### Task 13: Ch9 — 推理优化

**Files:**
- Modify: `notebooks/ch09_inference_optimization.ipynb`

**Notebook cell outline:**

Cell 1 [Markdown]: 标题 + 本章目标
```
# 第9章：推理优化

## 本章目标
- 理解 KV Cache 的原理和实现
- 掌握模型量化的基本方法（4-bit, 8-bit）
- 对比 FP16 vs 量化模型的推理速度和显存占用
```

Cell 2 [Code]: 环境检测 + 安装

Cell 3 [Markdown]: 推理优化速览

Cell 4 [Code]: KV Cache 原理演示
```python
import torch
import torch.nn.functional as F
import time

# 无 KV Cache 的生成（每次重新计算所有 attention）
def generate_no_cache(model, input_ids, max_new_tokens=50):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated)[0]  # 重新计算全部
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated

# 有 KV Cache 的生成（只计算新 token 的 attention）
def generate_with_kv_cache(model, input_ids, max_new_tokens=50):
    generated = input_ids.clone()
    past_key_values = None
    for _ in range(max_new_tokens):
        if past_key_values is None:
            outputs = model(generated, use_cache=True)
        else:
            next_input = generated[:, -1:]
            outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
        logits = outputs[0]
        past_key_values = outputs[1]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated

# 对比速度
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )
    input_ids = tokenizer("Hello, how are you", return_tensors="pt").input_ids.cuda()

    t0 = time.time()
    out_no_cache = generate_no_cache(model, input_ids, max_new_tokens=50)
    t_no_cache = time.time() - t0

    t0 = time.time()
    out_cache = generate_with_kv_cache(model, input_ids, max_new_tokens=50)
    t_cache = time.time() - t0

    print(f"No cache: {t_no_cache:.2f}s")
    print(f"KV cache: {t_cache:.2f}s")
    print(f"Speedup: {t_no_cache/t_cache:.1f}x")
```

Cell 5 [Code]: bitsandbytes 量化 (4-bit)
```python
from transformers import BitsAndBytesConfig

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 对比 FP16 vs 4-bit 的显存占用
model_fp16 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
)
fp16_mem = torch.cuda.memory_allocated() / 1e9
del model_fp16
torch.cuda.empty_cache()

model_4bit = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", quantization_config=bnb_config, device_map="auto"
)
q4_mem = torch.cuda.memory_allocated() / 1e9

print(f"FP16 显存: {fp16_mem:.2f} GB")
print(f"4-bit 显存: {q4_mem:.2f} GB")
print(f"压缩比: {fp16_mem/q4_mem:.1f}x")
```

Cell 6 [Code]: 推理速度 benchmark
```python
def benchmark_inference(model, tokenizer, prompt, num_runs=20, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model.generate(inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return sum(times) / len(times)

prompt = "Explain quantum computing in simple terms."
t_fp16 = benchmark_inference(model_fp16, tokenizer, prompt)
t_4bit = benchmark_inference(model_4bit, tokenizer, prompt)
print(f"FP16 avg: {t_fp16:.3f}s")
print(f"4-bit avg: {t_4bit:.3f}s")
```

Cell 7 [Markdown]: 练习 + 延伸阅读

- [ ] **Step 1: Create the complete notebook**

- [ ] **Step 2: Run on Colab T4**

- [ ] **Step 3: Commit**

```bash
git add notebooks/ch09_inference_optimization.ipynb
git commit -m "feat: ch09 inference optimization - KV cache and quantization"
```

---

### Task 14: Ch10 — 全流程实战

**Files:**
- Modify: `notebooks/ch10_full_pipeline.ipynb`

**Notebook cell outline:**

Cell 1 [Markdown]: 标题 + 本章目标
```
# 第10章：全流程实战

## 本章目标
- 将 Ch1-9 的知识串联成完整的端到端 pipeline
- 从数据准备到模型部署的完整流程
- 最佳实践 checklist

## 全流程概览
```
数据准备 → Tokenizer → Pretrain → SFT → DPO → 量化 → 部署
```
```

Cell 2 [Code]: 环境检测

Cell 3 [Code]: Step 1 - 数据准备
```python
# 加载原始数据 + 清洗 + 格式化
from datasets import load_dataset, concatenate_datasets

raw_data = load_dataset("tatsu-lab/alpaca", split="train[:2000]")
preference_data = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")

print(f"SFT 数据: {len(raw_data)} 样本")
print(f"偏好数据: {len(preference_data)} 样本")
```

Cell 4 [Code]: Step 2 - Base Model + Tokenizer
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
print(f"Base model loaded: {model_name}")
```

Cell 5 [Code]: Step 3 - SFT
```python
from trl import SFTTrainer, SFTConfig

def format_sft(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"}

sft_data = raw_data.map(format_sft)
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./pipeline_sft", num_train_epochs=1,
        per_device_train_batch_size=4, learning_rate=2e-4,
        max_seq_length=512, logging_steps=50, save_strategy="no", report_to="none",
    ),
    train_dataset=sft_data,
    processing_class=tokenizer,
)
trainer.train()
print("Step 3: SFT 完成")
```

Cell 6 [Code]: Step 4 - DPO
```python
from trl import DPOTrainer, DPOConfig

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    ),
    args=DPOConfig(
        output_dir="./pipeline_dpo", num_train_epochs=1,
        per_device_train_batch_size=2, learning_rate=5e-5,
        max_length=512, beta=0.1, logging_steps=50, save_strategy="no", report_to="none",
    ),
    train_dataset=preference_data,
    processing_class=tokenizer,
)
dpo_trainer.train()
print("Step 4: DPO 完成")
```

Cell 7 [Code]: Step 5 - 量化 + 保存
```python
# 保存最终模型
final_model_path = "./pipeline_final"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"模型已保存到 {final_model_path}")

# 量化测试
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16)
quantized = AutoModelForCausalLM.from_pretrained(
    final_model_path, quantization_config=bnb_config, device_map="auto"
)
print("4-bit 量化完成")
```

Cell 8 [Code]: Step 6 - 评估
```python
test_prompts = [
    "What is machine learning?",
    "Explain the concept of attention in transformers.",
    "Write a short poem about AI.",
]

for prompt in test_prompts:
    inputs = tokenizer(f"### Instruction:\n{prompt}\n\n### Response:\n", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

Cell 9 [Markdown]: Best Practices Checklist
```markdown
## Best Practices Checklist

### 数据
- [ ] 数据质量 > 数据数量
- [ ] 去重、清洗、去毒
- [ ] SFT 数据覆盖目标任务的 diversity

### 训练
- [ ] 先 SFT 再 DPO/RLHF
- [ ] LoRA rank 根据任务复杂度调整 (8-64)
- [ ] Learning rate: SFT ~2e-4, DPO ~5e-5
- [ ] 始终监控 train/val loss 曲线

### 评估
- [ ] 自动评估（perplexity, win rate）
- [ ] 人工评估（golden set）
- [ ] 安全性测试（red teaming）
```

Cell 10 [Markdown]: 延伸阅读

- [ ] **Step 1: Create the complete notebook**

- [ ] **Step 2: Run on Colab T4**

- [ ] **Step 3: Commit**

```bash
git add notebooks/ch10_full_pipeline.ipynb
git commit -m "feat: ch10 full pipeline end-to-end training recipe"
```

---

## Phase 5: Deployment

### Task 15: GitHub Actions CI/CD

**Files:**
- Create: `.github/workflows/build.yml`

- [ ] **Step 1: Write `.github/workflows/build.yml`**

```yaml
name: Build JupyterBook

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: |
          pip install jupyter-book
      - name: Build JupyterBook
        run: |
          jupyter-book build .
      - name: Upload artifact
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: actions/upload-pages-artifact@v3
        with:
          path: _build/html

  deploy:
    needs: build
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/build.yml
git commit -m "ci: add JupyterBook build and GitHub Pages deployment"
```

---

### Task 16: Final Build Verification

**Files:**
- None (verification only)

- [ ] **Step 1: Full build test**

```bash
jupyter-book build .
# Expected: build succeeds, all 10 chapters in TOC resolve
```

- [ ] **Step 2: Verify local HTML**

```bash
# Open _build/html/index.html in browser
# Verify: all 10 chapters listed, navigation works
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final build verification"
```

---

## Plan Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Phase 1 | Tasks 1-4 | Project scaffolding, intro, scripts, build verification |
| Phase 2 | Tasks 5-8 | Ch1-4: Architecture, tokenization, pretraining, distributed training |
| Phase 3 | Tasks 9-12 | Ch5-8: SFT, reward modeling, RLHF, DPO |
| Phase 4 | Tasks 13-14 | Ch9-10: Inference optimization, full pipeline |
| Phase 5 | Tasks 15-16 | CI/CD and final verification |

Total: 16 tasks, ~48 steps
