# LLM Training Handbook - Design Spec

## Overview

A JupyterBook-based learning website targeting AI algorithm engineers who have deep learning experience (CNN, RNN, Transformer, supervised learning) but want to learn LLM-specific paradigms: pretraining, SFT, RLHF, DPO, and inference optimization.

**Core philosophy**: Hands-on practice over theory. Each chapter is 70%+ executable code with analysis. Concepts are kept to 1-2 paragraphs per chapter with paper links for deeper reading.

**Language**: Chinese primary, English technical terms.

**Repository**: Independent project (new repo), with nanoGPT cloned as a git submodule or referenced via pip install for Ch1-4. Ch5-10 reference trl/transformers/peft via pip.

## Project Structure

```
llm-training-handbook/           # Independent repository
├── _config.yml                  # JupyterBook configuration
├── _toc.yml                     # Table of contents (chapter navigation)
├── intro.md                     # Preface + environment setup guide
├── requirements.txt             # Python dependencies
├── notebooks/
│   ├── ch01_gpt_architecture.ipynb
│   ├── ch02_tokenization.ipynb
│   ├── ch03_pretraining.ipynb
│   ├── ch04_distributed_training.ipynb
│   ├── ch05_sft.ipynb
│   ├── ch06_reward_modeling.ipynb
│   ├── ch07_rlhf_ppo.ipynb
│   ├── ch08_dpo.ipynb
│   ├── ch09_inference_optimization.ipynb
│   └── ch10_full_pipeline.ipynb
├── data/                        # Small datasets (Shakespeare, etc.)
├── scripts/                     # Helper scripts (env check, data download)
└── assets/                      # Images, diagrams
```

## Chapter Breakdown (10 chapters)

### Ch1: GPT Architecture Deep Dive
- **Content**: Line-by-line breakdown of nanoGPT's model.py. Hand-write CausalSelfAttention, MLP, Block, GPT class. Compare with GPT-2 official implementation.
- **Output**: A working GPT assembled from scratch.
- **Reference project**: nanoGPT
- **Hardware**: Any GPU / CPU OK (small model)

### Ch2: Tokenization & Data Engineering
- **Content**: BPE principles, tiktoken usage, build a mini tokenizer from scratch, data cleaning and mixing ratios.
- **Output**: Custom tokenization pipeline for a custom dataset.
- **Reference project**: nanoGPT data/
- **Hardware**: CPU OK

### Ch3: Pretraining in Practice
- **Content**: Train GPT on Shakespeare/OpenWebText, loss curve analysis, scaling laws experiments, learning rate scheduling.
- **Output**: A converged pretrained model checkpoint.
- **Reference project**: nanoGPT train.py
- **Hardware**: Colab T4 (local 2060S OK for Shakespeare)

### Ch4: Distributed Training & Efficiency
- **Content**: DDP principles, mixed precision (FP16/BF16), Flash Attention, gradient accumulation, MFU analysis. Single-GPU simulation for multi-GPU concepts.
- **Output**: Optimized training config + performance benchmark.
- **Reference project**: nanoGPT
- **Hardware**: Colab T4 (single GPU simulation for distributed concepts)

### Ch5: SFT (Instruction Tuning)
- **Content**: Instruction dataset construction (Alpaca format), loss masking (compute loss only on response tokens), LoRA/QLoRA fine-tuning with trl's SFTTrainer. Start by modifying nanoGPT train.py to support instruction data, then show how trl automates this.
- **Output**: A model that can follow instructions.
- **Reference project**: trl SFTTrainer + nanoGPT train.py as conceptual baseline
- **Hardware**: Colab T4 (LoRA on Qwen2.5-0.5B or SmolLM-135M)

### Ch6: Reward Modeling
- **Content**: Preference data collection & format, Bradley-Terry model, training a Reward Model.
- **Output**: A trained Reward Model.
- **Reference project**: trl
- **Hardware**: Colab T4

### Ch7: RLHF (PPO)
- **Content**: PPO algorithm principles, actor-critic framework, KL penalty, RLHF training loop.
- **Output**: RLHF-aligned model.
- **Reference project**: trl PPOTrainer
- **Hardware**: Colab T4 (small model + LoRA)

### Ch8: DPO (Direct Preference Optimization)
- **Content**: DPO vs RLHF comparison, DPO loss derivation & implementation, practical tuning.
- **Output**: DPO-aligned model.
- **Reference project**: trl DPOTrainer
- **Hardware**: Colab T4

### Ch9: Inference Optimization
- **Content**: KV cache principles with code demo, quantization fundamentals (GGUF format, bitsandbytes), inference speed benchmark (latency/throughput). Focus on practical: load a model, quantize it, measure speedup.
- **Output**: Quantized model + inference benchmark comparing FP16 vs quantized.
- **Reference project**: llama.cpp (GGUF quantization) + bitsandbytes (on-GPU quantization)
- **Hardware**: Colab T4

### Ch10: Full Pipeline
- **Content**: End-to-end pipeline from data preparation through alignment to deployment. Best practices checklist.
- **Output**: Reproducible training recipe.
- **Reference project**: Combined
- **Hardware**: Colab T4 (using small models throughout)

## Notebook Design Pattern

Every notebook follows this internal structure:
```
1. [Markdown] Chapter objectives + prerequisite knowledge checklist
2. [Markdown] Environment install cell (Colab/local differences marked)
3. [Code] imports & configuration
4. [Markdown] Core concept speed-run (1-2 paragraphs, paper links)
5. [Code + Markdown] Hands-on content (alternating code + analysis cells)
6. [Code] Exercises (optional challenge tasks)
7. [Markdown] Further reading + recommended GitHub projects
```

## Colab Compatibility Strategy

**Primary target: Colab T4 (16GB VRAM)**. All experiments must pass "Run All" on Colab T4.

- First cell of every notebook: detect runtime environment
- Colab: auto `pip install` dependencies + download data
- Local: assumes setup per `intro.md`
- Large models (>1B): provide HuggingFace pretrained weight loading paths, don't require local training
- Distributed training (Ch4): single-GPU simulation + conceptual explanation, no multi-GPU requirement
- All fine-tuning uses LoRA/QLoRA to fit in 16GB VRAM

## Technology Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Site generator | JupyterBook | Native .ipynb support, standard in scientific computing |
| DL framework | PyTorch 2.x | nanoGPT is PyTorch-based |
| SFT/RLHF/DPO | trl + transformers + peft | HuggingFace ecosystem, industry standard |
| Tokenization | tiktoken | Used by nanoGPT, high performance |
| Distributed training | PyTorch DDP + FSDP | Native PyTorch support |
| Inference optimization | llama.cpp + bitsandbytes | Quantization (GGUF on-CPU, bitsandbytes on-GPU) |
| Deployment | GitHub Pages | Free, native JupyterBook support |
| CI/CD | GitHub Actions | Auto-build on push |

## Dataset Strategy

| Chapter | Dataset | Size | Colab T4 |
|---------|---------|------|----------|
| Ch1-2 | tiny-shakespeare | 1MB | Yes |
| Ch3 | OpenWebText subset (100k docs) | ~500MB | Yes (subset) |
| Ch5 SFT | Alpaca/Guanaco format instruction data | ~50MB | Yes |
| Ch6 Reward | Preference data (HH-RLHF subset) | ~100MB | Yes |
| Ch7-8 | Same as Ch6 preference data + SFT model | same | Yes (small model) |
| Ch9 | Pre-trained models for quantization benchmark | - | Yes |

## Model Size Strategy

- **Ch1-4**: nanoGPT baby-GPT (~10M params) to GPT-2 small (124M)
- **Ch5-8**: Qwen2.5-0.5B or SmolLM-135M as base model (HuggingFace pretrained weights) + LoRA
- **Ch9**: Quantization benchmark on 1-3B models
- **Principle**: All experiments runnable on Colab T4 (16GB) or RTX 2060S (8GB, with adjustments)

## Deployment

- JupyterBook build → GitHub Pages
- Repository structured for `jupyter-book build .` one-command build
- GitHub Actions auto-build on push to main

## Success Criteria

1. Every notebook passes "Run All" on Colab T4 without errors
2. JupyterBook builds cleanly, GitHub Pages accessible
3. 10 chapters cover pretrain → SFT → RLHF → DPO → inference complete pipeline
4. 70%+ of each notebook is executable code + analysis
