# Advanced Chapters (Ch11-16) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 advanced chapter notebooks (Part IV) to the LLM Training Handbook covering modern architecture, MoE, pretraining engineering, GRPO, long context, and integration.

**Architecture:** Each notebook is a self-contained Jupyter notebook following the existing 7-part structure. Notebooks are theory-heavy (50-70%) with hand-written code demos. Primary reference: DeepSeek series (V2/V3/R1).

**Tech Stack:** PyTorch, matplotlib, trl (Ch14/Ch16), numpy (Ch13). All demos run on CPU except Ch14/Ch16 GRPO sections (Colab T4).

---

## File Structure

| File | Responsibility | Status |
|------|---------------|--------|
| `notebooks/ch11_modern_architecture.ipynb` | RMSNorm, SwiGLU, RoPE, GQA — modernizing GPT-2 | Pending |
| `notebooks/ch12_moe_architecture.ipynb` | MoE routing, load balancing, MLA, DeepSeek-V3 | Pending |
| `notebooks/ch13_pretraining_engineering.ipynb` | MinHash dedup, data mixing, FP8, multi-token prediction | Pending |
| `notebooks/ch14_reasoning_grpo.ipynb` | DeepSeek-R1, GRPO loss, PRM, test-time compute | Pending |
| `notebooks/ch15_long_context_inference.ipynb` | RoPE scaling, Ring Attention, speculative decoding, PagedAttention | Pending |
| `notebooks/ch16_modern_gpt_build.ipynb` | Integration: assemble Ch11-15 into one model, train + compare | Pending |
| `_toc.yml` | Add Part IV section with 6 chapter entries | Pending |

## Notebook Template

Every notebook MUST follow this structure (matching ch01-ch10 pattern):

```
Cell 0:  [markdown] Title + objectives + prerequisites (which chapters to review)
Cell 1:  [code]     Environment setup (Colab detection, pip install, data download)
Cell 2+:  [markdown + code] alternating — theory sections with paper references, then code implementation
Penultimate: [code]  Hands-on experiment or integration demo
Last:    [markdown] Exercises + paper references with arxiv URLs
```

Notebook metadata:
```json
{
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
  }
}
```

Language: Chinese markdown text, English technical terms (RMSNorm, SwiGLU, etc.), Chinese code comments.

---

### Task 1: Ch11 — Modern Transformer Architecture Improvements

**Files:**
- Create: `notebooks/ch11_modern_architecture.ipynb`

**Target cells:** ~22-25 cells (50% theory, 50% code)

- [ ] **Step 1: Write notebook with 5 sections**

Section layout:

| Cell | Type | Content |
|------|------|---------|
| 0 | md | Title `# 第11章：现代 Transformer 架构改进`, objectives, prerequisites (Ch1) |
| 1 | code | Environment setup (Colab/local detection) |
| 2 | md | `## 1. RMSNorm vs LayerNorm` — theory: formula, why drop mean-centering |
| 3 | code | Hand-write `RMSNorm` class (~10 lines): `x * weight / sqrt(mean(x^2) + eps)` |
| 4 | code | Compare LayerNorm vs RMSNorm: same input, show output difference and param count |
| 5 | md | `## 2. SwiGLU vs GELU` — theory: GLU family, Swish activation, gating |
| 6 | code | Hand-write `SwiGLUFFN` class: `W_gate`, `W_up`, `W_down` with SiLU activation |
| 7 | md | `## 3. RoPE (Rotary Position Embedding)` — theory: complex rotation, relative position |
| 8 | code | Hand-write `apply_rotary_emb` function (~30 lines): precompute freqs, apply rotation to Q and K |
| 9 | code | Demo: compare RoPE vs absolute PE attention distribution on same input |
| 10 | md | `## 4. GQA (Grouped Query Attention)` — theory: MHA→GQA→MQA spectrum, KV savings |
| 11 | code | Hand-write `GroupedQueryAttention` class: `n_kv_heads` param, repeat KV heads |
| 12 | code | Show KV cache size comparison: MHA(12h) vs GQA(4kv) vs MQA(1kv) with numbers |
| 13 | md | `## 5. 整合：ModernGPT` — combine all 4 improvements |
| 14 | code | `ModernBlock` class: RMSNorm + GQA + RoPE + SwiGLU |
| 15 | code | `ModernGPT` class: full model with all upgrades |
| 16 | code | Compare param count: GPT-2 config vs Modern config |
| 17 | md | Exercises + references (4 arxiv URLs + Llama source link) |

**Code requirements:**
- RMSNorm: `nn.Parameter(weight)` of shape `(n_embd,)`, no bias, no mean subtraction
- SwiGLU: 3 linear layers `c_gate`, `c_up`, `c_down` with `F.silu(c_gate(x)) * c_up(x)` pattern
- RoPE: precompute `torch.outer(positions, freqs)` → cos/sin → apply rotation `(q * cos) + (rotate_half(q) * sin)`
- GQA: `n_kv_heads` ≤ `n_head`, repeat_interleave K/V to match `n_head` groups
- All baby configs: `n_embd=128, n_head=4, block_size=128` for CPU

- [ ] **Step 2: Verify notebook is valid JSON**

Run: `python -c "import json; json.load(open('notebooks/ch11_modern_architecture.ipynb','r',encoding='utf-8')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify cell count ≥ 17**

Run: `python -c "import json; nb=json.load(open('notebooks/ch11_modern_architecture.ipynb','r',encoding='utf-8')); print(len(nb['cells']))"`
Expected: number ≥ 17

---

### Task 2: Ch12 — MoE Architecture

**Files:**
- Create: `notebooks/ch12_moe_architecture.ipynb`

**Target cells:** ~20-23 cells (60% theory, 40% code)

- [ ] **Step 1: Write notebook with 5 sections**

Section layout:

| Cell | Type | Content |
|------|------|---------|
| 0 | md | Title `# 第12章：MoE 混合专家架构`, objectives, prerequisites (Ch1, Ch11) |
| 1 | code | Environment setup |
| 2 | md | `## 1. 从 Dense 到 Sparse` — FLOPs analysis, why MoE scales params without compute |
| 3 | code | FLOPs comparison: dense 100B vs MoE 100B (10B active) calculation |
| 4 | md | `## 2. 基本 MoE 结构` — Router + N expert FFNs, top-k |
| 5 | code | `MoELayer` class: router Linear + N expert FFNs, top-2 routing (~60 lines) |
| 6 | code | Demo: run MoELayer on random input, show routing distribution |
| 7 | md | `## 3. 负载均衡问题` — expert collapse, auxiliary loss |
| 8 | code | Show collapse: train MoE without balance loss, show all tokens → 1 expert |
| 9 | code | `load_balance_loss` function: `num_experts * sum(f_i * P_i)` |
| 10 | md | `## 4. DeepSeek-V2: MLA (Multi-head Latent Attention)` — compress KV to latent |
| 11 | code | Simplified MLA: `W_kv_down` (compress) + `W_kv_up` (decompress), show KV cache ratio |
| 12 | md | `## 5. DeepSeek-V3: 无辅助损失的负载均衡` — bias-based routing |
| 13 | code | Bias-based routing: add learnable bias to router logits, compare convergence |
| 14 | md | Exercises + references (Switch, DeepSeek-V2/V3, GShard) |

**Code requirements:**
- MoELayer: `num_experts=4`, top-k=2, each expert is a simple 2-layer FFN
- Router: `nn.Linear(n_embd, num_experts)` → softmax → `torch.topk(k=2)`
- Auxiliary loss: `num_experts * sum(f_i * P_i)` where `f_i = tokens_per_expert / total_tokens`
- MLA: down-project K,V to `kv_dim` (e.g., 128→64→128), show cache savings ratio
- Baby configs: `n_embd=128, num_experts=4`

- [ ] **Step 2: Verify notebook is valid JSON + cell count ≥ 14**

---

### Task 3: Ch13 — Large-Scale Pretraining Engineering

**Files:**
- Create: `notebooks/ch13_pretraining_engineering.ipynb`

**Target cells:** ~18-22 cells (70% theory, 30% code)

- [ ] **Step 1: Write notebook with 5 sections**

Section layout:

| Cell | Type | Content |
|------|------|---------|
| 0 | md | Title `# 第13章：大规模预训练工程`, objectives, prerequisites (Ch3, Ch4) |
| 1 | code | Environment setup |
| 2 | md | `## 1. 数据清洗流水线` — MinHash, perplexity filtering, quality, PII |
| 3 | md | MinHash algorithm explanation: shingling → hash → min signature → Jaccard |
| 4 | code | `MinHashDedup` class: shingling, hash functions, signature matrix, Jaccard similarity (~30 lines) |
| 5 | code | Demo: create 5 toy documents, find near-duplicates with MinHash |
| 6 | md | `## 2. 数据混合比例` — domain proportions, DoReMi algorithm |
| 7 | md | DeepSeek-V3 data composition breakdown |
| 8 | md | `## 3. FP8 训练` — E4M3 vs E5M2, scaling factor granularity |
| 9 | code | Simulate FP8 quantization: clamp to E4M3/E5M2 range, show dequantize error |
| 10 | md | `## 4. Multi-token Prediction` — predict N future tokens, training signal efficiency |
| 11 | code | `MultiTokenHead`: shared backbone + N linear heads, `loss = sum(weight_i * loss_i)` |
| 12 | md | `## 5. 课程学习` — dynamic difficulty scheduling |
| 13 | md | Exercises + references (FineWeb, RefinedWeb, DoReMi, FP8, Multi-token) |

**Code requirements:**
- MinHash: `hashlib.sha256`, character shingles, N hash functions via `hash(seed + shingle)`
- FP8 simulation: define E4M3 range [-448, 448] and E5M2 range [-57344, 57344], quantize-dequantize
- Multi-token: baby GPT backbone (2 layers), 2 prediction heads (next-token + next-2-token)

- [ ] **Step 2: Verify notebook is valid JSON + cell count ≥ 13**

---

### Task 4: Ch14 — Reasoning Models and GRPO

**Files:**
- Create: `notebooks/ch14_reasoning_grpo.ipynb`

**Target cells:** ~20-25 cells (60% theory, 40% code)

- [ ] **Step 1: Write notebook with 6 sections**

Section layout:

| Cell | Type | Content |
|------|------|---------|
| 0 | md | Title `# 第14章：推理模型与 GRPO`, objectives, prerequisites (Ch7, Ch8) |
| 1 | code | Environment setup (include `pip install trl peft datasets` for Colab) |
| 2 | md | `## 1. 从 SFT+RLHF 到纯 RL 推理` — R1-Zero, CoT emergence, aha moments |
| 3 | md | Key quotes/observations from DeepSeek-R1 paper about emergent behaviors |
| 4 | md | `## 2. GRPO (Group Relative Policy Optimization)` — eliminate value network |
| 5 | md | Math: GRPO objective derivation, group baseline, clipped surrogate |
| 6 | code | `compute_grpo_loss` function: sample G responses → group normalize → clipped loss (~50 lines) |
| 7 | code | Demo: simulate GRPO with toy rewards, show advantage computation |
| 8 | md | `## 3. Process Reward Model (PRM)` — step-level vs outcome-level reward |
| 9 | code | Compare ORM (1 score) vs PRM (N scores per step) with mock reasoning chain |
| 10 | md | `## 4. R1 训练流水线` — cold-start SFT → RL → rejection sampling → distillation |
| 11 | md | Pipeline diagram in text with 4 stages |
| 12 | md | `## 5. Test-time Compute Scaling` — majority voting, best-of-N, tree search |
| 13 | code | Demo majority voting: generate N solutions, vote on answer |
| 14 | md | `## 6. 实战：GRPO 训练` — trl GRPOTrainer + Qwen2.5-0.5B |
| 15 | code | GRPOTrainer setup: load model, define reward function (math correctness), configure training |
| 16 | code | Note: needs Colab T4, show expected output format |
| 17 | md | Exercises + references (R1, PRM, test-time compute, trl) |

**Code requirements:**
- GRPO loss: generate G=4 mock responses, compute `advantage = (reward - mean) / (std + eps)`, clipped ratio
- PRM vs ORM: mock a 5-step reasoning chain, show PRM scores each step, ORM scores only final
- GRPOTrainer: use `trl.GRPOTrainer` API, `reward_fn` checks `float(answer) == float(target)`
- Graceful no-GPU fallback: wrap in try/except, print "需要 Colab T4 GPU" message

- [ ] **Step 2: Verify notebook is valid JSON + cell count ≥ 17**

---

### Task 5: Ch15 — Long Context and Efficient Inference

**Files:**
- Create: `notebooks/ch15_long_context_inference.ipynb`

**Target cells:** ~20-25 cells (50% theory, 50% code)

- [ ] **Step 1: Write notebook with 5 sections**

Section layout:

| Cell | Type | Content |
|------|------|---------|
| 0 | md | Title `# 第15章：长上下文与高效推理`, objectives, prerequisites (Ch11, Ch9) |
| 1 | code | Environment setup |
| 2 | md | `## 1. RoPE 缩放策略` — PI, NTK-aware, YaRN |
| 3 | code | `positional_interpolation`: scale positions from [0, L_new] to [0, L_train] |
| 4 | code | `ntk_aware_scaling`: adjust base frequency `base * (scale_factor) ** (dim / (dim-2))` |
| 5 | code | `yarn_scaling`: NTK + temperature adjustment on attention |
| 6 | code | Compare: apply all 3 to RoPE, visualize attention patterns with matplotlib heatmap |
| 7 | md | `## 2. 训练时上下文扩展` — progressive 4K→32K→128K schedule |
| 8 | md | `## 3. Ring Attention` — shard sequence across devices |
| 9 | code | Simulate ring attention: 4 "devices" each process a chunk, pass KV in ring order |
| 10 | md | `## 4. Speculative Decoding` — draft model + target model |
| 11 | code | Naive speculative decoding: draft generates k=4 tokens, target verifies in 1 forward pass |
| 12 | code | Calculate acceptance rate and theoretical speedup |
| 13 | md | `## 5. vLLM PagedAttention` — KV cache as virtual memory pages |
| 14 | md | Data structures: block table, physical blocks, virtual blocks, analogy with OS paging |
| 15 | code | Simple block manager simulation: allocate/free KV blocks, show fragmentation avoidance |
| 16 | md | Exercises + references (YaRN, NTK, Ring Attention, Speculative, vLLM) |

**Code requirements:**
- RoPE scaling: use baby model (n_embd=64, n_head=4), show attention heatmap for extended length
- Ring attention: loop over 4 "devices", each computes attention on its chunk, accumulate results
- Speculative decoding: 2 small GPT models (different sizes), draft k tokens, verify with target
- PagedAttention: `BlockManager` class with `allocate`, `free`, `append` methods, page_size=16

- [ ] **Step 2: Verify notebook is valid JSON + cell count ≥ 16**

---

### Task 6: Ch16 — Integration: From GPT-2 to Modern Architecture

**Files:**
- Create: `notebooks/ch16_modern_gpt_build.ipynb`

**Target cells:** ~25-30 cells (20% theory, 80% code)

- [ ] **Step 1: Write notebook with 5 sections**

Section layout:

| Cell | Type | Content |
|------|------|---------|
| 0 | md | Title `# 第16章：综合实战 — 从 GPT-2 到现代架构`, objectives, prerequisites (Ch11-15) |
| 1 | code | Environment setup (include data download for Shakespeare) |
| 2 | md | `## 1. 架构升级` — 4 improvements from Ch11 |
| 3 | code | Copy `RMSNorm`, `SwiGLUFFN`, `apply_rotary_emb`, `GroupedQueryAttention` from Ch11 |
| 4 | code | `ModernBlock`: RMSNorm + GQA + RoPE + SwiGLU |
| 5 | code | `ModernGPT`: token emb (no position emb — RoPE handles it) + N × ModernBlock + RMSNorm + head |
| 6 | code | Verify: forward pass on random input, compare params vs GPT-2 |
| 7 | md | `## 2. 添加 MoE` — from Ch12 |
| 8 | code | `ModernMoEGPT`: replace SwiGLUFFN with MoELayer in ModernBlock |
| 9 | code | Compare params: ModernGPT (dense) vs ModernMoEGPT (sparse, same active params) |
| 10 | md | `## 3. 训练对比` — same data, same budget |
| 11 | code | Load Shakespeare data, create character-level encoding |
| 12 | code | Define training loop: AdamW, cosine LR, gradient accumulation |
| 13 | code | Train GPT-2 baseline (baby config) for 500 steps, record loss every 50 |
| 14 | code | Train ModernGPT (baby config) for 500 steps, record loss every 50 |
| 15 | code | Plot loss curves comparison with matplotlib |
| 16 | md | Analysis: convergence speed, final loss, parameter efficiency |
| 17 | md | `## 4. GRPO 对齐` — from Ch14 |
| 18 | code | GRPOTrainer setup code for ModernGPT (conceptual, needs T4) |
| 19 | md | `## 5. 路线图` — 16-chapter summary, future directions |
| 20 | md | Summary table: Ch1-10 fundamentals, Ch11-15 advanced, next steps |
| 21 | md | Future: distillation, multimodal, agents, on-device, domain adaptation |
| 22 | md | Exercises + references (all Ch11-15 papers) |

**Code requirements:**
- All components from Ch11 must be self-contained (copy, not import)
- Training config: `n_embd=64, n_head=4, n_layer=4, block_size=64, batch_size=32, lr=3e-4, steps=500`
- Data: `wget` tiny-shakespeare, character-level encoding (65 chars)
- Loss plot: matplotlib line chart with legend, title, axis labels
- MoE: `num_experts=4, top_k=2`

- [ ] **Step 2: Verify notebook is valid JSON + cell count ≥ 22**

---

### Task 7: Update _toc.yml

**Files:**
- Modify: `_toc.yml`

- [ ] **Step 1: Add Part IV section**

Append after the existing Part III chapters:

```yaml
  - caption: Part IV - 进阶：走向 SOTA
    chapters:
      - file: notebooks/ch11_modern_architecture
        title: "第11章：现代 Transformer 架构改进"
      - file: notebooks/ch12_moe_architecture
        title: "第12章：MoE 混合专家架构"
      - file: notebooks/ch13_pretraining_engineering
        title: "第13章：大规模预训练工程"
      - file: notebooks/ch14_reasoning_grpo
        title: "第14章：推理模型与 GRPO"
      - file: notebooks/ch15_long_context_inference
        title: "第15章：长上下文与高效推理"
      - file: notebooks/ch16_modern_gpt_build
        title: "第16章：综合实战 — 从 GPT-2 到现代架构"
```

- [ ] **Step 2: Verify _toc.yml is valid YAML**

Run: `python -c "import yaml; yaml.safe_load(open('_toc.yml')); print('YAML OK')"`
Expected: `YAML OK`

---

### Task 8: Build Verification

- [ ] **Step 1: Install jupyter-book and build**

Run: `pip install "jupyter-book>=1.0.0,<2.0.0" && jupyter-book build .`
Expected: Build succeeds, `_build/html/` contains 16+ HTML files

- [ ] **Step 2: Verify all 16 chapters are in the build output**

Run: `find _build/html -name "notebooks/*.html" | sort`
Expected: ch01 through ch16 HTML files

---

### Task 9: Commit and Push

- [ ] **Step 1: Stage all new files**

```bash
git add notebooks/ch11_modern_architecture.ipynb \
        notebooks/ch12_moe_architecture.ipynb \
        notebooks/ch13_pretraining_engineering.ipynb \
        notebooks/ch14_reasoning_grpo.ipynb \
        notebooks/ch15_long_context_inference.ipynb \
        notebooks/ch16_modern_gpt_build.ipynb \
        _toc.yml
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat: add Part IV advanced chapters (Ch11-16) — modern architecture, MoE, GRPO, long context"
```

- [ ] **Step 3: Push to GitHub (trigger Pages rebuild)**

```bash
git -c http.proxy="" -c https.proxy="" push
```

---

## Quality Checklist (Review Agent Output Against This)

After subagents complete, verify each notebook against these criteria:

1. **Paper references:** Every technical claim has an explicit arxiv URL (no hallucinated references)
2. **Code runs on CPU:** Ch11-13, Ch15 demos must work without GPU; Ch14/Ch16 GRPO sections need T4 but have fallback
3. **Cell structure:** First cell = markdown title+objectives, second cell = code environment setup, last cell = markdown references
4. **Language:** Chinese markdown, English technical terms, Chinese code comments
5. **No placeholders:** No "TODO", "TBD", "implement later" in any cell
6. **Self-contained:** Each notebook can be run independently (copy code, don't import from other notebooks)
7. **Baby configs:** All models use small configs that fit in CPU memory
8. **Content accuracy:** RMSNorm formula, SwiGLU pattern, RoPE rotation, GQA KV repetition are all correct
