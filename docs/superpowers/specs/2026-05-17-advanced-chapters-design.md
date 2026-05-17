# LLM Training Handbook — Advanced Chapters (Part IV) Design Spec

## Overview

6 advanced chapters (Ch11-16) extending the existing LLM Training Handbook. These chapters cover the gap between the GPT-2-style fundamentals taught in Part I-III and current SOTA architectures and methods (DeepSeek-V3, R1, Llama 3).

**Philosophy shift from main chapters:** More theory (50-70%) than Part I-III, but still code-backed. Code style is "read source + hand-write" hybrid: core concepts hand-written, complex components reference existing open-source code.

**Primary reference model:** DeepSeek series (V2/V3/R1)

**Language:** Chinese primary, English technical terms.

**Hardware target:** Most chapters work on CPU for concept demos; GRPO experiment (Ch14) needs Colab T4.

---

## File Map (additions to existing project)

| File | Responsibility |
|------|---------------|
| `notebooks/ch11_modern_architecture.ipynb` | RMSNorm, SwiGLU, RoPE, GQA — modernizing GPT-2 |
| `notebooks/ch12_moe_architecture.ipynb` | Mixture-of-Experts, MLA, DeepSeek-V2/V3 innovations |
| `notebooks/ch13_pretraining_engineering.ipynb` | Data engineering, FP8, multi-token prediction |
| `notebooks/ch14_reasoning_grpo.ipynb` | DeepSeek-R1, GRPO, Process Reward Models |
| `notebooks/ch15_long_context_inference.ipynb` | RoPE scaling, Ring Attention, speculative decoding, PagedAttention |
| `notebooks/ch16_modern_gpt_build.ipynb` | Integration: assemble Ch11-15 improvements into one model |

**_toc.yml update:** Add Part IV with 6 chapters.

---

## Chapter Breakdown

### Ch11: Modern Transformer Architecture Improvements

**Objective:** Upgrade the Ch1 GPT-2 architecture to 2024-2025 mainstream design.

**Content (50% theory, 50% code):**

1. **RMSNorm vs LayerNorm**
   - Theory: Why RMSNorm drops the mean-centering (faster, similar quality)
   - Code: Hand-write RMSNorm (~10 lines)
   - Reference: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019)

2. **SwiGLU vs GELU**
   - Theory: GLU family, why gating helps, Swish activation
   - Code: Hand-write SwiGLU FFN layer
   - Reference: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020)

3. **RoPE (Rotary Position Embedding)**
   - Theory: Complex rotation intuition, why it enables length extrapolation, vs absolute/relative PE
   - Code: Hand-write RoPE application (~30 lines), compare with absolute PE
   - Reference: [RoFormer](https://arxiv.org/abs/2104.09864) (Su et al., 2021)

4. **GQA (Grouped Query Attention)**
   - Theory: MHA → GQA → MQA spectrum, KV cache savings
   - Code: Modify Ch1 CausalSelfAttention to support GQA
   - Reference: [GQA](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)

5. **Integration**
   - Assemble all 4 improvements into the Ch1 GPT class
   - Read HuggingFace `LlamaModel` source for comparison: [transformers/models/llama/modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

**Hardware:** CPU OK (small model verification)

---

### Ch12: MoE (Mixture-of-Experts) Architecture

**Objective:** Understand the core architectural innovation of DeepSeek-V2/V3.

**Content (60% theory, 40% code):**

1. **Dense to Sparse motivation**
   - Theory: Why MoE scales parameters without proportional compute cost
   - FLOPs analysis: MoE model with 100B params can run at 10B dense FLOPs

2. **Basic MoE structure**
   - Theory: Router (gating network) + N expert FFNs, top-k routing
   - Code: Hand-write simple top-2 MoE layer (~60 lines)
   - Reference: [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021)

3. **Load balancing problem**
   - Theory: Why experts collapse (all tokens go to one expert), auxiliary loss
   - Code: Implement auxiliary load balancing loss
   - Reference: Switch Transformers Section 2.2

4. **DeepSeek-V2: MLA (Multi-head Latent Attention)**
   - Theory: Compress K/V to low-dim latent, dramatically reduce KV cache
   - Code: Read DeepSeek-V2 model source, annotate key MLA sections
   - Reference: [DeepSeek-V2](https://arxiv.org/abs/2405.04434) Section 2.1

5. **DeepSeek-V3: Auxiliary-loss-free load balancing**
   - Theory: Bias-based routing instead of auxiliary loss, why it improves quality
   - Reference: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) Section 2.3
   - Source: [github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)

**Hardware:** CPU OK (small expert count demo)

---

### Ch13: Large-Scale Pretraining Engineering

**Objective:** Understand why "80% of pretraining is data engineering."

**Content (70% theory, 30% code):**

1. **Data cleaning pipeline**
   - Theory: MinHash deduplication, perplexity filtering, quality classifiers, PII removal
   - Code: Hand-write MinHash dedup (~30 lines)
   - Reference: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [RefinedWeb](https://arxiv.org/abs/2306.01116) (Penedo et al., 2023)

2. **Data mixing ratios**
   - Theory: How domain proportions (code/math/web/books) affect model capabilities
   - Reference: DoReMi ([arxiv.org/abs/2305.10429](https://arxiv.org/abs/2305.10429)), DeepSeek-V3 Section 3.1

3. **FP8 training**
   - Theory: E4M3 (forward) vs E5M2 (backward) formats, scaling factor granularity (per-tensor/block/channel)
   - Code: Demo `torch.float8_linear` or `transformers` FP8 quantization on a small matmul
   - Reference: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (Micikevicius et al., 2022), DeepSeek-V3 Section 3.3

4. **Multi-token prediction**
   - Theory: Predict N future tokens simultaneously, improves training signal efficiency
   - Code: Implement dual-head prediction (next-token + next-2-token) on small GPT
   - Reference: [Multi-token Prediction](https://arxiv.org/abs/2404.19737) (Gloeckle et al., 2024)

5. **Curriculum learning**
   - Theory: Dynamic data difficulty/domain scheduling during training
   - Reference: DeepSeek-V3 training details

**Hardware:** CPU for data engineering demos; FP8 needs H100/A100 (concept-focused, not hands-on)

---

### Ch14: Reasoning Models and GRPO

**Objective:** Understand DeepSeek-R1's breakthrough — training reasoning via pure RL.

**Content (60% theory, 40% code):**

1. **From SFT+RLHF to pure RL reasoning**
   - Theory: DeepSeek-R1-Zero experiment — skip SFT, directly RL-train reasoning
   - Emergence of Chain-of-Thought, self-correction, "aha moments"
   - Reference: [DeepSeek-R1](https://arxiv.org/abs/2501.12948) Section 2

2. **GRPO (Group Relative Policy Optimization)**
   - Theory: Eliminate Value Network — use group baseline instead. Math derivation of GRPO objective
   - Code: Hand-write simplified GRPO loss (~50 lines)
   - Reference: DeepSeek-R1 Section 2.3, [trl GRPOTrainer](https://huggingface.co/docs/trl/main_classes/grpo_trainer)

3. **Process Reward Model (PRM)**
   - Theory: Score each reasoning step vs Outcome RM (score final answer only)
   - Reference: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman et al., 2023)

4. **R1 training pipeline**
   - Theory: Multi-stage: cold-start SFT → RL (GRPO) → rejection sampling → distillation
   - Reference: DeepSeek-R1 Section 3

5. **Test-time compute scaling**
   - Theory: More inference compute (search, verification) → better results
   - Reference: [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314) (Snell et al., 2024)

6. **Hands-on:** Use trl `GRPOTrainer` to train a small model on math tasks (Colab T4)

**Hardware:** Colab T4 (GRPO + Qwen2.5-0.5B + LoRA + GSM8K math problems)

---

### Ch15: Long Context and Efficient Inference

**Objective:** Understand the engineering behind 4K → 128K → 1M context extension.

**Content (50% theory, 50% code):**

1. **RoPE scaling strategies**
   - Theory: NTK-aware interpolation, YaRN, positional interpolation — how each extends context
   - Code: Implement and compare 3 scaling strategies on attention distribution (~40 lines each)
   - Reference: [YaRN](https://arxiv.org/abs/2309.00071), [NTK-aware](https://arxiv.org/abs/2306.15595)

2. **Training-time context extension**
   - Theory: Gradually increase context length during training (4K → 32K → 128K)
   - Reference: Llama 3.1 technical report, DeepSeek-V3 training details

3. **Ring Attention**
   - Theory: Shard sequences across devices, achieving "infinite context"
   - Code: Conceptual implementation of ring attention pattern
   - Reference: [Ring Attention](https://arxiv.org/abs/2310.01889) (Liu et al., 2023)

4. **Speculative Decoding**
   - Theory: Small model drafts, large model verifies — speedup without quality loss
   - Code: Implement naive speculative decoding with draft/target pair
   - Reference: [Speculative Decoding](https://arxiv.org/abs/2302.01318) (Leviathan et al., 2023)

5. **vLLM PagedAttention**
   - Theory: KV cache memory management inspired by OS virtual memory paging
   - Code: Read vLLM `block_manager.py` source, annotate key data structures
   - Reference: [vLLM](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023), [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

**Hardware:** CPU for concept demos; long context experiments need Colab T4

---

### Ch16: Integration — From GPT-2 to Modern Architecture

**Objective:** Assemble Ch11-15 improvements into a single model, compare with GPT-2 baseline.

**Content (20% theory, 80% code):**

1. **Architecture upgrade**
   - Take Ch1 GPT class, apply: LayerNorm→RMSNorm, GELU→SwiGLU, abs PE→RoPE, MHA→GQA
   - Each upgrade from Ch11, integrated into one coherent model class

2. **Add MoE**
   - Replace FFN layers with MoE structure from Ch12

3. **Training comparison**
   - Same data (Shakespeare), same compute budget
   - Compare loss curves: GPT-2 architecture vs modern architecture
   - Analyze: does modern architecture converge faster/lower?

4. **GRPO alignment**
   - Apply GRPO from Ch14 to the modern architecture model
   - Compare reasoning capability before/after alignment

5. **Roadmap**
   - Summary of the full 16-chapter journey
   - Directions for continued learning: model distillation, multimodal, agent frameworks

**Hardware:** Colab T4 (LoRA + small model + GRPO)

---

## Notebook Design Pattern

Same 7-part structure as Part I-III, but adjusted for theory-heavy content:

```
1. [Markdown] Chapter objectives + prerequisites (which main chapters to review)
2. [Code] Environment setup (same Colab/local detection)
3. [Markdown] Theory deep-dive with paper references and key equations
4. [Code] Concept implementation (hand-written or source reading)
5. [Markdown] Analysis and intuition — why this improvement matters
6. [Code] Hands-on experiment (where feasible)
7. [Markdown] Paper references + recommended reading
```

## _toc.yml Addition

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

## Success Criteria

1. Every technical claim has an explicit paper reference (arxiv URL)
2. Code demos run on CPU (Ch11-13, Ch15) or Colab T4 (Ch14, Ch16)
3. Each chapter clearly shows what changed vs GPT-2 and why
4. All 6 chapters build on each other progressively
5. Ch16 successfully integrates all improvements into a working model
