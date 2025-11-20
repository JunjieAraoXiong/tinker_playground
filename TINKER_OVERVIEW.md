# Tinker: Complete Overview

A comprehensive guide to understanding Tinker, its features, and what you can do with it.

---

## What is Tinker?

**Tinker** is a hosted API service for fine-tuning Large Language Models (LLMs) created by Thinking Machines Lab (founded by Mira Murati, former OpenAI CTO).

### The Core Idea

You write simple Python code on your **CPU-only machine**, and Tinker handles all the complexity of distributed GPU training.

```
You Focus On                    Tinker Handles
─────────────                   ──────────────
- Your data                     - Distributed training
- Your algorithms               - GPU management
- Your training loop            - Hardware failures
- Your loss functions           - Scaling
```

### How It Works

1. You write a Python script with API calls
2. Tinker executes the heavy GPU computation
3. You get back trained model weights

```python
import tinker

# Create training client
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.2-1B"
)

# Train
training_client.forward_backward(data, "cross_entropy")
training_client.optim_step(tinker.AdamParams(learning_rate=1e-4))

# Sample from trained model
sampling_client = training_client.save_weights_and_get_sampling_client(name="my_model")
result = sampling_client.sample(prompt="Hello!")
```

---

## Core Features

### 1. LoRA Fine-Tuning

Tinker uses **Low-Rank Adaptation (LoRA)** instead of full fine-tuning:
- 10-100x fewer trainable parameters
- Same performance as full fine-tuning for most tasks
- Much lower cost and faster training

### 2. Multiple Loss Functions

Built-in loss functions for different training objectives:

| Loss Function | Type | Use Case |
|--------------|------|----------|
| `cross_entropy` | Supervised Learning | Instruction tuning, chat, distillation |
| `importance_sampling` | Reinforcement Learning | RLVR, reasoning tasks |
| `ppo` | Reinforcement Learning | On-policy training |
| `forward_backward_custom` | Custom | DPO, research, novel objectives |

### 3. Async Training

- Submit `forward_backward` and `optim_step` asynchronously
- Pipeline batches for maximum throughput
- Supports "off-by-K" async RL for higher throughput

### 4. Weight Management

- **Save/Load** model states and optimizer states
- **Download** weights to use with your own inference provider
- **Publish** models for sharing

### 5. Multi-Tenant Efficiency

- Shared GPU pools with synchronized "clock cycles"
- Efficient even with small batch sizes
- You only pay for compute used

---

## Supported Models

### Full Model Lineup

| Model | Type | Architecture | Size |
|-------|------|--------------|------|
| **Qwen/Qwen3-235B-A22B-Instruct** | Instruction | MoE | Large (235B) |
| **Qwen/Qwen3-30B-A3B-Instruct** | Instruction | MoE | Medium (30B) |
| **Qwen/Qwen3-30B-A3B** | Hybrid | MoE | Medium (30B) |
| **Qwen/Qwen3-30B-A3B-Base** | Base | MoE | Medium (30B) |
| **Qwen/Qwen3-32B** | Hybrid | Dense | Medium (32B) |
| **Qwen/Qwen3-8B** | Hybrid | Dense | Small (8B) |
| **Qwen/Qwen3-8B-Base** | Base | Dense | Small (8B) |
| **Qwen/Qwen3-4B-Instruct** | Instruction | Dense | Compact (4B) |
| **meta-llama/Llama-3.3-70B** | Instruction | Dense | Large (70B) |
| **meta-llama/Llama-3.1-70B** | Base | Dense | Large (70B) |
| **meta-llama/Llama-3.1-8B** | Base | Dense | Small (8B) |
| **meta-llama/Llama-3.1-8B-Instruct** | Instruction | Dense | Small (8B) |
| **meta-llama/Llama-3.2-3B** | Base | Dense | Compact (3B) |
| **meta-llama/Llama-3.2-1B** | Base | Dense | Compact (1B) |
| **DeepSeek-V3.1** | - | MoE | Large |

### Model Selection Guide

- **For prototyping**: Llama-3.2-1B (cheapest)
- **For production**: Qwen3-30B-A3B (MoE) - 30B quality at 3B cost
- **For fast inference**: Instruction models (no chain-of-thought)
- **For best reasoning**: Hybrid models (can use chain-of-thought)
- **For research**: Base models (full post-training control)

### Why MoE Models are Recommended

MoE (Mixture of Experts) models activate only a subset of parameters:
- **Qwen3-30B-A3B**: 30B total, but only 3B active
- **Cost**: Same as a 3B dense model
- **Performance**: Close to 30B dense model

---

## Training Methods

### 1. Supervised Learning (SL)

**What it does**: Train model to produce specific outputs for given inputs

**Loss function**: `cross_entropy`

**Use cases**:
- Instruction tuning (teach model to follow instructions)
- Chat fine-tuning (improve conversation quality)
- Prompt distillation (compress long prompts into model behavior)
- Domain adaptation (specialize for medical, legal, etc.)

**Expected improvement**: 20-50% accuracy gain on target tasks

**Example**:
```bash
python -m tinker_cookbook.recipes.sl_basic \
    model_name=meta-llama/Llama-3.2-1B \
    dataset_path=my_conversations.jsonl \
    log_path=/tmp/sl_training
```

---

### 2. Reinforcement Learning (RL)

**What it does**: Train model through trial-and-error with reward signals

**Loss functions**: `importance_sampling` or `ppo`

**Use cases**:
- **RLVR** (RL with Verifiable Rewards): Math, coding, factual tasks
- **Reasoning**: Teach chain-of-thought thinking
- **Tool use**: Train to use APIs and tools correctly
- **Multi-agent**: Games, debates, negotiations

**Expected improvement**: 30-100% on reasoning tasks

**Example**:
```bash
python -m tinker_cookbook.recipes.rl_basic \
    model_name=meta-llama/Llama-3.1-8B \
    log_path=/tmp/rl_training
```

---

### 3. Direct Preference Optimization (DPO)

**What it does**: Train model to prefer better responses over worse ones

**Loss function**: Custom (via `forward_backward_custom`)

**Use cases**:
- Align with human preferences
- Improve helpfulness, harmlessness, honesty
- Style and tone alignment
- Quality improvements

**Expected improvement**: 10-30% preference win rate vs base model

**Key parameters**:
- `dpo_beta`: Start with 0.1
- `learning_rate`: 1e-5 to 1e-6 (lower than SL)

**Example**:
```bash
python -m tinker_cookbook.preference.train_dpo_cli \
    model_name=meta-llama/Llama-3.2-1B \
    dataset=hhh \
    dpo_beta=0.1 \
    learning_rate=1e-5 \
    log_path=/tmp/dpo_training
```

**Available DPO datasets**:
- `hhh`: Anthropic's Helpful-Harmless-Honest
- `helpsteer3`: NVIDIA's HelpSteer3
- `ultrafeedback`: UltraFeedback binarized preferences

---

### 4. RLHF (RL from Human Feedback)

**What it does**: Full preference learning pipeline

**Three stages**:
1. **SFT**: Supervised fine-tuning warm-start
2. **Reward Model**: Train a preference model
3. **RL**: Optimize policy against reward model

**Use cases**:
- Production-grade alignment
- Research on post-training pipelines

**Example**:
```bash
python -m tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline \
    log_path=/tmp/rlhf_experiment
```

---

### 5. Custom Loss Functions

**What it does**: Define arbitrary differentiable objectives

**Use cases**:
- Research on novel training objectives
- Bradley-Terry preference models
- Contrastive learning
- Multi-sequence losses

**Example**:
```python
def my_custom_loss(data, logprobs):
    loss = (logprobs ** 2).sum()
    return loss, {"my_metric": loss.item()}

training_client.forward_backward_custom(data, my_custom_loss)
```

---

## Deep Dive: How Each Training Method Works

This section provides detailed explanations of the algorithms, with diagrams, math, and references.

---

### Supervised Learning (SL) - Detailed Explanation

#### The Core Idea

Supervised learning teaches a model to mimic examples. You show the model input-output pairs, and it learns to produce similar outputs for similar inputs.

```
┌─────────────────────────────────────────────────────────┐
│                  SUPERVISED LEARNING                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Training Data:                                         │
│   ┌─────────────┐      ┌─────────────┐                  │
│   │   Input     │ ──→  │   Output    │                  │
│   │  "2 + 2 ="  │      │    "4"      │                  │
│   └─────────────┘      └─────────────┘                  │
│                                                          │
│   Model learns: P(output | input)                        │
│                                                          │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│   │  Input  │ →  │  Model  │ →  │ Output  │            │
│   └─────────┘    └─────────┘    └─────────┘            │
│                       ↑                                  │
│                  Update weights                          │
│                  to minimize                             │
│                  prediction error                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### The Math: Cross-Entropy Loss

The model learns by minimizing the **negative log-likelihood** of the correct tokens:

**LaTeX Formula:**
```latex
\mathcal{L}(\theta) = -\sum_{i=1}^{N} w_i \cdot \log P_\theta(x_i | x_{<i})
```

**Rendered:**
$$
\mathcal{L}(\theta) = -\sum_{i=1}^{N} w_i \cdot \log P_\theta(x_i | x_{<i})
$$

**Where:**
- $\theta$ = model parameters
- $w_i$ = weight for token $i$ (0 for prompt, 1 for completion)
- $P_\theta(x_i | x_{<i})$ = model's predicted probability of token $x_i$ given all previous tokens
- $N$ = total number of tokens in the sequence

**In plain English**: The model is penalized when it assigns low probability to the correct next token. It learns to assign high probability to the tokens in your training examples.

#### Training Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    SL TRAINING LOOP                           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Load batch of examples                                    │
│     ┌─────────────────────────────────┐                      │
│     │ [User: What is 2+2?]            │                      │
│     │ [Assistant: 2+2 equals 4.]      │  ← Train on this     │
│     └─────────────────────────────────┘                      │
│                     ↓                                         │
│  2. Forward pass: compute probabilities                       │
│     Model predicts: P("4") = 0.3  (too low!)                 │
│                     ↓                                         │
│  3. Compute loss                                              │
│     Loss = -log(0.3) = 1.2  (high loss = bad prediction)     │
│                     ↓                                         │
│  4. Backward pass: compute gradients                          │
│     ∂Loss/∂weights → how to adjust each weight               │
│                     ↓                                         │
│  5. Update weights                                            │
│     weights = weights - lr × gradients                        │
│                     ↓                                         │
│  6. Repeat → Model now predicts P("4") = 0.9                 │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

#### Token Weights Visualization

```
Tokens:    [What] [is] [2+2] [?] [2+2] [equals] [4] [.]
Weights:   [ 0  ] [ 0] [ 0 ] [0] [ 1 ] [  1   ] [1] [1]
           ←──── Prompt ────→ ←──── Completion ────→
           (don't train)       (train on these)
```

#### When to Use SL

| Scenario | Why SL Works |
|----------|--------------|
| You have labeled data | SL directly learns from examples |
| Predictable outputs | Answers are deterministic |
| Format compliance | Teaches exact response structure |
| Knowledge injection | Adds specific domain knowledge |

#### References

- **Cross-Entropy Loss**: Standard in all neural network training
- **Language Model Fine-tuning**: [Radford et al., 2018 - GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- **Instruction Tuning**: [Wei et al., 2021 - FLAN](https://arxiv.org/abs/2109.01652)
- **LoRA**: [Hu et al., 2021 - LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

---

### Reinforcement Learning (RL) - Detailed Explanation

#### The Core Idea

RL trains models through **trial and error**. The model generates outputs, receives rewards for good ones, and learns to produce more high-reward outputs.

```
┌─────────────────────────────────────────────────────────┐
│              REINFORCEMENT LEARNING                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│   │  Prompt │ →  │  Model  │ →  │Response │            │
│   │"2 + 2 ="│    │(Policy) │    │  "4"    │            │
│   └─────────┘    └─────────┘    └─────────┘            │
│                       ↑               ↓                  │
│                       │         ┌─────────┐             │
│                       │         │ Reward  │             │
│                       │         │Function │             │
│                       │         └────┬────┘             │
│                       │              ↓                   │
│                  Update to      Reward = +1              │
│                  increase       (correct!)               │
│                  P(good outputs)                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Key Difference from SL

```
┌─────────────────────────┬─────────────────────────────┐
│   SUPERVISED LEARNING   │   REINFORCEMENT LEARNING    │
├─────────────────────────┼─────────────────────────────┤
│ You provide: answers    │ You provide: reward function│
│                         │                             │
│ "2+2=" → "4"           │ "2+2=" → reward(answer)     │
│                         │                             │
│ Model copies examples   │ Model discovers solutions   │
│                         │                             │
│ Limited by your data    │ Can exceed your data        │
└─────────────────────────┴─────────────────────────────┘
```

#### The Math: Policy Gradient

The goal is to maximize expected reward:

**LaTeX Formula - Objective:**
```latex
J(\theta) = \mathbb{E}_{x \sim \pi_\theta}[R(x)]
```

**LaTeX Formula - Gradient:**
```latex
\nabla_\theta J(\theta) = \mathbb{E}_{x \sim \pi_\theta}[R(x) \cdot \nabla_\theta \log \pi_\theta(x)]
```

**Rendered:**
$$
J(\theta) = \mathbb{E}_{x \sim \pi_\theta}[R(x)]
$$

$$
\nabla_\theta J(\theta) = \mathbb{E}_{x \sim \pi_\theta}[R(x) \cdot \nabla_\theta \log \pi_\theta(x)]
$$

**Where:**
- $J(\theta)$ = objective function (expected reward)
- $\pi_\theta$ = policy (the model)
- $R(x)$ = reward for output $x$
- $\nabla_\theta \log \pi_\theta(x)$ = gradient of log-probability

**In words:**
- If output got high reward → increase its probability
- If output got low reward → decrease its probability

#### Importance Sampling (Tinker's Default RL)

When the model that generated samples ($q$) differs from the current model ($p_\theta$):

**LaTeX Formula:**
```latex
\nabla_\theta J(\theta) = \mathbb{E}_{x \sim q}\left[R(x) \cdot \frac{\nabla_\theta \pi_\theta(x)}{q(x)}\right]
```

**Rendered:**
$$
\nabla_\theta J(\theta) = \mathbb{E}_{x \sim q}\left[R(x) \cdot \frac{\nabla_\theta \pi_\theta(x)}{q(x)}\right]
$$

**Where:**
- $q(x)$ = probability under the sampling policy (old model)
- $\pi_\theta(x)$ = probability under the current policy (new model)
- The ratio $\frac{\pi_\theta(x)}{q(x)}$ corrects for the distribution mismatch

**Why this matters**: During training, weights change, but we might still be learning from outputs generated by older weights. Importance sampling corrects for this bias.

#### RL Training Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      RL TRAINING LOOP                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Sample prompts from dataset                               │
│     ["Solve: 15 + 27", "What is 8 × 7?", ...]                │
│                     ↓                                         │
│  2. Generate rollouts (model responses)                       │
│     ┌──────────────────────────────────┐                     │
│     │ Prompt: "Solve: 15 + 27"         │                     │
│     │ Response: "15 + 27 = 42" ✓       │ → Reward = 1        │
│     │                                   │                     │
│     │ Prompt: "What is 8 × 7?"         │                     │
│     │ Response: "8 × 7 = 54" ✗         │ → Reward = 0        │
│     └──────────────────────────────────┘                     │
│                     ↓                                         │
│  3. Compute advantages                                        │
│     Advantage = Reward - Baseline                             │
│     (Baseline = average reward, for variance reduction)       │
│                     ↓                                         │
│  4. Policy gradient update                                    │
│     - Increase P(correct responses)                           │
│     - Decrease P(incorrect responses)                         │
│                     ↓                                         │
│  5. Repeat → Model discovers better strategies                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

#### Advantage Computation (GRPO Style)

Tinker uses group-relative advantages:

```
For each group of responses to the same prompt:

Advantage_i = Reward_i - mean(Rewards in group)

Example:
  Response 1: Reward = 1.0 → Advantage = 1.0 - 0.5 = +0.5
  Response 2: Reward = 0.0 → Advantage = 0.0 - 0.5 = -0.5

This means:
  - Increase probability of Response 1
  - Decrease probability of Response 2
```

#### PPO (Proximal Policy Optimization)

An alternative to importance sampling that clips updates to prevent too-large changes:

**LaTeX Formula:**
```latex
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
```

**Where the ratio is:**
```latex
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}
```

**Rendered:**
$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}
$$

**Where:**
- $r_t(\theta)$ = probability ratio between new and old policy
- $\hat{A}_t$ = advantage estimate at time $t$
- $\epsilon$ = clipping parameter (typically 0.1 or 0.2)
- $\text{clip}(r, 1-\epsilon, 1+\epsilon)$ = constrains $r$ to $[1-\epsilon, 1+\epsilon]$

This prevents the policy from changing too drastically in one update, improving training stability.

#### When to Use RL

| Scenario | Why RL Works |
|----------|--------------|
| Clear success metric | Can define reward function |
| Verifiable answers | Math, code, factual QA |
| Discovery needed | Model finds novel solutions |
| Beyond human demos | Can exceed training data quality |

#### References

- **Policy Gradient**: [Sutton et al., 1999 - Policy Gradient Methods](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- **PPO**: [Schulman et al., 2017 - Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **RLHF**: [Ouyang et al., 2022 - InstructGPT](https://arxiv.org/abs/2203.02155)
- **GRPO**: [Shao et al., 2024 - DeepSeekMath](https://arxiv.org/abs/2402.03300)
- **RLVR**: [Lambert et al., 2024 - RL with Verifiable Rewards](https://arxiv.org/abs/2411.15124)

---

### Direct Preference Optimization (DPO) - Detailed Explanation

#### The Core Idea

DPO trains models to prefer better responses over worse ones, without needing a separate reward model. It uses pairs of (chosen, rejected) responses.

```
┌─────────────────────────────────────────────────────────┐
│           DIRECT PREFERENCE OPTIMIZATION                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Training Data (preference pairs):                      │
│                                                          │
│   Prompt: "Explain quantum computing"                    │
│                                                          │
│   ┌─────────────────┐    ┌─────────────────┐            │
│   │ Chosen Response │    │Rejected Response│            │
│   │ "Quantum comp-  │    │ "It's just     │            │
│   │  uting uses..." │    │  magic..."     │            │
│   │     (Better)    │    │    (Worse)     │            │
│   └────────┬────────┘    └────────┬───────┘            │
│            │                      │                     │
│            └──────────┬───────────┘                     │
│                       ↓                                  │
│              Model learns to                             │
│              prefer chosen over                          │
│              rejected                                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### DPO vs RLHF

```
┌─────────────────────────────────────────────────────────┐
│                    TRADITIONAL RLHF                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Step 1: Train Reward Model                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │Preference│ →  │  Train   │ →  │  Reward  │          │
│  │   Data   │    │          │    │  Model   │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│                                        ↓                 │
│  Step 2: RL against Reward Model                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Policy  │ ←  │    RL    │ ←  │  Reward  │          │
│  │  Model   │    │ Training │    │  Scores  │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│                                                          │
│  Complexity: 2 models, RL instability                    │
│                                                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                         DPO                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Single Step: Direct Policy Optimization                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │Preference│ →  │   DPO    │ →  │ Aligned  │          │
│  │   Data   │    │ Training │    │  Policy  │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│                                                          │
│  Simplicity: 1 model, stable training                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### The Math: DPO Loss

DPO derives from the RLHF objective but eliminates the reward model:

**LaTeX Formula:**
```latex
\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]
```

**Rendered:**
$$
\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]
$$

**Where:**
- $\pi_\theta$ = current policy (model being trained)
- $\pi_{ref}$ = reference policy (initial model, frozen)
- $y_w$ = chosen (winning) response
- $y_l$ = rejected (losing) response
- $\beta$ = temperature parameter (default: 0.1)
- $\sigma$ = sigmoid function
- $\mathcal{D}$ = dataset of preference pairs

**In plain English**:
- Increase probability of chosen response relative to reference
- Decrease probability of rejected response relative to reference
- The reference model prevents the policy from deviating too much

#### Implicit Reward Model

DPO implicitly learns a reward model:

**LaTeX Formula:**
```latex
r(x, y) = \beta \log \frac{\pi_\theta(y | x)}{\pi_{ref}(y | x)}
```

**Rendered:**
$$
r(x, y) = \beta \log \frac{\pi_\theta(y | x)}{\pi_{ref}(y | x)}
$$

**This means:**
- High reward = model much prefers this response vs reference
- Low reward = model prefers this response less than reference
- The reward is defined by how much the policy has shifted from the reference

#### DPO Training Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      DPO TRAINING LOOP                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Load preference pair                                      │
│     ┌─────────────────────────────────────┐                  │
│     │ Prompt: "How do I learn Python?"    │                  │
│     │ Chosen: "Start with basics like..." │                  │
│     │ Rejected: "Just Google it."         │                  │
│     └─────────────────────────────────────┘                  │
│                     ↓                                         │
│  2. Compute log probabilities                                 │
│     π_θ(chosen), π_θ(rejected)                               │
│     π_ref(chosen), π_ref(rejected)                           │
│                     ↓                                         │
│  3. Compute implicit rewards                                  │
│     r_chosen = β × log(π_θ(chosen)/π_ref(chosen))            │
│     r_rejected = β × log(π_θ(rejected)/π_ref(rejected))      │
│                     ↓                                         │
│  4. Compute DPO loss                                          │
│     loss = -log σ(r_chosen - r_rejected)                     │
│                     ↓                                         │
│  5. Update to increase margin                                 │
│     Goal: r_chosen >> r_rejected                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

#### Key Parameters

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| `dpo_beta` | 0.1 | Higher = stronger preference enforcement |
| `learning_rate` | 1e-5 to 1e-6 | Lower than SL to prevent collapse |

**Warning**: Too high beta or LR can cause the model to collapse (output only very short or very long responses).

#### DPO Metrics

During training, you'll see:

```
┌────────────────────────────┬───────────┐
│ Metric                     │ Value     │
├────────────────────────────┼───────────┤
│ dpo_loss                   │ 0.683     │  ← Lower is better
│ accuracy                   │ 0.568     │  ← % choosing correct
│ margin                     │ 0.002     │  ← r_chosen - r_rejected
│ chosen_reward              │ 0.054     │  ← Implicit reward
│ rejected_reward            │ 0.032     │  ← Should be lower
└────────────────────────────┴───────────┘
```

**Ideal trends**:
- `accuracy` → increases (model prefers chosen)
- `margin` → increases (bigger gap)
- `dpo_loss` → decreases

#### When to Use DPO

| Scenario | Why DPO Works |
|----------|---------------|
| Preference data available | Need (chosen, rejected) pairs |
| Simpler than RLHF | No reward model to train |
| Stable training | No RL instability |
| Subjective quality | Helpfulness, style, tone |

#### References

- **DPO**: [Rafailov et al., 2023 - Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **RLHF Background**: [Christiano et al., 2017 - Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)
- **Bradley-Terry Model**: [Bradley & Terry, 1952 - Rank Analysis of Incomplete Block Designs](https://www.jstor.org/stable/2334029)

---

### RLHF Pipeline - Detailed Explanation

#### The Full Pipeline

RLHF (Reinforcement Learning from Human Feedback) is a three-stage process:

```
┌─────────────────────────────────────────────────────────────┐
│                    RLHF PIPELINE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: Supervised Fine-Tuning (SFT)                       │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    │
│  │ Base Model   │ →  │  SFT on     │ →  │  SFT Model   │    │
│  │ (Pretrained) │    │  Demos      │    │              │    │
│  └──────────────┘    └─────────────┘    └──────┬───────┘    │
│                                                 │            │
│  Stage 2: Reward Model Training                 ↓            │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    │
│  │  Preference  │ →  │  Train RM   │ →  │   Reward     │    │
│  │    Data      │    │             │    │   Model      │    │
│  └──────────────┘    └─────────────┘    └──────┬───────┘    │
│                                                 │            │
│  Stage 3: RL Fine-Tuning                        ↓            │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    │
│  │  SFT Model   │ →  │  RL with    │ →  │   RLHF       │    │
│  │              │    │  RM scores  │    │   Model      │    │
│  └──────────────┘    └─────────────┘    └──────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Stage 1: SFT Warm-Start

**Purpose**: Get the model "in distribution" - able to produce responses similar to what humans prefer.

```python
# SFT on demonstration data
training_client.forward_backward(demos, "cross_entropy")
training_client.optim_step(...)
```

**Why needed**: RL struggles if the initial policy is too far from good responses.

#### Stage 2: Reward Model

**Purpose**: Learn to score responses like a human would.

```
Training data:
  (prompt, response_A, response_B, A_is_better)

Reward model learns:
  Score(prompt, response) → float

Loss: Bradley-Terry
  P(A > B) = σ(r_A - r_B)
```

#### Stage 3: RL Optimization

**Purpose**: Optimize the policy to maximize reward model scores.

```
For each prompt:
  1. Generate response from policy
  2. Score with reward model
  3. Update policy to increase score

With KL penalty to stay close to SFT model:
  Objective = E[Reward(x,y)] - β × KL(π || π_SFT)
```

#### Why DPO is Often Preferred

```
┌──────────────────┬────────────────────┐
│      RLHF        │        DPO         │
├──────────────────┼────────────────────┤
│ 3 stages         │ 1 stage            │
│ 2 models to train│ 1 model to train   │
│ RL instability   │ Stable training    │
│ More flexible    │ Simpler            │
│ Can iterate      │ One-shot           │
└──────────────────┴────────────────────┘
```

#### When to Use Full RLHF

- Research on post-training pipelines
- Need iterative preference data collection
- Reward model reusable across tasks
- More control over the process

---

### Comparison: Choosing the Right Method

```
┌─────────────────────────────────────────────────────────────┐
│              WHICH METHOD SHOULD I USE?                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Do you have labeled input-output pairs?                     │
│  │                                                           │
│  ├─ YES → SUPERVISED LEARNING                                │
│  │        (Instruction tuning, chat, distillation)           │
│  │                                                           │
│  └─ NO ─┬─ Do you have a reward function?                    │
│         │                                                    │
│         ├─ YES → REINFORCEMENT LEARNING                      │
│         │        (Math, code, verifiable tasks)              │
│         │                                                    │
│         └─ NO ─┬─ Do you have preference pairs?              │
│                │                                             │
│                ├─ YES → DPO                                  │
│                │        (Alignment, quality, style)          │
│                │                                             │
│                └─ NO → Collect data first!                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Method Comparison Table

| Aspect | SL | RL | DPO |
|--------|----|----|-----|
| **Data needed** | Input-output pairs | Prompts + reward function | Preference pairs |
| **Training stability** | Very stable | Can be unstable | Stable |
| **Compute cost** | Low | High (sampling) | Medium |
| **Can exceed demos** | No | Yes | Limited |
| **Best for** | Format, knowledge | Reasoning, discovery | Alignment, quality |

#### Typical Improvement by Method

```
┌─────────────────────────────────────────────────────────────┐
│                   EXPECTED IMPROVEMENTS                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SUPERVISED LEARNING                                         │
│  ████████████████████░░░░░░░░░░  20-50%                      │
│  Good for: format compliance, domain knowledge               │
│                                                              │
│  REINFORCEMENT LEARNING                                      │
│  ██████████████████████████████████████░░  30-100%           │
│  Good for: math (2x), code (1.5x), reasoning                 │
│                                                              │
│  DPO                                                         │
│  ████████████░░░░░░░░░░░░░░░░░░  10-30%                      │
│  Good for: preference win rate, quality scores               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### Further Reading & References

#### Foundational Papers

| Topic | Paper | Link |
|-------|-------|------|
| LoRA | Hu et al., 2021 | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
| InstructGPT (RLHF) | Ouyang et al., 2022 | [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155) |
| DPO | Rafailov et al., 2023 | [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290) |
| PPO | Schulman et al., 2017 | [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347) |
| GRPO | Shao et al., 2024 | [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300) |

#### Tinker-Specific Resources

| Resource | Link |
|----------|------|
| Tinker Docs | [tinker-docs.thinkingmachines.ai](https://tinker-docs.thinkingmachines.ai) |
| LoRA Blog Post | [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) |
| Cookbook GitHub | [github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) |

#### Books & Courses

- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (free online)
- **Hugging Face Course**: [huggingface.co/learn](https://huggingface.co/learn)
- **DeepMind x UCL RL Course**: [YouTube playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)

---

## Real-World Experiments & Case Studies

This section showcases actual experiments and results from research groups using Tinker, demonstrating what's achievable with the platform.

---

### Case Study 1: Princeton Gödel Team - Mathematical Theorem Proving

**Goal**: Train a model to prove mathematical theorems automatically

**Method**: Reinforcement learning with verifiable rewards

**Results**:
```
┌─────────────────────────────────────────────────────────────┐
│           MATHEMATICAL THEOREM PROVING                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Benchmark: MiniF2F (Formal Mathematics)                     │
│                                                              │
│  ┌──────────────────┬───────────────┐                       │
│  │ Method           │ Accuracy      │                       │
│  ├──────────────────┼───────────────┤                       │
│  │ Base Model       │ ~40%          │                       │
│  │ After RL         │ 88.1%         │ ← pass@32             │
│  │ With Self-Corr.  │ 90.4%         │ ← beat larger models  │
│  └──────────────────┴───────────────┘                       │
│                                                              │
│  Key Achievement: Beat larger closed models with             │
│  only 20% of the typical training data                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**How it works**:

1. **Environment**: Formal proof checker (Lean/Coq)
2. **Reward**: +1 if proof verifies, 0 otherwise
3. **Training**: RL with importance sampling
4. **Innovation**: Self-correction loop improves proofs iteratively

**LaTeX - Reward Function:**
```latex
r(proof) = \begin{cases}
1 & \text{if } \text{verify}(proof) = \text{True} \\
0 & \text{otherwise}
\end{cases}
```

$$
r(proof) = \begin{cases}
1 & \text{if } \text{verify}(proof) = \text{True} \\
0 & \text{otherwise}
\end{cases}
$$

---

### Case Study 2: Stanford Rotskoff Lab - Chemistry Reasoning

**Goal**: Train a model to complete chemistry reasoning tasks

**Method**: Supervised learning + reinforcement learning

**Results**:
```
┌─────────────────────────────────────────────────────────────┐
│              CHEMISTRY REASONING                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Task: Chemical reaction prediction & reasoning              │
│                                                              │
│  Before Fine-tuning:  ██░░░░░░░░░░░░░░░░░░  15%              │
│  After Fine-tuning:   ██████████░░░░░░░░░░  50%              │
│                                                              │
│  Improvement: 3.3x accuracy gain                             │
│                                                              │
│  Quote: "A level of improvement previously out of reach      │
│  without Tinker's robust infrastructure support"             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**How it works**:

1. **Dataset**: Chemistry problems with verifiable answers
2. **Stage 1**: SFT on chemistry Q&A pairs
3. **Stage 2**: RL with correctness-based rewards
4. **Key insight**: Domain-specific fine-tuning dramatically improves specialized reasoning

---

### Case Study 3: Berkeley SkyRL - Multi-Agent Tool Use

**Goal**: Train agents for complex multi-turn tool use scenarios

**Method**: Async off-policy RL with multiple agents

**Setup**:
```
┌─────────────────────────────────────────────────────────────┐
│           MULTI-AGENT TOOL USE                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Architecture:                                               │
│                                                              │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│  │ Agent 1 │ ←→  │  Tools  │ ←→  │ Agent 2 │               │
│  │ (Query) │     │ (APIs)  │     │(Verify) │               │
│  └────┬────┘     └────┬────┘     └────┬────┘               │
│       │               │               │                     │
│       └───────────────┼───────────────┘                     │
│                       ↓                                      │
│              ┌─────────────────┐                             │
│              │  Reward Signal  │                             │
│              │ (Task Success)  │                             │
│              └─────────────────┘                             │
│                                                              │
│  Training: Async off-policy RL                               │
│  - Multiple model versions generate data simultaneously      │
│  - Higher throughput than on-policy                          │
│  - "Off-by-K" approach for bounded staleness                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**How it works**:

1. **Multi-agent**: Agents collaborate or compete
2. **Tool use**: Agents call APIs (search, calculator, etc.)
3. **Multi-turn**: Extended conversations with multiple tool calls
4. **Async training**: Higher throughput with managed staleness

---

### Case Study 4: Redwood Research - AI Control Tasks

**Goal**: RL training on difficult AI control/safety tasks

**Method**: RL on Qwen3-32B (large model)

**Key Points**:
- Demonstrates Tinker can handle large 32B parameter models
- Focus on AI safety and control scenarios
- Uses custom reward functions for alignment

---

### Case Study 5: Math Reasoning (AIME Competition)

**Goal**: Achieve high performance on competition math problems

**Results**:
```
┌─────────────────────────────────────────────────────────────┐
│              MATH COMPETITION RESULTS                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Benchmark: AIME 2024 (American Invitational Math Exam)      │
│                                                              │
│  Accuracy achieved: 70%                                      │
│                                                              │
│  Compute comparison:                                         │
│  ┌────────────────────────┬──────────────┐                  │
│  │ Traditional approach   │ 17,920 GPU-h │                  │
│  │ With Tinker            │  1,800 GPU-h │                  │
│  └────────────────────────┴──────────────┘                  │
│                                                              │
│  Efficiency gain: 10x less compute                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### LoRA vs Full Fine-Tuning: Experimental Results

Thinking Machines Lab conducted extensive experiments comparing LoRA to full fine-tuning:

#### Key Findings

**1. Performance Parity**
```
┌─────────────────────────────────────────────────────────────┐
│         LORA VS FULL FINE-TUNING                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Supervised Learning:                                        │
│  - LoRA matches full fine-tuning when properly configured    │
│  - Both show "similar learning curves with loss decreasing   │
│    linearly with the logarithm of steps"                     │
│                                                              │
│  Reinforcement Learning:                                     │
│  - LoRA "fully matches" full fine-tuning                     │
│  - Works even with ranks as low as 1 for policy gradient     │
│  - Identical peak performance on MATH and GSM8K              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**2. Learning Rate Discovery**

The optimal learning rate for LoRA is consistently **10x higher** than full fine-tuning:

**LaTeX:**
```latex
\text{LR}_{LoRA} \approx 10 \times \text{LR}_{FullFT}
```

$$
\text{LR}_{LoRA} \approx 10 \times \text{LR}_{FullFT}
$$

**Tinker's recommendation**: Use `hyperparam_utils.get_lr(model_name)` to get the correct learning rate.

**3. Layer Application**

| Configuration | Performance |
|--------------|-------------|
| Attention-only LoRA | Significantly underperforms |
| MLP-only LoRA | Better than attention-only |
| All layers (default) | Best results |

**Key insight**: Always apply LoRA to MLPs/MoE layers, not just attention.

**4. Compute Efficiency**

```
Compute per pass:
- Full fine-tuning: 100%
- LoRA:             ~67% (saves ⅓ of FLOPs)
```

---

### On-Policy Distillation Research

Thinking Machines Lab's research on distillation methods:

**Comparison**: On-policy vs off-policy distillation

```
┌─────────────────────────────────────────────────────────────┐
│         DISTILLATION METHODS                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Off-Policy Distillation (Traditional):                      │
│  - Train on pre-generated teacher samples                    │
│  - Student never generates its own data                      │
│                                                              │
│  On-Policy Distillation (Recommended):                       │
│  - Student generates, teacher corrects                       │
│  - Iterative improvement                                     │
│                                                              │
│  Results:                                                    │
│  - On-policy matches RL performance                          │
│  - Uses only 10% of compute (9-30x more efficient)           │
│  - Works across reasoning and assistant tasks                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### Community Research Directions

Thinking Machines Lab has proposed several open research questions:

#### 1. Constitutional AI from Base Models

**Question**: Does RLAIF work better from instruction-tuned or base models?

**Experiment design**:
```python
# Compare two approaches:
# Approach A: Base model → RLAIF
# Approach B: Instruct model → RLAIF

# Hypothesis: Base model may learn constitution better
# without prior instruction-tuning biases
```

#### 2. Noisy Student with RLVR

**Question**: Can self-distillation improve with RL iterations?

**LaTeX - Noisy Student Objective:**
```latex
\mathcal{L} = \mathcal{L}_{labeled} + \lambda \cdot \mathcal{L}_{pseudo}
```

$$
\mathcal{L} = \mathcal{L}_{labeled} + \lambda \cdot \mathcal{L}_{pseudo}
$$

Where pseudo-labels come from student after each RL iteration.

#### 3. Direct RL on Pairwise Judge

**Question**: Can we skip the reward model in RLHF?

```
Traditional RLHF:
  Preferences → Reward Model → RL

Direct approach:
  Preferences → Pairwise Judge (prompted) → RL

Potential benefits:
- Simpler pipeline
- No reward model to train
- More flexible
```

#### 4. RL Memory Test

**Question**: How fast can RL memorize random sequences?

**Purpose**: Validate theoretical bounds on information acquisition

```python
# Environment: Memorize random number sequence
sequence = [7, 3, 9, 1, 4, 8, 2, 6, 5, 0]

# Reward: +1 for each correct digit
# Compare: SFT vs RL learning rates
```

#### 5. GAN for Humor

**Question**: Can adversarial training improve joke generation?

```
┌──────────────┐      ┌──────────────┐
│   Generator  │ ←──→ │ Discriminator │
│ (Joke maker) │      │ (Joke judge)  │
└──────────────┘      └──────────────┘
      ↓                      ↓
 "Why did the..."      "Funny: 0.7"
```

---

### Expert Testimonials

**Xi Ye, Princeton University**:
> "RLing >10B models on a typical academic setup… is a hassle, but with Tinker I can focus more on the data/algorithms."

**Andrej Karpathy, Former OpenAI/Tesla AI Head**:
> "[Tinker] lets users retain ~90% of algorithmic control while removing ~90% of infrastructure pain."

---

### Summary: What These Experiments Show

| Domain | Improvement | Method |
|--------|-------------|--------|
| Math (theorem proving) | 40% → 90% | RL |
| Chemistry reasoning | 15% → 50% | SL + RL |
| Competition math | 70% accuracy, 10x less compute | RL |
| General reasoning | Matches RL with 10% compute | On-policy distillation |

**Key Takeaways**:

1. **RL works**: Massive improvements on reasoning tasks (2-3x)
2. **LoRA is enough**: Matches full fine-tuning at 67% compute
3. **Learning rate matters**: LoRA needs 10x higher LR
4. **Efficiency is possible**: 10-30x compute savings with right methods
5. **Domain-specific wins**: Chemistry went from 15% to 50% with fine-tuning

---

## Available Recipes (What You Can Build)

The Tinker Cookbook includes ready-to-use recipes:

### Math Reasoning
**Path**: `tinker_cookbook/recipes/math_rl/`

Train models to solve math problems with chain-of-thought reasoning.
- Dataset: GSM8K, MATH
- Expected: 30% → 70%+ accuracy

### Chat/Instruction Tuning
**Path**: `tinker_cookbook/recipes/chat_sl/`

Fine-tune on conversational datasets like Tulu3.
- Improves instruction following
- Better response quality

### Code Generation
**Path**: `tinker_cookbook/recipes/code_rl/`

Train models to write and debug code.
- Reward: Passing unit tests
- Improves HumanEval/MBPP scores

### Tool Use
**Path**: `tinker_cookbook/recipes/tool_use/search/`

Train models to use retrieval tools and APIs.
- Better tool selection
- More accurate tool usage

### Multi-Agent
**Path**: `tinker_cookbook/recipes/multiplayer_rl/`

Train models in competitive/cooperative scenarios:
- `twenty_questions/`: Question-asking game
- `guess_number/`: Number guessing
- `text_arena/`: Text-based competitions

### Prompt Distillation
**Path**: `tinker_cookbook/recipes/prompt_distillation/`

Compress long system prompts into model behavior.
- Reduce inference costs
- Faster responses

### Preference Learning
**Path**: `tinker_cookbook/recipes/preference/`

DPO and RLHF implementations:
- `dpo/`: Direct Preference Optimization
- `rlhf/`: Full RLHF pipeline
- `shorter/`: Preference for concise responses

### Distillation
**Path**: `tinker_cookbook/recipes/distillation/`

Distill knowledge from larger to smaller models.

---

## Pricing

### Pricing Model

Tinker uses **usage-based pricing** with three operation types:
- **Prefill**: Processing input tokens
- **Sample**: Generating output tokens
- **Train**: Training/fine-tuning operations

### Cost by Model Size

| Model | Prefill | Train |
|-------|---------|-------|
| Llama-3.2-1B | $0.03 | $0.09 |
| Llama-3.2-3B | - | - |
| Qwen3-4B | - | - |
| Llama-3.1-8B | - | - |
| ... | ... | ... |
| DeepSeek-V3.1 | $1.13 | $3.38 |

### Estimated Training Costs

For a 1B-3B model:
- **Small experiment** (100 steps): $9-27
- **Full training** (1,000 steps): $90-270
- **Production** (10,000 steps): $900-2,700

### Cost Optimization Tips

1. **Use MoE models**: 30B quality at 3B cost
2. **Start small**: Prototype with Llama-3.2-1B
3. **Pipeline batches**: Submit next batch before current completes
4. **Use async training**: Maximize GPU utilization

### Current Status

- **Private beta**: Free during beta period
- **Future**: Usage-based pricing

---

## Getting Started

### Prerequisites

1. Sign up at [waitlist](https://thinkingmachines.ai/tinker)
2. Get API key from [console](https://tinker-console.thinkingmachines.ai)
3. Python >= 3.11

### Installation

```bash
# Install Tinker SDK
pip install tinker

# Set API key
export TINKER_API_KEY=your_key_here

# Clone and install cookbook
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
pip install -e .
```

### Quick Start Commands

**1. Basic Supervised Learning**:
```bash
python -m tinker_cookbook.recipes.sl_basic \
    model_name=meta-llama/Llama-3.2-1B \
    log_path=/tmp/sl_test
```

**2. Reinforcement Learning**:
```bash
python -m tinker_cookbook.recipes.rl_basic \
    model_name=meta-llama/Llama-3.1-8B \
    log_path=/tmp/rl_test
```

**3. DPO Training**:
```bash
python -m tinker_cookbook.preference.train_dpo_cli \
    model_name=meta-llama/Llama-3.2-1B \
    dataset=hhh \
    log_path=/tmp/dpo_test
```

**4. Run Evaluations**:
```bash
python -m tinker_cookbook.eval.run_inspect_evals \
    model_path=tinker://YOUR_MODEL \
    model_name=meta-llama/Llama-3.2-1B \
    tasks=ifeval
```

### Verify Installation

```python
import tinker

service_client = tinker.ServiceClient()
print("Tinker connected successfully!")

# Create a simple training client
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.2-1B",
    rank=32
)
print(f"Training client created: {training_client}")
```

---

## Key Concepts

### Datums

A `Datum` is a single training example containing:
- `model_input`: Tokenized input sequence
- `loss_fn_inputs`: Target tokens and weights

### Renderers

Convert between chat messages and token sequences:
- `llama3`: For Llama models
- `qwen3`: For Qwen models
- `role_colon`: Generic format

### Completers

Abstractions for sampling:
- `TokenCompleter`: Token-level (for RL)
- `MessageCompleter`: Message-level (for evaluation)

### Environments

RL environments that provide:
- Initial observations (prompts)
- Reward functions
- Stop conditions

---

## Expected Improvements

| Training Method | Typical Improvement |
|-----------------|---------------------|
| Supervised Learning | 20-50% accuracy gain |
| Reinforcement Learning | 30-100% on reasoning |
| DPO | 10-30% preference win rate |

### Benchmark Examples

- **Math (GSM8K)**: 30% → 70%+ accuracy
- **Code (HumanEval)**: 20-40% better pass@1
- **Instruction (IFEval)**: 50-100% format compliance

---

## Resources

### Documentation
- [Tinker Docs](https://tinker-docs.thinkingmachines.ai/)
- [API Reference](https://tinker-docs.thinkingmachines.ai/training-sampling)

### Code
- [Tinker SDK](https://github.com/thinking-machines-lab/tinker)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)

### Support
- Email: tinker@thinkingmachines.ai
- Issues: https://github.com/thinking-machines-lab/tinker-cookbook/issues

---

## Summary

**Tinker is**:
- A hosted API for LLM fine-tuning
- Simple Python interface, they handle GPUs
- Supports SL, RL, DPO, and custom training
- Works with Llama, Qwen, DeepSeek models
- Usage-based pricing (free during beta)

**Best for**:
- Researchers exploring post-training methods
- Developers building specialized models
- Teams without GPU infrastructure
- Anyone wanting full control over training

**Start with**:
1. Get API key from console
2. Install `pip install tinker`
3. Run a basic recipe
4. Iterate on your use case
