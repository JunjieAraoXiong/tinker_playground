# Research Proposal: Cross-Domain Transfer in Fine-Tuned Language Models

A research project using Tinker to study how reasoning capabilities transfer across domains.

---

## Executive Summary

**Research Question**: When we fine-tune a language model on Domain A, how much does it improve on Domain B?

**Why It Matters**:
- Reduce training costs by leveraging transfer
- Enable learning in data-scarce domains
- Understand what models actually learn during fine-tuning
- Guide curriculum design for training pipelines

**Approach**: Systematic experiments measuring transfer between domain pairs using Tinker's training infrastructure.

---

## Background & Motivation

### The Transfer Learning Question

```
┌─────────────────────────────────────────────────────────────┐
│         THE FUNDAMENTAL QUESTION                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  When a model learns to reason in one domain,                │
│  what transfers to other domains?                            │
│                                                              │
│  Possibilities:                                              │
│  • General reasoning skills transfer                         │
│  • Only domain-specific knowledge transfers                  │
│  • Nothing transfers (or negative transfer)                  │
│  • Transfer depends on domain similarity                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why This Matters for the Community

Based on Tinker Discord community use cases:

| Community Member | Their Domain | Transfer Question |
|-----------------|--------------|-------------------|
| Nelly | Biosignals/Neuroscience | Does scientific reasoning transfer? |
| Enda | Medical/Regulatory | Can legal reasoning help compliance? |
| Yifu | Finance/Trading | Does math reasoning help trading? |
| Nolan | Code modernization | Does code skill transfer across languages? |

**If transfer works**: These builders can leverage training from related domains.
**If transfer fails**: They need domain-specific data for everything.

---

## Research Design

### Phase 1: Domain Selection

#### Primary Domain Pairs to Study

| Source Domain | Target Domain | Hypothesis |
|--------------|---------------|------------|
| **Math** | **Code** | Logical reasoning transfers |
| **Math** | **Physics** | Quantitative reasoning transfers |
| **Code** | **Math** | Structured thinking transfers |
| **Chemistry** | **Biology** | Scientific reasoning transfers |
| **Logic** | **Law** | Argument structure transfers |

#### Available Datasets

| Domain | Datasets | Task Type |
|--------|----------|-----------|
| Math | GSM8K, MATH, AQuA | Problem solving |
| Code | HumanEval, MBPP | Generation + verification |
| Physics | SciQ, ARC-Challenge | QA + reasoning |
| Chemistry | ChemQA, MoleculeNet | Scientific reasoning |
| Biology | BioASQ, PubMedQA | Scientific QA |
| Logic | LogiQA, ReClor | Logical reasoning |
| Law | LegalBench, CaseHOLD | Legal reasoning |

### Phase 2: Experimental Design

#### Control Conditions

For each domain pair (A → B):

```
┌─────────────────────────────────────────────────────────────┐
│         EXPERIMENTAL CONDITIONS                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. BASELINE                                                 │
│     Base model → Test on B                                   │
│     (No training, establishes floor)                         │
│                                                              │
│  2. TRANSFER                                                 │
│     Train on A → Test on B                                   │
│     (The transfer condition)                                 │
│                                                              │
│  3. DIRECT                                                   │
│     Train on B → Test on B                                   │
│     (Upper bound, best we can do)                            │
│                                                              │
│  4. SEQUENTIAL                                               │
│     Train on A → Train on B → Test on B                      │
│     (Does A help as pre-training?)                           │
│                                                              │
│  5. JOINT                                                    │
│     Train on A+B → Test on B                                 │
│     (Multi-task learning)                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Metrics

**Primary Metrics**:
- **Transfer Gain** = Performance(Transfer) - Performance(Baseline)
- **Transfer Efficiency** = Performance(Transfer) / Performance(Direct)
- **Relative Transfer** = Transfer Gain / Direct Gain

**Secondary Metrics**:
- Compute efficiency (performance per training dollar)
- Generalization to held-out test sets
- Robustness to prompt variations

### Phase 3: Training Protocol

#### Model Selection

Start with smaller models for cost efficiency:

| Model | Size | Cost/1K steps | Use Case |
|-------|------|---------------|----------|
| Llama-3.2-1B | 1B | ~$90 | Initial experiments |
| Llama-3.2-3B | 3B | ~$150 | Validation |
| Qwen3-30B-A3B | 30B (3B active) | ~$150 | Final experiments |

#### Training Configuration

```python
# Standard configuration for all experiments
config = {
    "base_model": "meta-llama/Llama-3.2-1B",
    "lora_rank": 32,
    "learning_rate": 1e-4,  # Use hyperparam_utils.get_lr()
    "num_steps": 1000,
    "batch_size": 32,
    "eval_every": 100,
}

# Training methods to compare
methods = ["SL", "RL", "SL+RL"]
```

#### Implementation Code Structure

```python
import tinker
from tinker_cookbook import hyperparam_utils

async def run_transfer_experiment(
    source_domain: str,
    target_domain: str,
    model_name: str = "meta-llama/Llama-3.2-1B"
):
    """Run complete transfer experiment for one domain pair."""

    service_client = tinker.ServiceClient()
    results = {}

    # ========================================
    # Condition 1: Baseline
    # ========================================
    base_client = service_client.create_lora_training_client(
        base_model=model_name
    )
    base_sampler = base_client.save_weights_and_get_sampling_client(
        name="baseline"
    )
    results["baseline"] = await evaluate(base_sampler, target_domain)

    # ========================================
    # Condition 2: Transfer (A → B)
    # ========================================
    transfer_client = service_client.create_lora_training_client(
        base_model=model_name
    )

    # Train on source domain
    await train_on_domain(transfer_client, source_domain)

    transfer_sampler = transfer_client.save_weights_and_get_sampling_client(
        name=f"trained_on_{source_domain}"
    )
    results["transfer"] = await evaluate(transfer_sampler, target_domain)

    # ========================================
    # Condition 3: Direct (B → B)
    # ========================================
    direct_client = service_client.create_lora_training_client(
        base_model=model_name
    )

    # Train on target domain
    await train_on_domain(direct_client, target_domain)

    direct_sampler = direct_client.save_weights_and_get_sampling_client(
        name=f"trained_on_{target_domain}"
    )
    results["direct"] = await evaluate(direct_sampler, target_domain)

    # ========================================
    # Compute transfer metrics
    # ========================================
    results["transfer_gain"] = results["transfer"] - results["baseline"]
    results["transfer_efficiency"] = results["transfer"] / results["direct"]

    return results
```

---

## Hypotheses

### H1: Reasoning Skills Transfer

**Prediction**: Math training improves code performance more than random baseline.

**Mechanism**: Both require:
- Step-by-step logical reasoning
- Variable tracking
- Error checking

### H2: Transfer Depends on Similarity

**Prediction**: Transfer correlates with domain similarity.

**Test**: Measure transfer across multiple pairs, correlate with:
- Vocabulary overlap
- Reasoning type similarity
- Problem structure similarity

### H3: RL Transfers Better Than SL

**Prediction**: RL-trained models show more transfer than SL-trained.

**Mechanism**: RL learns generalizable reasoning strategies; SL memorizes patterns.

### H4: Sequential Training Helps

**Prediction**: A → B → Test(B) outperforms just B → Test(B).

**Mechanism**: Source domain provides useful initialization.

---

## Expected Results

### Transfer Matrix (Hypothetical)

```
                     Target Domain
                 Math   Code   Physics   Logic
Source    Math    -     70%     80%      60%
Domain    Code   50%     -      40%      55%
         Physics 75%    35%      -       50%
          Logic  45%    50%     45%       -

Values = Transfer Efficiency (% of direct training performance)
```

### Key Findings to Look For

1. **Asymmetric transfer**: Does Math → Code ≠ Code → Math?
2. **Transfer ceiling**: Is there a maximum transfer possible?
3. **Negative transfer**: Do any pairs hurt each other?
4. **Method effects**: Does RL transfer better than SL?

---

## Timeline & Budget

### Phase 1: Pilot Study (Week 1-2)
- Single domain pair (Math → Code)
- 1B model only
- Establish methodology

**Cost**: ~$300

### Phase 2: Main Experiments (Week 3-5)
- All 5 primary domain pairs
- Both 1B and 3B models
- SL and RL methods

**Cost**: ~$1,500

### Phase 3: Analysis & Ablations (Week 6-7)
- Effect of training data size
- Effect of model size
- Detailed error analysis

**Cost**: ~$500

### Phase 4: Write-up (Week 8)
- Paper/report preparation
- Visualizations
- Code release

**Cost**: ~$0

### Total Budget: ~$2,300

---

## Deliverables

### 1. Transfer Matrix
Complete transfer efficiency measurements for all domain pairs.

### 2. Analysis Report
- Which transfers work and why
- Practical recommendations for practitioners
- Theoretical implications

### 3. Open Source Code
- Experiment scripts
- Evaluation code
- Reproduction instructions

### 4. Paper/Publication
Potential venues:
- NeurIPS / ICML / ICLR (if novel findings)
- EMNLP / ACL (NLP focus)
- arXiv preprint (rapid dissemination)

---

## Broader Impact

### For Researchers
- Understanding of what fine-tuning actually learns
- Guidelines for training curriculum design
- Baseline for future transfer studies

### For Practitioners
- Know which domains can share training
- Reduce data requirements for rare domains
- Cost-effective training strategies

### For the Tinker Community
- Directly applicable to community use cases:
  - Nelly: Can other scientific training help biosignals?
  - Enda: Can legal training help regulatory?
  - Yifu: Can math training help trading models?

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| API access delays | Start with pilot; scale up when access confirmed |
| Negative results | Negative transfer is still publishable/useful |
| Cost overruns | Start small; checkpoint frequently |
| Reproducibility | Document everything; use fixed seeds |

---

## Next Steps

1. **Get Tinker access** (contact about API key issue)
2. **Set up evaluation benchmarks** for each domain
3. **Run pilot study** on Math → Code
4. **Iterate on methodology** based on pilot
5. **Scale to full experiment matrix**

---

## Contact

For questions about this research proposal:
- Your name/email
- Research institution
- Advisor (if applicable)

For Tinker access issues:
- tinker@thinkingmachines.ai
