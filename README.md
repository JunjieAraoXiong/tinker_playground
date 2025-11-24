# Tinker Cookbook

A comprehensive toolkit for fine-tuning Large Language Models using the [Tinker API](https://tinker.thinkingmachines.ai/). Write simple Python code on your CPU machine while Tinker handles GPU infrastructure, distributed training, and scaling.

## Overview

This cookbook provides practical implementations for post-training LLMs with multiple methods:

- **Supervised Fine-Tuning (SFT)**: Train on input-output pairs for instruction tuning, chat, and domain adaptation
- **Reinforcement Learning (RL)**: Trial-and-error training with reward signals for math, code, and reasoning tasks
- **Direct Preference Optimization (DPO)**: Learn from preference pairs without separate reward models
- **RLHF Pipeline**: Full preference learning (SFT → Reward Model → RL)
- **Custom Loss Functions**: Define arbitrary differentiable objectives

All methods support **LoRA fine-tuning** for 10-100x fewer trainable parameters with comparable performance.

## Quick Start

```bash
# Install
pip install -e .

# Set API key
export TINKER_API_KEY=your_key

# Run supervised fine-tuning
python -m tinker_cookbook.recipes.sl_basic

# Run RL training
python -m tinker_cookbook.recipes.rl_basic
```

## Recipes

Ready-to-use training recipes for common tasks:

| Recipe | Description |
|--------|-------------|
| `math_rl/` | Solve math problems (GSM8K, MATH) with chain-of-thought |
| `code_rl/` | Code generation with test verification |
| `chat_sl/` | Conversation quality improvement |
| `tool_use/` | Train models to use APIs and search |
| `multiplayer_rl/` | Multi-agent scenarios (games, debates) |
| `distillation/` | Knowledge transfer from large to small models |
| `preference/` | DPO and RLHF implementations |

## Supported Models

- **Llama-3.x**: 1B, 3B, 8B, 70B
- **Qwen3**: 4B-235B (including MoE variants)
- **DeepSeek-V3.1**

## Tech Stack

- Python ≥3.11
- PyTorch
- HuggingFace Transformers
- Tinker SDK

## Documentation

See [TINKER_OVERVIEW.md](TINKER_OVERVIEW.md) for comprehensive technical documentation including:
- Detailed explanations of each training method
- Hyperparameter recommendations per model
- Architecture deep dives
- Mathematical foundations

## License

Apache 2.0
