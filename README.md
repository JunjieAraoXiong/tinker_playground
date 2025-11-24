# CS61A LLM Tutor

Fine-tuning LLMs to solve and explain UC Berkeley CS61A problems using reinforcement learning with test verification.

**Status: Just getting started**

## Goal

Build an LLM that can:
- Solve CS61A programming problems (Python, Scheme)
- Explain solutions step-by-step
- Provide hints without giving away full answers

## Approach

Using the [Tinker API](https://tinker.thinkingmachines.ai/) for fine-tuning with:
- **RL with test verification**: Reward signal from code passing unit tests
- **SFT on explanations**: Train on problem + solution + explanation pairs

## Tech Stack

- Python ≥3.11
- Tinker SDK
- PyTorch
- HuggingFace Transformers

## Documentation

See [TINKER_OVERVIEW.md](TINKER_OVERVIEW.md) for Tinker API documentation.

## License

Apache 2.0
