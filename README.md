# BlicketTest_CausalReasoning

Multi-turn causal reasoning environment based on the Blicket detector paradigm from developmental psychology. Built with Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

## Setup

```bash
prime env install BlicketTest_CausalReasoning
```

## Usage

```bash
# Install locally
prime env install BlicketTest_CausalReasoning

# Run evaluation
prime eval run BlicketTest_CausalReasoning

# Push to Prime Hub
prime env push -p ./environments/BlicketTest_CausalReasoning
```

## Environment

| Environment | Description |
| ----------- | ----------- |
| [BlicketTest_CausalReasoning](environments/BlicketTest_CausalReasoning/) | Multi-turn environment where an agent must identify which objects are "Blickets" by designing experiments with a Blicket-detecting machine. Tests causal reasoning, hypothesis elimination, and experimental design. Inspired by [Do LLMs Think Like Scientists? Causal Reasoning and Hypothesis Testing in LLMs](https://arxiv.org/pdf/2505.09614) |
