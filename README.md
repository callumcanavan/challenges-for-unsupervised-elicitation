# Challenges for Unsupervised Elicitation

Code for *Three Concrete Challenges and Two Hopes for the Safety of Unsupervised Elicitation*[link]—specifically for running unsupervised elicitation (UE) and easy-to-hard generalization (E2H) methods on datasets with imbalanced training sets and impossible tasks, including ensembling and combining UE and E2H techniques described in the paper.

## Setup

```bash
# Clone and install
git clone https://github.com/callumcanavan/challenges-for-unsupervised-elicitation.git
cd challenges-for-unsupervised-elicitation
pip install uv
uv sync

# Download datasets from HuggingFace
python scripts/download_datasets.py
```

`scripts/download_datasets.py` downloads from [callum-canavan/challenges-for-unsupervised-elicitation](https://huggingface.co/datasets/callum-canavan/challenges-for-unsupervised-elicitation) on HuggingFace and converts them to the format expected by the code.

For prompting experiments, also install vLLM:
```bash
uv sync --extra vllm
```

## Datasets

Hosted on HuggingFace at [callum-canavan/challenges-for-unsupervised-elicitation](https://huggingface.co/datasets/callum-canavan/challenges-for-unsupervised-elicitation). See the paper for full details on dataset derivations.

| Dataset | Challenge | Description |
|---------|-----------|-------------|
| **GSM8K** | Imbalanced training sets | Derived from [GSM8K](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021) using candidate LLM solutions from [Wen et al. (2025)](https://arxiv.org/abs/2410.09700). Each question is paired with one correct and one incorrect solution, with varying train set class balance. |
| **Ctrl-Z** | Imbalanced training sets | Sequences of bash commands labeled safe/malicious, derived from [Bhatt et al. (2025)](https://arxiv.org/abs/2412.09084), with varying train set class balance. |
| **GSM-Political** | Impossible tasks | Mix of GSM8K problems and normative political statements from [Political Ideologies](https://huggingface.co/datasets/JyotiNayak/political_ideologies) (Nayak et al.) — methods should be confident on math but uncertain on normative statements. |
| **Larger Than** | (Easy dataset for E2H) | Simple number comparison statements from [Marks & Tegmark (2023)](https://arxiv.org/abs/2310.06824) with added question fields. Used as the easy task for E2H probes. |

## Running experiments

### 1. Extract activations

Extract contrastive hidden states from Llama 3.1 for use in probing experiments:

```bash
python scripts/get_activations.py <dataset> <model> [--layer LAYER]
```

For example:
```bash
python scripts/get_activations.py gsm8k_preference meta-llama/Llama-3.1-8B --layer 18
python scripts/get_activations.py ctrl_z_10steps meta-llama/Llama-3.1-8B --layer 18
python scripts/get_activations.py political_normative meta-llama/Llama-3.1-70B --layer 36
```

### 2. Probing experiments

Run probing methods (CCS, PCA, supervised, ensembles, etc.) on extracted activations:

```bash
python scripts/run_probing.py --dataset <dataset> --method <method> --seed <seed> [options]
```

For imbalanced dataset experiments, vary the training set class balance:
```bash
python scripts/run_probing.py --dataset gsm8k_preference --method ccs --train-true-proportion 0.5 --seed 0
python scripts/run_probing.py --dataset ctrl_z_10steps --method ccs --train-true-proportion 0.01 --seed 0
```

For impossible task experiments:
```bash
python scripts/run_probing.py --dataset gsm_political --method supervised --seed 0
```

### 3. Prompting experiments

Run prompting methods (zero-shot, few-shot, bootstrap) using vLLM:

```bash
python scripts/run_prompting.py --dataset <dataset> --method <method> --seed <seed> [options]
```

For example:
```bash
python scripts/run_prompting.py --dataset gsm8k_preference --method bootstrap --train-true-proportion 0.5 --seed 0
```

### 4. Sweep experiments

To run sweeps of parameters:

```bash
python experiments/probing_sweep.py [options]
python experiments/prompting_sweep.py [options]
```

## Methods

**Prompting**: zero-shot, random few-shot, bootstrapped few-shot, golden few-shot (ceiling)

**Probing**: random probe, CCS, PCA, E2H, supervised (ceiling), UE+E2H

**Ensembles**: consensus-weighted random/PCA ensembles, E2H-weighted random/PCA ensembles

## Tests

```bash
uv sync --extra dev
pytest
```
