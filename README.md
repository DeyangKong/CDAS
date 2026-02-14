# CDAS: Competence-Difficulty Alignment Sampling for RL Training

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-AAAI%202026-blue)](https://arxiv.org/abs/2505.17652)
</div>、

## 🌟 Overview

**CDAS** (Competence-Difficulty Alignment Sampling) is a novel sampling strategy for Reinforcement Learning training that addresses the low sampling efficiency challenge in LLM reasoning tasks. By dynamically aligning problem difficulty with model competence, CDAS achieves superior performance while significantly reducing training overhead.

### Key Features

- ✨ **Stable Difficulty Estimation**: Aggregates historical performance to provide robust problem difficulty assessment
- 🎯 **Dynamic Alignment**: Adaptively selects problems that match the model's current competence level
- 🚀 **High Efficiency**: Reduces training time overhead by **57.06%** compared to Dynamic Sampling
- 📊 **Strong Performance**: Achieves **45.89%** average accuracy across 6 mathematical benchmarks
- 🔧 **Easy Integration**: Works seamlessly with popular RL algorithms (GRPO, PPO)

## 🔧 Installation


### Setup

```bash
# Clone the repository
git clone https://github.com/DeyangKong/CDAS.git
cd CDAS

# Create conda environment
conda create -n cdas python=3.8
conda activate cdas

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```bash
# Set environment variables
export DATA_PATH="./data"
export MODEL_PATH="./models"
export CHECKPOINT_PATH="./checkpoints"
export LOG_PATH="./logs"

# Run training with default configuration
bash ./scripts/train_grpo_cdas.sh
```

> **Note**: The script uses predefined default values.

### Custom Configuration

**Command-line arguments will override script defaults.** Only specify parameters you want to change:

```bash
# Example: Adjust rollout and temperature settings
bash ./scripts/train_grpo_cdas.sh \
    --rollout_n 16 \
    --temperature 0.85
```



## 🎯 Training Configuration

### Dataset Format

CDAS requires training data in **Parquet format** with a specific structure. The dataset should contain prompts and associated metadata for reinforcement learning training.


#### Basic Structure

```python
{
    "prompt": [
        {
            "role": "user",
            "content": "Solve the following math problem: ..."
        }
    ],
    "extra_info": {
        "difficulty": "medium",  # Optional: problem difficulty label
        "category": "algebra"  # Optional: problem category
    }
}
```

### CDAS Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cdas.enable` | Enable/disable CDAS | `true` |
| `cdas.warmup_steps` | Warmup period before adaptive sampling | `1` |
| `cdas.skip_warmup` | Skip warmup and start immediately | `false` |
| `cdas.extra_samples` | Extra samples for batch size guarantee | `0` |
| `cdas.checkpoint_dir` | Directory for CDAS checkpoints | `cdas_checkpoints` |
| `cdas.sampling_log_dir` | Directory for sampling logs | `sampling_logs` |
| `cdas.save_versioned_checkpoints` | Save versioned checkpoints | `false` |

