# ARC-AGI Challenge Solution

This repository contains a machine learning solution for the **ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence)** challenge, which tests AI systems' ability to perform abstract reasoning on visual grid-based puzzles.

## Overview

The ARC-AGI challenge consists of visual reasoning tasks where AI models must:
- Analyze input-output transformation patterns from training examples
- Apply learned abstract rules to solve new test cases
- Demonstrate human-like reasoning capabilities on novel problems

This project implements a test-time fine-tuning approach using **Llama 3.2 1B** model with LoRA (Low-Rank Adaptation) to solve ARC-AGI tasks efficiently.

## 📁 Project Structure

```
arc-agi/
├── README.md                           # Project documentation
├── dataloader.py                      # Data loading utilities
├── plot.py                            # Visualization tools
├── arc-agi.ipynb                      # Main experiment notebook
├── arc-prize-2025/                    # ARC-AGI dataset files
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_evaluation_challenges.json
│   ├── arc-agi_evaluation_solutions.json
│   ├── arc-agi_test_challenges.json
│   └── sample_submission.json
├── test-time-train/                   # Test-time training experiments
│   ├── README.md
│   ├── trelis-ttt.ipynb              # Current TTT implementation
│   └── archive/
│       └── trelis-ttt-20.ipynb       # Previous experiment versions
├── kaggle/                            # Kaggle-specific utilities
└── pyproject.toml                     # Project dependencies
```

## Key Features

### 1. **Test-Time Fine-Tuning (TTT)**
- Dynamic model adaptation during inference
- Uses Unsloth framework for efficient LoRA fine-tuning
- Optimized for GPU memory usage and training speed

### 2. **Advanced Inference Strategies**
- **Standard Generation**: Direct model output
- **Depth-First Search (DFS)**: Multiple candidate generation with best-first selection
- **Batch Processing**: Efficient parallel inference

### 3. **Data Augmentation**
- Color swapping for pattern generalization
- Grid rotations and reflections
- Training example shuffling

### 4. **Visualization Tools**
- Grid pattern visualization
- Training/test example plotting
- Performance analysis charts

## Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended)
- UV package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/Allenwang2004/arc-agi.git
cd arc-agi

# Install dependencies using UV
uv sync

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### Additional Requirements
```bash
# Install Unsloth for efficient fine-tuning
uv pip install unsloth

# Optional: Install Flash Attention for faster training
uv pip install flash-attn --no-build-isolation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.