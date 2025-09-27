# ARC-AGI Challenge Solution

A complete machine learning solution for the **ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence)** challenge using fine-tuned language models with LoRA (Low-Rank Adaptation).

## Overview

The ARC-AGI challenge tests AI systems on abstract visual reasoning tasks. This project demonstrates:

- **Pattern Recognition**: Analyzing grid-based input-output transformations
- **Abstract Reasoning**: Learning rules from few examples
- **Visual Intelligence**: Understanding spatial relationships and colors
- **Efficient Fine-tuning**: Parameter-efficient adaptation with LoRA


## Project Structure

```
arc-agi/
├── README.md                           # Project documentation
├── dataloader.py                       # ARC dataset loading and preprocessing
├── helper.py                           # Utility functions
├── plot.py                            # Grid visualization tools
├── train.ipynb                        # Main fine-tuning notebook
├── result.ipynb                       # Model evaluation and results
├── pyproject.toml                     # Dependencies management
├── uv.lock                            # Dependency lock file
├── arc-prize-2025/                    # Official ARC-AGI dataset
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_evaluation_challenges.json
│   ├── arc-agi_evaluation_solutions.json
│   └── arc-agi_test_challenges.json
├── model/                             # Saved fine-tuned models
└── .venv/                            # Virtual environment
```

## Key Features

### **Efficient Fine-tuning**
- **LoRA (Low-Rank Adaptation)**: Train only 0.65% of model parameters
- **Memory Efficient**: Works on both CPU and GPU
- **Fast Training**: Quick adaptation to ARC tasks

### **Data Augmentation**
- **Color Swapping**: Generalize across different color schemes
- **Grid Transformations**: Rotations, mirrors, shuffling
- **Pattern Variations**: Multiple perspectives of the same task

### **Comprehensive Evaluation**
- **Visual Comparison**: Side-by-side input/output/prediction plots
- **Accuracy Metrics**: Pixel-level and grid-level accuracy
- **Error Analysis**: Detailed visualization of prediction differences

### **Modular Design**
- Clean separation of data loading, training, and evaluation
- Reusable components for different model architectures
- Easy experimentation with different hyperparameters

## Quick Start

### Prerequisites
- Python 3.12+
- UV package manager (recommended) or pip
- 4GB+ RAM (8GB+ recommended)
- Optional: CUDA-compatible GPU for faster training

### Installation

```bash
# Clone the repository
git clone https://github.com/Allenwang2004/arc-agi.git
cd arc-agi

# Install dependencies with UV (recommended)
uv sync

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### Running the Experiments

1. **Training**: Open and run `train.ipynb`
   - Loads ARC training data
   - Fine-tunes GPT-2 with LoRA
   - Saves the trained model

2. **Evaluation**: Open and run `result.ipynb`
   - Loads fine-tuned model
   - Tests on evaluation dataset
   - Generates visual comparisons


## License

This project is open source and available under the [MIT License](LICENSE).