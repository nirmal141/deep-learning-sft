# Math Answer Verification - Llama 3.1 Fine-tuning

Fine-tunes Llama 3.1 8B to verify correctness of mathematical solutions using LoRA and Unsloth.

## Overview
- **Model**: Meta-Llama-3.1-8B (4-bit quantized)
- **Task**: Binary classification (True/False) for math answer verification
- **Method**: LoRA fine-tuning with supervised training
- **Dataset**: nyu-dl-teach-maths-comp (90k training samples)

## Key Parameters
- LoRA rank: 16, alpha: 32
- Batch size: 4, gradient accumulation: 2
- Learning rate: 2e-4, 1 epoch
- Max sequence length: 2048 tokens

## Usage
Run cells sequentially in `DL_Kaggle_nb3964.ipynb`:
1. Install dependencies
2. Load model and dataset
3. Configure LoRA and train
4. Generate predictions on test set

## Output
Generates `submission.csv` with predictions for test dataset.

## Requirements
- GPU with 16GB+ VRAM (A100 recommended)
- Python 3.12+
- See notebook cell 1 for package versions

