# Mathy Model Project

A machine learning project for generating mathematical art from formulas and text prompts.

## Project Overview

This project aims to create a model that can generate beautiful mathematical art visualizations from mathematical formulas, parametric equations, and descriptive text prompts. The model learns patterns from a curated dataset of mathematical art and can produce novel visualizations.

## Features

- **Dataset Management**: Curated collection of mathematical art with associated formulas and metadata
- **Model Training**: Custom architecture for formula-to-image generation
- **Inference Engine**: Generate art from mathematical expressions or text prompts
- **Validation Tools**: Verify dataset quality and model performance

## Project Structure

```
mathy_model_project/
├── dataset/           # Training data and metadata
├── scripts/           # Data ingestion and preprocessing
├── model/            # Model architecture and training
├── utils/            # Utility functions and helpers
├── checkpoints/      # Saved model weights
├── outputs/          # Generated outputs
├── inference/        # Inference scripts and demos
└── docs/            # Documentation
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the dataset:
   ```bash
   python scripts/ingest_math_art.py
   python scripts/verify_dataset.py
   ```

3. Train the model:
   ```bash
   python model/train.py
   ```

4. Generate art:
   ```bash
   python inference/run_prompt.py --prompt "spiral with golden ratio"
   ```

## Requirements

- Python 3.8+
- PyTorch
- See `requirements.txt` for full dependencies

## License

See LICENSE file for details.
