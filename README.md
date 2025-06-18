# TabTransformer Crack Prediction

This project implements a semi-supervised TabTransformer model for predicting crack length in machining processes. It combines physics-inspired losses, teacher--student training and several data augmentation techniques to improve generalization on tabular data.

## Repository Structure

- `data/` &mdash; dataset classes, preprocessing utilities and augmentation methods.
- `models/` &mdash; TabTransformer architecture, custom layers and loss functions.
- `training/` &mdash; training utilities including the teacher–student framework and evaluation helpers.
- `scripts/` &mdash; command line scripts for training, prediction and synthetic data generation.
- `config.json` &mdash; example configuration used by the training script.

## Getting Started

### 1. Prepare Data

You can create a synthetic dataset for experimentation:

```bash
python scripts/generate_synthetic.py --create-splits --output-dir data/
```

This generates labelled and unlabelled CSV files in `data/` and optionally train/validation/test splits.

### 2. Train the Model

Use the `train.py` script with a configuration file:

```bash
python scripts/train.py --config config.json --data-dir data/ \
    --labeled-data labeled_data.csv --unlabeled-data unlabeled_data.csv \
    --output-dir results/
```

During training the script saves checkpoints and logs to the specified output directory.

### 3. Make Predictions

After training you can run inference with uncertainty estimation:

```bash
python scripts/predict.py --model results/checkpoints/final_model.pth \
    --data my_test_data.csv --output predictions.csv --visualize
```

The command produces a CSV with predicted crack lengths and uncertainty bounds, and optional diagnostic plots.

## Features

- **TabTransformer architecture** for mixed categorical and continuous features.
- **Physics-informed loss** enforcing consistency with Paris law.
- **Teacher–student semi-supervised training** using pseudo labels.
- **Data augmentation** including MixUp, feature dropout, Gaussian noise and CutMix.
- **Comprehensive visualization tools** for training curves, feature importance and uncertainty analysis.

## Requirements

The code relies on PyTorch, pandas, scikit-learn and standard scientific Python libraries. Install the dependencies with your favourite package manager, e.g.:

```bash
pip install torch pandas scikit-learn matplotlib seaborn tqdm
```

## License

This project is released under the MIT License.

