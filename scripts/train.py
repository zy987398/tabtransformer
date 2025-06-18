# ===== scripts/train.py =====
#!/usr/bin/env python
"""
Main training script for crack prediction model.

Usage:
    python train.py --config config.json --data-dir data/
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from data import CrackPredictionDataset, DataPreprocessor
from training import SemiSupervisedTrainer
from utils import plot_training_history, set_seed, create_logger, get_timestamp

def main():
    parser = argparse.ArgumentParser(description='Train crack prediction model')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='Directory containing data files')
    parser.add_argument('--labeled-data', type=str, default='labeled_data.csv',
                        help='Filename for labeled data')
    parser.add_argument('--unlabeled-data', type=str, default='unlabeled_data.csv',
                        help='Filename for unlabeled data')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    if args.output_dir is None:
        args.output_dir = f'results/run_{get_timestamp()}'
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Create logger
    logger = create_logger('training', os.path.join(args.output_dir, 'training.log'))
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info("Loading and preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Load labeled data
    labeled_data_path = os.path.join(args.data_dir, args.labeled_data)
    labeled_data = pd.read_csv(labeled_data_path)
    logger.info(f"Loaded {len(labeled_data)} labeled samples")
    
    # Fit preprocessor and transform data
    labeled_data = preprocessor.fit_transform(labeled_data)
    
    # Split into train/validation
    train_data, val_data = train_test_split(
        labeled_data, test_size=0.2, random_state=args.seed
    )
    
    # Load unlabeled data
    unlabeled_data_path = os.path.join(args.data_dir, args.unlabeled_data)
    if os.path.exists(unlabeled_data_path):
        unlabeled_data = pd.read_csv(unlabeled_data_path)
        unlabeled_data = preprocessor.transform(unlabeled_data)
        logger.info(f"Loaded {len(unlabeled_data)} unlabeled samples")
    else:
        unlabeled_data = None
        logger.warning("No unlabeled data found, training in supervised mode only")
    
    # Update config with categorical dimensions
    config['model']['cat_dims'] = preprocessor.cat_dims
    
    # Create datasets
    train_dataset = CrackPredictionDataset(train_data, labeled=True, preprocessor=None)
    val_dataset = CrackPredictionDataset(val_data, labeled=True, preprocessor=None)
    
    if unlabeled_data is not None:
        unlabeled_dataset = CrackPredictionDataset(
            unlabeled_data, labeled=False, preprocessor=None
        )
    else:
        unlabeled_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    if unlabeled_dataset:
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=config['training']['batch_size'] * 2,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    else:
        unlabeled_loader = None
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    if unlabeled_dataset:
        logger.info(f"Unlabeled samples: {len(unlabeled_dataset)}")
    
    # Initialize trainer
    trainer = SemiSupervisedTrainer(config, device=args.device)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader, unlabeled_loader)
    
    # Save final model
    trainer._save_checkpoint(
        os.path.join(args.output_dir, 'checkpoints/final_model.pth'),
        trainer.teacher_model
    )
    
    # Save preprocessor
    preprocessor.save(os.path.join(args.output_dir, 'preprocessor'))
    
    # Plot training history
    plot_training_history(history, os.path.join(args.output_dir, 'training_history.png'))
    
    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    logger.info(f"Training completed! Results saved to {args.output_dir}")
    
    # Print final metrics
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    logger.info(f"Final train loss: {final_train_loss:.4f}")
    logger.info(f"Final validation loss: {final_val_loss:.4f}")

if __name__ == '__main__':
    main()