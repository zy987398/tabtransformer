# ===== scripts/predict.py =====
#!/usr/bin/env python
"""
Prediction script for crack prediction model.

Usage:
    python predict.py --model checkpoints/best_model.pth --data test_data.csv
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from models import TabTransformer
from data import CrackPredictionDataset, DataPreprocessor
from training.evaluation import plot_predictions, plot_residuals, calculate_metrics
from utils import plot_uncertainty_analysis, create_logger

def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data for prediction')
    parser.add_argument('--preprocessor', type=str, default=None,
                        help='Path to saved preprocessor')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output file for predictions')
    parser.add_argument('--n-samples', type=int, default=50,
                        help='Number of MC samples for uncertainty estimation')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for prediction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Create logger
    logger = create_logger('prediction')
    
    # Load model checkpoint
    logger.info("Loading model...")
    checkpoint = torch.load(args.model, map_location=args.device)
    config = checkpoint['config']
    
    # Initialize model
    model = TabTransformer(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    # Load preprocessor
    logger.info("Loading preprocessor...")
    if args.preprocessor is None:
        # Try to find preprocessor in the same directory as model
        model_dir = os.path.dirname(os.path.dirname(args.model))
        preprocessor_path = os.path.join(model_dir, 'preprocessor')
    else:
        preprocessor_path = args.preprocessor
    
    preprocessor = DataPreprocessor(config)
    preprocessor.load(preprocessor_path)
    
    # Load and preprocess data
    logger.info("Loading data...")
    data = pd.read_csv(args.data)
    original_data = data.copy()
    data = preprocessor.transform(data)
    
    # Check if data has labels
    has_labels = preprocessor.target in data.columns
    
    # Create dataset and loader
    dataset = CrackPredictionDataset(data, labeled=has_labels, preprocessor=None)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Make predictions with uncertainty
    logger.info("Making predictions...")
    all_predictions = []
    all_uncertainties = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            if has_labels:
                x_cat, x_cont, y = [x.to(args.device) for x in batch]
                all_targets.append(y.cpu())
            else:
                x_cat, x_cont = [x.to(args.device) for x in batch]
            
            # Monte Carlo predictions
            batch_predictions = []
            for _ in range(args.n_samples):
                pred = model(x_cat, x_cont)
                batch_predictions.append(pred)
            
            batch_predictions = torch.stack(batch_predictions)
            mean_pred = batch_predictions.mean(dim=0)
            std_pred = batch_predictions.std(dim=0)
            
            all_predictions.append(mean_pred.cpu())
            all_uncertainties.append(std_pred.cpu())
    
    # Concatenate results
    predictions = torch.cat(all_predictions).squeeze().numpy()
    uncertainties = torch.cat(all_uncertainties).squeeze().numpy()
    
    # Create output dataframe
    results_df = original_data.copy()
    results_df['predicted_crack_length'] = predictions
    results_df['prediction_uncertainty'] = uncertainties
    results_df['lower_95_ci'] = predictions - 1.96 * uncertainties
    results_df['upper_95_ci'] = predictions + 1.96 * uncertainties
    
    # Calculate prediction intervals
    results_df['prediction_interval_width'] = results_df['upper_95_ci'] - results_df['lower_95_ci']
    
    # Save predictions
    results_df.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")
    
    # Print summary statistics
    logger.info("\nPrediction Summary:")
    logger.info(f"Mean prediction: {predictions.mean():.2f} ± {predictions.std():.2f}")
    logger.info(f"Mean uncertainty: {uncertainties.mean():.2f}")
    logger.info(f"Min prediction: {predictions.min():.2f}")
    logger.info(f"Max prediction: {predictions.max():.2f}")
    
    # If we have targets, calculate metrics and create visualizations
    if has_labels:
        targets = torch.cat(all_targets).numpy()
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, targets)
        
        logger.info("\nEvaluation Metrics:")
        for name, value in metrics.items():
            logger.info(f"{name.upper()}: {value:.4f}")
        
        # Save metrics
        with open(args.output.replace('.csv', '_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Create visualizations if requested
        if args.visualize:
            output_dir = os.path.dirname(args.output)
            if not output_dir:
                output_dir = '.'
            
            # Prediction scatter plot
            plot_predictions(predictions, targets, 
                           os.path.join(output_dir, 'prediction_scatter.png'))
            
            # Residual analysis
            plot_residuals(predictions, targets,
                         os.path.join(output_dir, 'residual_analysis.png'))
            
            # Uncertainty analysis
            plot_uncertainty_analysis(predictions, uncertainties, targets,
                                    os.path.join(output_dir, 'uncertainty_analysis.png'))
            
            logger.info("Visualization plots saved.")

if __name__ == '__main__':
    main()
