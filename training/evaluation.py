# ===== training/evaluation.py =====
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

class Evaluator:
    """Model evaluation utilities."""
    
    def evaluate(self, model, data_loader, device):
        """Evaluate model on a dataset."""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                x_cat, x_cont, y = [x.to(device) for x in batch]
                
                pred = model(x_cat, x_cont)
                
                predictions.append(pred.squeeze().cpu())
                targets.append(y.cpu())
        
        predictions = torch.cat(predictions).numpy()
        targets = torch.cat(targets).numpy()
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, targets)
        loss = metrics['mse']
        
        return loss, metrics

def calculate_metrics(predictions, targets):
    """Calculate regression metrics."""
    # Ensure arrays are 1D
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    return {
        'mse': mean_squared_error(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions),
        'r2': r2_score(targets, predictions),
        'mape': np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    }

def plot_predictions(predictions, targets, save_path='predictions.png'):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('Actual Crack Length (μm)')
    plt.ylabel('Predicted Crack Length (μm)')
    plt.title('Predictions vs Actual Values')
    
    # Add R² score
    r2 = r2_score(targets, predictions)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(predictions, targets, save_path='residuals.png'):
    """Plot residual analysis."""
    residuals = targets - predictions
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs predictions
    axes[0].scatter(predictions, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted Values')
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=50, edgecolor='black')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()