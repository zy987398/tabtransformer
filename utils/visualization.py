# ===== utils/visualization.py =====
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training/validation loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Physics loss
    if 'physics_loss' in history and history['physics_loss']:
        axes[0, 1].plot(history['physics_loss'], linewidth=2, color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Physics Loss')
        axes[0, 1].set_title('Physics Consistency Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Pseudo-label count
    if 'pseudo_label_count' in history and history['pseudo_label_count']:
        axes[1, 0].plot(history['pseudo_label_count'], 'o-', markersize=8)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Number of Pseudo Labels')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history and history['learning_rate']:
        axes[1, 1].plot(history['learning_rate'], linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(feature_names, importance_scores, save_path='feature_importance.png'):
    """Plot feature importance."""
    plt.figure(figsize=(10, 6))
    
    # Sort features by importance
    indices = np.argsort(importance_scores)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_scores = importance_scores[indices]
    
    # Create bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
    bars = plt.barh(range(len(sorted_features)), sorted_scores, color=colors)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center')
    
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance Analysis')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_analysis(predictions, uncertainties, targets, save_path='uncertainty_analysis.png'):
    """Plot uncertainty analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Predictions with uncertainty bands
    sorted_indices = np.argsort(targets)
    sorted_targets = targets[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    sorted_uncertainties = uncertainties[sorted_indices]
    
    axes[0, 0].plot(range(len(sorted_targets)), sorted_targets, 'b-', label='Actual', alpha=0.7, linewidth=2)
    axes[0, 0].plot(range(len(sorted_predictions)), sorted_predictions, 'r-', label='Predicted', alpha=0.7, linewidth=2)
    axes[0, 0].fill_between(
        range(len(sorted_predictions)),
        sorted_predictions - 1.96 * sorted_uncertainties,
        sorted_predictions + 1.96 * sorted_uncertainties,
        alpha=0.3, color='red', label='95% CI'
    )
    axes[0, 0].set_xlabel('Sample Index (sorted by actual value)')
    axes[0, 0].set_ylabel('Crack Length (μm)')
    axes[0, 0].set_title('Predictions with Uncertainty Bands')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Uncertainty vs error
    errors = np.abs(predictions - targets)
    axes[0, 1].scatter(uncertainties, errors, alpha=0.5, s=20)
    
    # Add trend line
    z = np.polyfit(uncertainties, errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(uncertainties.min(), uncertainties.max(), 100)
    axes[0, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    # Add correlation coefficient
    corr = np.corrcoef(uncertainties, errors)[0, 1]
    axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[0, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[0, 1].set_xlabel('Predicted Uncertainty (std)')
    axes[0, 1].set_ylabel('Absolute Error')
    axes[0, 1].set_title('Uncertainty vs Prediction Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Calibration plot
    confidence_levels = np.arange(0.1, 1.0, 0.1)
    empirical_coverage = []
    
    for conf in confidence_levels:
        z_score = np.abs(np.percentile(np.random.randn(10000), (1-conf)*100))
        lower = predictions - z_score * uncertainties
        upper = predictions + z_score * uncertainties
        coverage = np.mean((targets >= lower) & (targets <= upper))
        empirical_coverage.append(coverage)
    
    axes[1, 0].plot(confidence_levels, confidence_levels, 'k--', label='Perfect calibration')
    axes[1, 0].plot(confidence_levels, empirical_coverage, 'bo-', label='Model calibration')
    axes[1, 0].set_xlabel('Expected Coverage')
    axes[1, 0].set_ylabel('Empirical Coverage')
    axes[1, 0].set_title('Uncertainty Calibration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Uncertainty distribution
    axes[1, 1].hist(uncertainties, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(uncertainties.mean(), color='red', linestyle='--', 
                       label=f'Mean: {uncertainties.mean():.3f}')
    axes[1, 1].set_xlabel('Uncertainty (std)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Uncertainties')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(data, features, save_path='correlation_matrix.png'):
    """Plot correlation matrix for features."""
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr_matrix = data[features].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()