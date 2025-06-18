# ===== data/augmentation.py =====
import torch
import numpy as np

def mixup_data(x_cat, x_cont, y, alpha=0.75):
    """
    MixUp data augmentation for tabular data.
    
    Args:
        x_cat: Categorical features
        x_cont: Continuous features
        y: Target values
        alpha: MixUp parameter
    
    Returns:
        Mixed features and targets
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x_cat.size(0)
    index = torch.randperm(batch_size).to(x_cat.device)
    
    # Only mix continuous features
    mixed_x_cat = x_cat
    mixed_x_cont = lam * x_cont + (1 - lam) * x_cont[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x_cat, mixed_x_cont, mixed_y, lam, index

def feature_dropout(x_cont, drop_rate=0.2):
    """
    Feature-level dropout for continuous features.
    
    Args:
        x_cont: Continuous features
        drop_rate: Dropout rate
    
    Returns:
        Features with dropout applied
    """
    if x_cont is None or not x_cont.requires_grad:
        return x_cont
    
    mask = torch.bernoulli(torch.ones_like(x_cont) * (1 - drop_rate))
    return x_cont * mask

def add_gaussian_noise(x_cont, noise_std=0.1):
    """
    Add Gaussian noise to continuous features.
    
    Args:
        x_cont: Continuous features
        noise_std: Standard deviation of noise
    
    Returns:
        Features with noise added
    """
    if x_cont is None:
        return x_cont
    
    noise = torch.randn_like(x_cont) * noise_std
    return x_cont + noise

def cutmix_data(x_cat, x_cont, y, beta=1.0):
    """
    CutMix data augmentation for tabular data.
    
    Args:
        x_cat: Categorical features
        x_cont: Continuous features
        y: Target values
        beta: Beta parameter for Beta distribution
    
    Returns:
        Mixed features and targets
    """
    batch_size = x_cat.size(0)
    index = torch.randperm(batch_size).to(x_cat.device)
    
    # Sample lambda from Beta distribution
    lam = np.random.beta(beta, beta)
    
    # Determine which features to mix
    num_features = x_cont.size(1)
    num_mix = int(num_features * (1 - lam))
    
    if num_mix > 0:
        # Randomly select features to mix
        mix_indices = torch.randperm(num_features)[:num_mix]
        
        # Mix selected continuous features
        mixed_x_cont = x_cont.clone()
        mixed_x_cont[:, mix_indices] = x_cont[index, :][:, mix_indices]
    else:
        mixed_x_cont = x_cont
    
    # Mix targets based on actual mixing ratio
    actual_lam = 1 - (num_mix / num_features)
    mixed_y = actual_lam * y + (1 - actual_lam) * y[index]
    
    return x_cat, mixed_x_cont, mixed_y, actual_lam, index