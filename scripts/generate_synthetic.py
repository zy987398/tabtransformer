# ===== scripts/generate_synthetic.py =====
#!/usr/bin/env python
"""
Generate synthetic data for testing the crack prediction model.

Usage:
    python generate_synthetic.py --n-labeled 300 --n-unlabeled 700 --output-dir data/
"""

import argparse
import numpy as np
import pandas as pd
import os
from utils import set_seed

def generate_synthetic_data(n_labeled=300, n_unlabeled=700, seed=42):
    """Generate synthetic crack prediction data."""
    np.random.seed(seed)
    
    def generate_features(n_samples):
        """Generate synthetic features."""
        data = {
            # Categorical features
            'tool_type': np.random.choice(['carbide', 'hss', 'ceramic', 'cbn'], n_samples,
                                        p=[0.4, 0.3, 0.2, 0.1]),
            'material': np.random.choice(['steel', 'aluminum', 'titanium', 'inconel'], n_samples,
                                       p=[0.35, 0.3, 0.2, 0.15]),
            'coolant_type': np.random.choice(['flood', 'mist', 'dry', 'mql'], n_samples,
                                           p=[0.4, 0.25, 0.2, 0.15]),
            'machining_method': np.random.choice(['turning', 'milling', 'drilling'], n_samples,
                                               p=[0.4, 0.35, 0.25]),
            
            # Continuous features with realistic ranges
            'cutting_speed': np.random.normal(200, 50, n_samples),      # m/min
            'feed_rate': np.random.lognormal(-2, 0.5, n_samples),       # mm/rev
            'depth_of_cut': np.random.lognormal(0, 0.5, n_samples),     # mm
            'tool_wear': np.random.exponential(0.05, n_samples),        # mm
            'temperature': np.random.normal(200, 50, n_samples),        # °C
            'vibration': np.random.exponential(0.5, n_samples)          # g
        }
        
        # Ensure positive values and reasonable ranges
        data['cutting_speed'] = np.clip(data['cutting_speed'], 50, 500)
        data['feed_rate'] = np.clip(data['feed_rate'], 0.05, 0.5)
        data['depth_of_cut'] = np.clip(data['depth_of_cut'], 0.1, 5.0)
        data['tool_wear'] = np.clip(data['tool_wear'], 0, 0.3)
        data['temperature'] = np.abs(data['temperature'])
        
        return pd.DataFrame(data)
    
    # Generate labeled data
    print(f"Generating {n_labeled} labeled samples...")
    labeled_df = generate_features(n_labeled)
    
    # Generate crack length based on physics-inspired relationships
    # Base crack length from cutting parameters
    base_crack = (
        0.1 * labeled_df['cutting_speed'] +
        50 * labeled_df['feed_rate'] +
        10 * labeled_df['depth_of_cut'] +
        100 * labeled_df['tool_wear'] +
        0.05 * labeled_df['temperature'] +
        20 * labeled_df['vibration']
    )
    
    # Material effects
    material_factor = labeled_df['material'].map({
        'steel': 1.0, 'aluminum': 0.7, 'titanium': 1.3, 'inconel': 1.5
    })
    
    # Tool effects
    tool_factor = labeled_df['tool_type'].map({
        'carbide': 0.8, 'hss': 1.2, 'ceramic': 0.9, 'cbn': 0.7
    })
    
    # Coolant effects
    coolant_factor = labeled_df['coolant_type'].map({
        'flood': 0.8, 'mist': 0.9, 'dry': 1.2, 'mql': 0.85
    })
    
    # Machining method effects
    method_factor = labeled_df['machining_method'].map({
        'turning': 1.0, 'milling': 1.1, 'drilling': 1.2
    })
    
    # Apply all factors
    crack_length = base_crack * material_factor * tool_factor * coolant_factor * method_factor
    
    # Add realistic noise and interactions
    noise = np.random.normal(0, 5, n_labeled)
    interaction_term = 0.001 * labeled_df['cutting_speed'] * labeled_df['feed_rate']
    
    crack_length = crack_length + noise + interaction_term
    
    # Ensure positive values and reasonable range
    crack_length = np.clip(crack_length, 0, 500)
    
    labeled_df['crack_length'] = crack_length
    
    # Generate unlabeled data
    print(f"Generating {n_unlabeled} unlabeled samples...")
    unlabeled_df = generate_features(n_unlabeled)
    
    # Add some distribution shift to unlabeled data (optional)
    # This simulates real-world scenarios where unlabeled data might be slightly different
    unlabeled_df['cutting_speed'] += np.random.normal(10, 5, n_unlabeled)
    unlabeled_df['cutting_speed'] = np.clip(unlabeled_df['cutting_speed'], 50, 500)
    
    return labeled_df, unlabeled_df

def create_data_splits(labeled_df, test_ratio=0.1, val_ratio=0.1):
    """Create train/val/test splits from labeled data."""
    n_samples = len(labeled_df)
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Shuffle data
    labeled_df = labeled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    test_df = labeled_df[:n_test]
    val_df = labeled_df[n_test:n_test+n_val]
    train_df = labeled_df[n_test+n_val:]
    
    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data')
    parser.add_argument('--n-labeled', type=int, default=300,
                        help='Number of labeled samples')
    parser.add_argument('--n-unlabeled', type=int, default=700,
                        help='Number of unlabeled samples')
    parser.add_argument('--output-dir', type=str, default='data/',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--create-splits', action='store_true',
                        help='Create train/val/test splits')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate data
    labeled_df, unlabeled_df = generate_synthetic_data(
        args.n_labeled, args.n_unlabeled, args.seed
    )
    
    # Save data
    if args.create_splits:
        # Create splits
        train_df, val_df, test_df = create_data_splits(labeled_df)
        
        # Save splits
        train_path = os.path.join(args.output_dir, 'train_data.csv')
        val_path = os.path.join(args.output_dir, 'val_data.csv')
        test_path = os.path.join(args.output_dir, 'test_data.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\nData splits created:")
        print(f"Training data: {train_path} ({len(train_df)} samples)")
        print(f"Validation data: {val_path} ({len(val_df)} samples)")
        print(f"Test data: {test_path} ({len(test_df)} samples)")
    else:
        # Save full datasets
        labeled_path = os.path.join(args.output_dir, 'labeled_data.csv')
        labeled_df.to_csv(labeled_path, index=False)
        
    unlabeled_path = os.path.join(args.output_dir, 'unlabeled_data.csv')
    unlabeled_df.to_csv(unlabeled_path, index=False)
    
    print(f"\nData generated successfully!")
    print(f"Labeled data: {labeled_path if not args.create_splits else 'Split into train/val/test'}")
    print(f"Unlabeled data: {unlabeled_path}")
    
    # Print statistics
    print(f"\nLabeled data statistics:")
    print(labeled_df['crack_length'].describe())
    
    # Create a sample visualization
    if args.create_splits:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        labeled_df['crack_length'].hist(bins=50, alpha=0.7, label='All labeled')
        if args.create_splits:
            train_df['crack_length'].hist(bins=50, alpha=0.5, label='Train')
            val_df['crack_length'].hist(bins=50, alpha=0.5, label='Val')
            test_df['crack_length'].hist(bins=50, alpha=0.5, label='Test')
        ax.set_xlabel('Crack Length (μm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Crack Length')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'data_distribution.png'))
        print(f"\nData distribution plot saved to {os.path.join(args.output_dir, 'data_distribution.png')}")

if __name__ == '__main__':
    main()