# ===== data/dataset.py =====
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CrackDataset(Dataset):
    """Simple dataset for crack prediction."""
    
    def __init__(self, data, labeled=True):
        self.data = data
        self.labeled = labeled
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.labeled:
            return sample['x_cat'], sample['x_cont'], sample['y']
        else:
            return sample['x_cat'], sample['x_cont']

class CrackPredictionDataset(Dataset):
    """Advanced dataset with preprocessing for crack prediction."""
    
    def __init__(self, data_path, labeled=True, preprocessor=None):
        self.data = pd.read_csv(data_path) if isinstance(data_path, str) else data_path
        self.labeled = labeled
        self.preprocessor = preprocessor
        
        # Default feature names
        self.categorical_features = ['tool_type', 'material', 'coolant_type', 'machining_method']
        self.continuous_features = ['cutting_speed', 'feed_rate', 'depth_of_cut', 
                                   'tool_wear', 'temperature', 'vibration']
        self.target = 'crack_length'
        
        # Preprocess if preprocessor is provided
        if self.preprocessor:
            self.data = self.preprocessor.transform(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Extract categorical features
        cat_features = [col for col in self.categorical_features if col in self.data.columns]
        x_cat = torch.tensor([row[col] for col in cat_features], dtype=torch.long)
        
        # Extract continuous features
        cont_features = [col for col in self.continuous_features if col in self.data.columns]
        x_cont = torch.tensor([row[col] for col in cont_features], dtype=torch.float32)
        
        if self.labeled and self.target in self.data.columns:
            y = torch.tensor(row[self.target], dtype=torch.float32)
            return x_cat, x_cont, y
        else:
            return x_cat, x_cont