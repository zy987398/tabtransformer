# ===== data/preprocessing.py =====
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import json

class DataPreprocessor:
    """Data preprocessor for crack prediction data."""
    
    def __init__(self, config):
        self.config = config
        self.categorical_features = config['data']['categorical_features']
        self.continuous_features = config['data']['continuous_features']
        self.target = config['data']['target']
        
        self.label_encoders = {}
        self.scaler = None
        self.cat_dims = []
        
    def fit(self, data):
        """Fit preprocessor on training data."""
        df = data.copy()
        
        # Handle missing values
        self._handle_missing_values(df)
        
        # Fit label encoders for categorical features
        self.cat_dims = []
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col])
                self.label_encoders[col] = le
                self.cat_dims.append(len(le.classes_))
        
        # Fit scaler for continuous features
        cont_features = [col for col in self.continuous_features if col in df.columns]
        if cont_features:
            self.scaler = StandardScaler()
            self.scaler.fit(df[cont_features])
        
        return self
    
    def transform(self, data):
        """Transform data using fitted preprocessor."""
        df = data.copy()
        
        # Handle missing values
        self._handle_missing_values(df)
        
        # Transform categorical features
        for col in self.categorical_features:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unknown categories
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        # Transform continuous features
        cont_features = [col for col in self.continuous_features if col in df.columns]
        if cont_features and self.scaler:
            df[cont_features] = self.scaler.transform(df[cont_features])
        
        return df
    
    def fit_transform(self, data):
        """Fit and transform data."""
        return self.fit(data).transform(data)
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataframe."""
        # Categorical features: fill with 'unknown'
        for col in self.categorical_features:
            if col in df.columns:
                df[col].fillna('unknown', inplace=True)
        
        # Continuous features: fill with median
        for col in self.continuous_features:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
    
    def save(self, path):
        """Save preprocessor to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save label encoders
        for name, le in self.label_encoders.items():
            joblib.dump(le, os.path.join(path, f'le_{name}.pkl'))
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        
        # Save configuration
        with open(os.path.join(path, 'preprocessor_config.json'), 'w') as f:
            json.dump({
                'cat_dims': self.cat_dims,
                'categorical_features': self.categorical_features,
                'continuous_features': self.continuous_features,
                'target': self.target
            }, f)
    
    def load(self, path):
        """Load preprocessor from disk."""
        # Load configuration
        with open(os.path.join(path, 'preprocessor_config.json'), 'r') as f:
            config = json.load(f)
            self.cat_dims = config['cat_dims']
            self.categorical_features = config['categorical_features']
            self.continuous_features = config['continuous_features']
            self.target = config['target']
        
        # Load label encoders
        self.label_encoders = {}
        for col in self.categorical_features:
            le_path = os.path.join(path, f'le_{col}.pkl')
            if os.path.exists(le_path):
                self.label_encoders[col] = joblib.load(le_path)
        
        # Load scaler
        scaler_path = os.path.join(path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        return self