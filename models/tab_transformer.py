# ===== models/tab_transformer.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TransformerBlock

class TabTransformer(nn.Module):
    """TabTransformer model for tabular data with mixed categorical and continuous features."""
    
    def __init__(self, 
                 cat_dims,
                 num_continuous,
                 dim=32,
                 depth=6,
                 heads=8,
                 dim_head=16,
                 mlp_hidden_mults=(4, 2),
                 mlp_act=nn.ReLU(),
                 num_special_tokens=1,
                 dropout=0.1,
                 mc_dropout=0.2):
        super().__init__()
        
        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, dim) for cat_dim in cat_dims
        ])
        
        # Continuous feature projection
        self.num_continuous = num_continuous
        if num_continuous > 0:
            self.continuous_embedder = nn.Linear(num_continuous, dim)
        
        # Special tokens (CLS)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Positional embeddings
        total_tokens = len(cat_dims) + num_special_tokens + (1 if num_continuous > 0 else 0)
        self.pos_embedding = nn.Parameter(torch.randn(1, total_tokens, dim))
        
        # Transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, dropout)
            for _ in range(depth)
        ])
        
        # Output MLP with Monte Carlo Dropout
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden_mults[0] * dim),
            mlp_act,
            nn.Dropout(mc_dropout),
            nn.Linear(mlp_hidden_mults[0] * dim, mlp_hidden_mults[1] * dim),
            mlp_act,
            nn.Dropout(mc_dropout),
            nn.Linear(mlp_hidden_mults[1] * dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_cat, x_cont=None):
        # Process categorical features
        cat_tokens = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_tokens.append(emb(x_cat[:, i]))
        cat_tokens = torch.stack(cat_tokens, dim=1)
        
        # Process continuous features
        tokens = [self.cls_token.expand(x_cat.size(0), -1, -1)]
        if x_cont is not None and self.num_continuous > 0:
            cont_token = self.continuous_embedder(x_cont).unsqueeze(1)
            tokens.append(cont_token)
        tokens.append(cat_tokens)
        
        # Concatenate all tokens
        x = torch.cat(tokens, dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Transformer encoding
        for transformer in self.transformer:
            x = transformer(x)
        
        # Use CLS token output
        x = x[:, 0]
        
        # Output prediction
        return self.to_logits(x)