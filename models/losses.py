# ===== models/losses.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss based on Paris law for crack growth."""
    
    def __init__(self, C=1e-10, m=3.0):
        super().__init__()
        self.C = C
        self.m = m
        
    def forward(self, predictions, features, params=None):
        """
        Calculate physics consistency loss based on Paris law.
        
        Args:
            predictions: Model predictions
            features: Input features containing physical parameters
            params: Optional physical parameters dict
        """
        # Extract stress intensity factor (simplified)
        cutting_speed = features[:, 0]
        feed_rate = features[:, 1]
        depth_of_cut = features[:, 2] if features.shape[1] > 2 else 1.0
        
        # Simplified stress intensity factor calculation
        delta_K = torch.sqrt(cutting_speed * feed_rate * depth_of_cut) * 10
        
        # Paris law prediction
        theoretical_growth = self.C * (delta_K ** self.m)
        
        # Simplified physics loss
        physics_loss = F.mse_loss(predictions.squeeze() * 0.01, theoretical_growth)
        
        return physics_loss

class ConsistencyLoss(nn.Module):
    """Consistency loss for semi-supervised learning."""
    
    def __init__(self, consistency_type='mse'):
        super().__init__()
        self.consistency_type = consistency_type
        
    def forward(self, pred1, pred2, confidence=None):
        """
        Calculate consistency loss between two predictions.
        
        Args:
            pred1: First prediction
            pred2: Second prediction
            confidence: Optional confidence weights
        """
        if self.consistency_type == 'mse':
            loss = F.mse_loss(pred1, pred2, reduction='none')
        elif self.consistency_type == 'kl':
            loss = F.kl_div(F.log_softmax(pred1, dim=-1), 
                           F.softmax(pred2, dim=-1), 
                           reduction='none')
        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")
        
        if confidence is not None:
            loss = loss * confidence
            
        return loss.mean()