# ===== training/semi_supervised.py =====
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class PseudoLabelGenerator:
    """Generate pseudo-labels for unlabeled data."""
    
    def __init__(self, confidence_threshold=0.7, mc_samples=20):
        self.confidence_threshold = confidence_threshold
        self.mc_samples = mc_samples
    
    def generate(self, model, unlabeled_loader, device):
        """Generate pseudo-labels using Monte Carlo dropout."""
        # Enable dropout during prediction for Monte Carlo sampling
        model.train()
        pseudo_data = []

        with torch.no_grad():
            for batch in tqdm(unlabeled_loader, desc='Generating pseudo-labels'):
                x_cat, x_cont = [x.to(device) for x in batch]
                
                # Monte Carlo predictions
                predictions = []
                for _ in range(self.mc_samples):
                    pred = model(x_cat, x_cont)
                    predictions.append(pred)
                
                predictions = torch.stack(predictions)
                mean_pred = predictions.mean(dim=0)
                std_pred = predictions.std(dim=0)
                
                # Calculate confidence (inverse of uncertainty)
                confidence = 1.0 / (1.0 + std_pred.squeeze())
                
                # Filter high-confidence samples
                mask = confidence > self.confidence_threshold
                
                if mask.any():
                    for i in range(len(x_cat)):
                        if mask[i]:
                            pseudo_data.append({
                                'x_cat': x_cat[i],
                                'x_cont': x_cont[i],
                                'y': mean_pred[i].squeeze(),
                                'confidence': confidence[i]
                            })
        
        print(
            f'Generated {len(pseudo_data)} pseudo-labels '
            f'({len(pseudo_data)/len(unlabeled_loader.dataset)*100:.1f}% of unlabeled data)'
        )

        # Switch back to evaluation mode after generation
        model.eval()
        
        return pseudo_data

class TeacherStudentTrainer:
    """Teacher-Student training framework."""
    
    def __init__(self, teacher_model, student_model, ema_decay=0.999):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.ema_decay = ema_decay
    
    def update_teacher(self):
        """Update teacher model using EMA of student weights."""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(),
                self.student_model.parameters()
            ):
                teacher_param.data = (
                    self.ema_decay * teacher_param.data +
                    (1 - self.ema_decay) * student_param.data
                )
    
    def consistency_loss(self, student_pred, teacher_pred, confidence=None):
        """Calculate consistency loss between student and teacher predictions."""
        loss = F.mse_loss(student_pred, teacher_pred, reduction='none')
        
        if confidence is not None:
            loss = loss * confidence
        
        return loss.mean()