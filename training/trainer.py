# ===== training/trainer.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy

from models import TabTransformer, PhysicsInformedLoss
from .semi_supervised import PseudoLabelGenerator
from .evaluation import Evaluator

class SemiSupervisedTrainer:
    """Main trainer for semi-supervised crack prediction."""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self._setup_models()
        
        # Initialize loss functions
        self.mse_loss = nn.MSELoss()
        self.physics_loss = PhysicsInformedLoss(
            C=config.get('physics_C', 1e-10),
            m=config.get('physics_m', 3.0)
        )
        
        # Initialize training components
        self.pseudo_label_generator = PseudoLabelGenerator(
            confidence_threshold=config['training']['confidence_threshold']
        )
        self.evaluator = Evaluator()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'physics_loss': [],
            'pseudo_label_count': [],
            'learning_rate': []
        }
        
    def _setup_models(self):
        """Initialize teacher and student models."""
        model_config = self.config['model']
        
        self.teacher_model = TabTransformer(**model_config).to(self.device)
        self.student_model = TabTransformer(**model_config).to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
    def train(self, train_loader, val_loader, unlabeled_loader=None):
        """Main training loop."""
        # Phase 1: Train teacher model
        print("=== Phase 1: Training Teacher Model ===")
        self._train_teacher(train_loader, val_loader)
        
        # Phase 2: Semi-supervised training with pseudo-labels
        if unlabeled_loader:
            print("\n=== Phase 2: Semi-Supervised Training ===")
            self._train_semi_supervised(train_loader, val_loader, unlabeled_loader)
        else:
            print("\n=== Phase 2: Supervised Fine-tuning ===")
            self._train_supervised(train_loader, val_loader)
        
        return self.history
    
    def _train_teacher(self, train_loader, val_loader):
        """Train teacher model on labeled data."""
        epochs = self.config['training']['teacher_epochs']
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(self.teacher_model, train_loader)
            
            # Validation
            val_loss, val_metrics = self.evaluator.evaluate(
                self.teacher_model, val_loader, self.device
            )
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint('teacher_best.pth', self.teacher_model)
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val MAE: {val_metrics["mae"]:.4f}')
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model and copy to student
        self.teacher_model.load_state_dict(
            torch.load('checkpoints/teacher_best.pth')['model_state_dict']
        )
        self.student_model.load_state_dict(self.teacher_model.state_dict())
    
    def _train_supervised(self, train_loader, val_loader):
        """Supervised fine-tuning when no unlabeled data is available."""
        epochs = self.config['training']['student_epochs'] // 2  # Fewer epochs for supervised
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(self.student_model, train_loader)
            val_loss, val_metrics = self.evaluator.evaluate(
                self.student_model, val_loader, self.device
            )
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val MAE: {val_metrics["mae"]:.4f}')
    
    def _train_semi_supervised(self, train_loader, val_loader, unlabeled_loader):
        """Semi-supervised training with pseudo-labels."""
        epochs = self.config['training']['student_epochs']
        update_freq = self.config['training']['pseudo_label_update_freq']
        
        # Generate initial pseudo-labels
        pseudo_data = self.pseudo_label_generator.generate(
            self.teacher_model, unlabeled_loader, self.device
        )
        self.history['pseudo_label_count'].append(len(pseudo_data))
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Update pseudo-labels periodically
            if epoch > 0 and epoch % update_freq == 0:
                print(f"\nUpdating pseudo-labels at epoch {epoch}")
                pseudo_data = self.pseudo_label_generator.generate(
                    self.teacher_model, unlabeled_loader, self.device
                )
                self.history['pseudo_label_count'].append(len(pseudo_data))
            
            # Train with labeled and pseudo-labeled data
            train_loss, physics_loss = self._train_epoch_semi_supervised(
                self.student_model, train_loader, pseudo_data
            )
            
            # Validation
            val_loss, val_metrics = self.evaluator.evaluate(
                self.student_model, val_loader, self.device
            )
            
            # Update teacher with EMA
            self._update_teacher_ema()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['physics_loss'].append(physics_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint('student_best.pth', self.student_model)
            
            print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val MAE: {val_metrics["mae"]:.4f}, '
                  f'Physics Loss: {physics_loss:.4f}')
    
    def _train_epoch(self, model, data_loader):
        """Train one epoch."""
        model.train()
        total_loss = 0
        
        for batch in tqdm(data_loader, desc='Training'):
            x_cat, x_cont, y = [x.to(self.device) for x in batch]
            
            # Forward pass
            predictions = model(x_cat, x_cont)
            loss = self.mse_loss(predictions.squeeze(), y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def _train_epoch_semi_supervised(self, model, labeled_loader, pseudo_data):
        """Train one epoch with semi-supervised learning."""
        model.train()
        total_loss = 0
        total_physics_loss = 0
        
        # Import augmentation functions
        from data.augmentation import mixup_data, feature_dropout
        
        lambda_u = self.config['training']['lambda_u']
        lambda_p = self.config['training'].get('lambda_p', 0.1)
        
        for batch in tqdm(labeled_loader, desc='Semi-supervised training'):
            x_cat, x_cont, y = [x.to(self.device) for x in batch]
            
            # Apply augmentations
            x_cat_aug, x_cont_aug, y_aug, _, _ = mixup_data(x_cat, x_cont, y)
            x_cont_aug = feature_dropout(x_cont_aug)
            
            # Labeled data loss
            predictions = model(x_cat_aug, x_cont_aug)
            labeled_loss = self.mse_loss(predictions.squeeze(), y_aug)
            
            # Physics loss
            physics_loss = self.physics_loss(predictions, x_cont_aug)
            
            # Pseudo-label loss
            pseudo_loss = 0
            if pseudo_data and len(pseudo_data) > 0:
                # Sample pseudo-labeled data
                n_pseudo = min(len(x_cat), len(pseudo_data))
                indices = np.random.choice(len(pseudo_data), n_pseudo, replace=False)
                
                pseudo_batch = [pseudo_data[i] for i in indices]
                px_cat = torch.stack([item['x_cat'] for item in pseudo_batch])
                px_cont = torch.stack([item['x_cont'] for item in pseudo_batch])
                py = torch.stack([item['y'] for item in pseudo_batch])
                pconf = torch.stack([item['confidence'] for item in pseudo_batch])
                
                # Predict on pseudo-labeled data
                pseudo_pred = model(px_cat, px_cont)
                
                # Weighted pseudo-label loss
                pseudo_loss = (pconf * F.mse_loss(
                    pseudo_pred.squeeze(), py, reduction='none'
                )).mean()
            
            # Total loss
            total_loss_batch = labeled_loss + lambda_u * pseudo_loss + lambda_p * physics_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_physics_loss += physics_loss.item()
        
        return total_loss / len(labeled_loader), total_physics_loss / len(labeled_loader)
    
    def _update_teacher_ema(self):
        """Update teacher model using exponential moving average."""
        alpha = self.config['training']['ema_decay']
        
        for teacher_param, student_param in zip(
            self.teacher_model.parameters(),
            self.student_model.parameters()
        ):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1-alpha)
    
    def _save_checkpoint(self, filename, model):
        """Save model checkpoint."""
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, os.path.join('checkpoints', filename))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def predict_with_uncertainty(self, x_cat, x_cont, n_samples=50):
        """Make predictions with uncertainty estimation."""
        self.teacher_model.eval()
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.teacher_model(x_cat, x_cont)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Calculate confidence intervals
        lower_95 = mean_pred - 1.96 * std_pred
        upper_95 = mean_pred + 1.96 * std_pred
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower_95': lower_95,
            'upper_95': upper_95
        }