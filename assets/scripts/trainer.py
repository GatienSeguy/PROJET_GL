import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from config import ModelConfig


class Trainer:
    """Classe pour l'entraînement du modèle"""
    
    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: torch.device,
        save_dir: Path = Path('checkpoints')
    ):
        self.model = model
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Critère et optimiseur
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Historique
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def train_epoch(self, train_loader) -> float:
        """Entraîne le modèle pour une époque"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> float:
        """Valide le modèle"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1],
            'best_val_loss': self.best_val_loss
        }
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
        
        torch.save(checkpoint, self.save_dir / 'last_model.pth')
    
    def train(self, train_loader, val_loader) -> Tuple[list, list]:
        """Entraîne le modèle"""
        print("\n" + "="*60)
        print("Début de l'entraînement")
        print("="*60)
        
        for epoch in range(self.config.num_epochs):
            # Entraînement
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Affichage
            print(f'\nEpoch {epoch+1}/{self.config.num_epochs}:')
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss: {val_loss:.6f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Sauvegarde du meilleur modèle
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f'  ✓ Meilleur modèle sauvegardé (val_loss: {val_loss:.6f})')
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                print(f'\nEarly stopping après {epoch+1} époques')
                break
        
        # Sauvegarde finale
        self.save_checkpoint(epoch, is_best=False)
        self._plot_training_curves()
        
        return self.train_losses, self.val_losses
    
    def _plot_training_curves(self):
        """Visualise les courbes d'apprentissage"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Courbes d\'apprentissage')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Sauvegarde des données
        np.save(self.save_dir / 'train_losses.npy', self.train_losses)
        np.save(self.save_dir / 'val_losses.npy', self.val_losses)