import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict
from config import ModelConfig


class Evaluator:
    """Classe pour l'évaluation du modèle"""
    
    def __init__(self, model: nn.Module, config: ModelConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calcule les métriques RMSE, MAE et MAPE"""
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE avec protection division par zéro
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    
    def evaluate(self, test_loader, denormalize_fn) -> Dict:
        """Évalue le modèle sur le test set"""
        print("\n" + "="*60)
        print("Évaluation du modèle")
        print("="*60)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                
                # Prédiction
                pred = self.model(batch_x)
                pred = pred.cpu().numpy()
                batch_y = batch_y.numpy()
                
                # Dénormalisation
                for i in range(len(pred)):
                    pred_denorm = denormalize_fn(pred[i])
                    y_denorm = denormalize_fn(batch_y[i])
                    all_predictions.append(pred_denorm)
                    all_targets.append(y_denorm)
        
        # Conversion en arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calcul des métriques pour chaque step
        results = {
            'predictions': predictions,
            'targets': targets,
            'metrics_per_step': []
        }
        
        print("\nMétriques par step de prédiction:")
        for step in range(self.config.pred_steps):
            y_true = targets[:, step, 0]
            y_pred = predictions[:, step, 0]
            metrics = self.calculate_metrics(y_true, y_pred)
            results['metrics_per_step'].append(metrics)
            
            print(f"\nStep {step + 1}:")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  MAE: {metrics['MAE']:.6f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
        
        # Métriques globales
        all_targets_flat = targets.reshape(-1)
        all_predictions_flat = predictions.reshape(-1)
        global_metrics = self.calculate_metrics(all_targets_flat, all_predictions_flat)
        results['global_metrics'] = global_metrics
        
        print(f"\nMétriques globales:")
        print(f"  RMSE: {global_metrics['RMSE']:.6f}")
        print(f"  MAE: {global_metrics['MAE']:.6f}")
        print(f"  MAPE: {global_metrics['MAPE']:.2f}%")
        
        return results
    
    def plot_results(self, results: Dict, save_dir: Path):
        """Génère les visualisations"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        predictions = results['predictions']
        targets = results['targets']
        
        # 1. Comparaison prédictions vs réalité
        self._plot_predictions_comparison(predictions, targets, results, save_dir)
        
        # 2. Scatter plots
        self._plot_scatter(predictions, targets, results, save_dir)
        
        # 3. Métriques par step
        self._plot_metrics_evolution(results, save_dir)
        
        print(f"\n✓ Graphiques sauvegardés dans {save_dir}")
    
    def _plot_predictions_comparison(self, predictions, targets, results, save_dir):
        """Plot comparaison prédictions vs réalité"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for step in range(min(self.config.pred_steps, 6)):
            ax = axes[step]
            
            y_true = targets[:, step, 0]
            y_pred = predictions[:, step, 0]
            
            ax.plot(y_true, label='Vérité terrain', alpha=0.7, linewidth=1.5)
            ax.plot(y_pred, label='Prédiction', alpha=0.7, linewidth=1.5)
            ax.set_title(f'Prédiction step {step + 1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Échantillon')
            ax.set_ylabel('Valeur')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Métriques
            metrics = results['metrics_per_step'][step]
            textstr = f"RMSE: {metrics['RMSE']:.4f}\nMAE: {metrics['MAE']:.4f}\nMAPE: {metrics['MAPE']:.2f}%"
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scatter(self, predictions, targets, results, save_dir):
        """Plot scatter"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for step in range(min(self.config.pred_steps, 6)):
            ax = axes[step]
            
            y_true = targets[:, step, 0]
            y_pred = predictions[:, step, 0]
            
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)
            
            # Ligne y=x
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prédiction parfaite')
            
            ax.set_title(f'Scatter plot step {step + 1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Vérité terrain')
            ax.set_ylabel('Prédiction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # R²
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            ax.text(0.05, 0.95, f'R² = {corr**2:.4f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_evolution(self, results, save_dir):
        """Plot évolution des métriques"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        steps = range(1, self.config.pred_steps + 1)
        rmse_values = [m['RMSE'] for m in results['metrics_per_step']]
        mae_values = [m['MAE'] for m in results['metrics_per_step']]
        mape_values = [m['MAPE'] for m in results['metrics_per_step']]
        
        axes[0].plot(steps, rmse_values, marker='o', linewidth=2, markersize=8)
        axes[0].set_title('RMSE par step', fontweight='bold')
        axes[0].set_xlabel('Step de prédiction')
        axes[0].set_ylabel('RMSE')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(steps, mae_values, marker='o', color='orange', linewidth=2, markersize=8)
        axes[1].set_title('MAE par step', fontweight='bold')
        axes[1].set_xlabel('Step de prédiction')
        axes[1].set_ylabel('MAE')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(steps, mape_values, marker='o', color='green', linewidth=2, markersize=8)
        axes[2].set_title('MAPE par step', fontweight='bold')
        axes[2].set_xlabel('Step de prédiction')
        axes[2].set_ylabel('MAPE (%)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_per_step.png', dpi=300, bbox_inches='tight')
        plt.close()