import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from config import ModelConfig
import json


class Evaluator:
    """Classe pour l'évaluation complète du modèle"""
    
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
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
        
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    
    def evaluate_on_loader(
        self, 
        loader, 
        denormalize_fn,
        phase_name: str = "Test"
    ) -> Dict:
        """Évalue le modèle sur un DataLoader"""
        print(f"\n{'='*60}")
        print(f"Évaluation - Phase {phase_name}")
        print(f"{'='*60}")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
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
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        results = {
            'predictions': predictions,
            'targets': targets,
            'metrics_per_step': [],
            'phase': phase_name
        }
        
        print(f"\nMétriques par step de prédiction:")
        for step in range(self.config.pred_steps):
            y_true = targets[:, step, 0]
            y_pred = predictions[:, step, 0]
            metrics = self.calculate_metrics(y_true, y_pred)
            results['metrics_per_step'].append(metrics)
            
            print(f"\n  Step {step + 1}:")
            print(f"    RMSE: {metrics['RMSE']:.6f}")
            print(f"    MAE: {metrics['MAE']:.6f}")
            print(f"    MAPE: {metrics['MAPE']:.2f}%")
        
        # Métriques globales
        all_targets_flat = targets.reshape(-1)
        all_predictions_flat = predictions.reshape(-1)
        global_metrics = self.calculate_metrics(all_targets_flat, all_predictions_flat)
        results['global_metrics'] = global_metrics
        
        print(f"\n  Métriques globales:")
        print(f"    RMSE: {global_metrics['RMSE']:.6f}")
        print(f"    MAE: {global_metrics['MAE']:.6f}")
        print(f"    MAPE: {global_metrics['MAPE']:.2f}%")
        
        return results
    
    def get_predictions_for_phase(
        self,
        dataset,
        phase_indices: List[int],
        denormalize_fn
    ) -> Tuple[List[int], List[float]]:
        """
        Obtient toutes les prédictions (step 1) pour une phase donnée
        
        Returns:
            positions: Liste des positions temporelles
            predictions: Liste des valeurs prédites
        """
        positions = []
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for idx in phase_indices:
                if idx >= len(dataset):
                    break
                
                try:
                    x, _ = dataset[idx]
                    x = x.unsqueeze(0).to(self.device)
                    
                    # Prédiction
                    pred = self.model(x)
                    pred = pred.squeeze(0).cpu().numpy()
                    
                    # Dénormalisation (seulement step 1)
                    pred_denorm = denormalize_fn(pred[0:1])
                    
                    # Position dans la série (après la fenêtre d'entrée)
                    position = idx + self.config.window_size
                    
                    positions.append(position)
                    predictions.append(pred_denorm[0, 0])
                    
                except Exception as e:
                    print(f"Erreur idx {idx}: {e}")
                    continue
        
        return positions, predictions
    
    def plot_complete_analysis(
        self,
        full_dataset,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        denormalize_fn,
        save_dir: Path
    ):
        """
        Génère le graphique complet avec train/val/test ET leurs prédictions
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("Génération du graphique complet")
        print("="*60)
        
        # Récupération de toute la série réelle
        if hasattr(full_dataset, 'dataset'):
            all_values_norm = full_dataset.dataset.values_norm
            mean = full_dataset.dataset.mean
            std = full_dataset.dataset.std
        else:
            all_values_norm = full_dataset.values_norm
            mean = full_dataset.mean
            std = full_dataset.std
        
        # Dénormalisation de toute la série
        complete_true = denormalize_fn(all_values_norm).flatten()
        
        # Calcul des prédictions pour chaque phase
        print("  Calcul prédictions Train...")
        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        train_pos, train_pred = self.get_predictions_for_phase(
            train_subset, 
            list(range(len(train_indices))),
            denormalize_fn
        )
        
        print("  Calcul prédictions Validation...")
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)
        val_pos, val_pred = self.get_predictions_for_phase(
            val_subset,
            list(range(len(val_indices))),
            denormalize_fn
        )
        # Ajustement des positions pour validation
        val_pos = [p + train_indices[-1] + 1 for p in val_pos]
        
        print("  Calcul prédictions Test...")
        test_subset = torch.utils.data.Subset(full_dataset, test_indices)
        test_pos, test_pred = self.get_predictions_for_phase(
            test_subset,
            list(range(len(test_indices))),
            denormalize_fn
        )
        # Ajustement des positions pour test
        test_pos = [p + val_indices[-1] + 1 for p in test_pos]
        
        print(f"  ✓ Train: {len(train_pred)} prédictions")
        print(f"  ✓ Validation: {len(val_pred)} prédictions")
        print(f"  ✓ Test: {len(test_pred)} prédictions")
        
        # Création du graphique
        fig, ax = plt.subplots(figsize=(20, 8))
        
        time_axis = np.arange(len(complete_true))
        
        # Calcul des limites de zones
        train_end = train_indices[-1] + self.config.window_size + self.config.pred_steps
        val_end = val_indices[-1] + self.config.window_size + self.config.pred_steps
        
        # 1. Série réelle (ligne noire continue)
        ax.plot(time_axis, complete_true, 'k-', 
                label='Série Réelle', linewidth=1.5, alpha=0.8, zorder=1)
        
        # 2. Prédictions Train (points bleus)
        if len(train_pred) > 0:
            ax.scatter(train_pos, train_pred, 
                      color='blue', s=15, alpha=0.6, 
                      label=f'Prédictions Train (n={len(train_pred)})', 
                      zorder=3, edgecolors='darkblue', linewidths=0.5)
        
        # 3. Prédictions Validation (points verts)
        if len(val_pred) > 0:
            ax.scatter(val_pos, val_pred, 
                      color='green', s=15, alpha=0.6, 
                      label=f'Prédictions Validation (n={len(val_pred)})', 
                      zorder=3, edgecolors='darkgreen', linewidths=0.5)
        
        # 4. Prédictions Test (points rouges)
        if len(test_pred) > 0:
            ax.scatter(test_pos, test_pred, 
                      color='red', s=15, alpha=0.6, 
                      label=f'Prédictions Test (n={len(test_pred)})', 
                      zorder=3, edgecolors='darkred', linewidths=0.5)
        
        # 5. Lignes de séparation verticales
        ax.axvline(x=train_end, color='blue', linestyle='--', 
                  linewidth=2.5, label='Fin Train', zorder=2)
        ax.axvline(x=val_end, color='green', linestyle='--', 
                  linewidth=2.5, label='Fin Validation', zorder=2)
        
        # 6. Zones colorées de fond
        ax.axvspan(0, train_end, alpha=0.08, color='blue', zorder=0)
        ax.axvspan(train_end, val_end, alpha=0.08, color='green', zorder=0)
        ax.axvspan(val_end, len(complete_true), alpha=0.08, color='red', zorder=0)
        
        # 7. Annotations de zones
        y_pos = ax.get_ylim()[1] * 0.95
        ax.text(train_end/2, y_pos, 'TRAIN', 
               ha='center', va='top', fontsize=14, fontweight='bold', 
               color='blue', alpha=0.7)
        ax.text((train_end + val_end)/2, y_pos, 'VALIDATION', 
               ha='center', va='top', fontsize=14, fontweight='bold', 
               color='green', alpha=0.7)
        ax.text((val_end + len(complete_true))/2, y_pos, 'TEST', 
               ha='center', va='top', fontsize=14, fontweight='bold', 
               color='red', alpha=0.7)
        
        # Configuration des axes
        ax.set_xlabel('Index Temporel', fontsize=13, fontweight='bold')
        ax.set_ylabel('Valeur', fontsize=13, fontweight='bold')
        ax.set_title('Analyse Complète: Train / Validation / Test avec Prédictions', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'complete_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Graphique complet sauvegardé: {save_dir / 'complete_analysis.png'}")
        
        # Calcul et affichage des métriques par phase
        print("\n" + "="*60)
        print("Métriques par phase (Step 1 uniquement)")
        print("="*60)
        
        # Train
        if len(train_pred) > 0:
            train_true_vals = [complete_true[p] for p in train_pos]
            train_metrics = self.calculate_metrics(
                np.array(train_true_vals), 
                np.array(train_pred)
            )
            print(f"\nTRAIN:")
            print(f"  RMSE: {train_metrics['RMSE']:.6f}")
            print(f"  MAE: {train_metrics['MAE']:.6f}")
            print(f"  MAPE: {train_metrics['MAPE']:.2f}%")
        
        # Validation
        if len(val_pred) > 0:
            val_true_vals = [complete_true[p] for p in val_pos]
            val_metrics = self.calculate_metrics(
                np.array(val_true_vals), 
                np.array(val_pred)
            )
            print(f"\nVALIDATION:")
            print(f"  RMSE: {val_metrics['RMSE']:.6f}")
            print(f"  MAE: {val_metrics['MAE']:.6f}")
            print(f"  MAPE: {val_metrics['MAPE']:.2f}%")
        
        # Test
        if len(test_pred) > 0:
            test_true_vals = [complete_true[p] for p in test_pos]
            test_metrics = self.calculate_metrics(
                np.array(test_true_vals), 
                np.array(test_pred)
            )
            print(f"\nTEST:")
            print(f"  RMSE: {test_metrics['RMSE']:.6f}")
            print(f"  MAE: {test_metrics['MAE']:.6f}")
            print(f"  MAPE: {test_metrics['MAPE']:.2f}%")
    
    def plot_test_detailed(self, results: Dict, save_dir: Path):
        """Génère les graphiques détaillés pour la phase test"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        predictions = results['predictions']
        targets = results['targets']
        
        # 1. Comparaison pour chaque step
        n_steps = min(self.config.pred_steps, 6)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for step in range(n_steps):
            ax = axes[step]
            
            y_true = targets[:, step, 0]
            y_pred = predictions[:, step, 0]
            
            ax.plot(y_true, label='Vérité Terrain', alpha=0.7, linewidth=1.5, color='black')
            ax.plot(y_pred, label='Prédiction', alpha=0.7, linewidth=1.5, color='red')
            ax.set_title(f'Test - Step {step + 1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Échantillon')
            ax.set_ylabel('Valeur')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Métriques
            metrics = results['metrics_per_step'][step]
            textstr = f"RMSE: {metrics['RMSE']:.4f}\nMAE: {metrics['MAE']:.4f}\nMAPE: {metrics['MAPE']:.2f}%"
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'test_predictions_per_step.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scatter plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for step in range(n_steps):
            ax = axes[step]
            
            y_true = targets[:, step, 0]
            y_pred = predictions[:, step, 0]
            
            ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='red')
            
            # Ligne y=x
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, 
                   label='Prédiction Parfaite')
            
            ax.set_title(f'Scatter Test - Step {step + 1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Vérité Terrain')
            ax.set_ylabel('Prédiction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # R²
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            ax.text(0.05, 0.95, f'R² = {corr**2:.4f}', transform=ax.transAxes,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'test_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Évolution des métriques
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        steps = range(1, self.config.pred_steps + 1)
        rmse_values = [m['RMSE'] for m in results['metrics_per_step']]
        mae_values = [m['MAE'] for m in results['metrics_per_step']]
        mape_values = [m['MAPE'] for m in results['metrics_per_step']]
        
        axes[0].plot(steps, rmse_values, 'o-', linewidth=2, markersize=8, color='blue')
        axes[0].set_title('RMSE par Step', fontweight='bold')
        axes[0].set_xlabel('Step de Prédiction')
        axes[0].set_ylabel('RMSE')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(steps, mae_values, 'o-', linewidth=2, markersize=8, color='orange')
        axes[1].set_title('MAE par Step', fontweight='bold')
        axes[1].set_xlabel('Step de Prédiction')
        axes[1].set_ylabel('MAE')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(steps, mape_values, 'o-', linewidth=2, markersize=8, color='green')
        axes[2].set_title('MAPE par Step', fontweight='bold')
        axes[2].set_xlabel('Step de Prédiction')
        axes[2].set_ylabel('MAPE (%)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'test_metrics_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Graphiques test détaillés sauvegardés dans {save_dir}")