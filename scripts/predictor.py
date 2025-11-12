import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Optional
from config import ModelConfig


class Predictor:
    """Classe pour faire des prédictions dans le futur"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        norm_stats: Dict,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.norm_stats = norm_stats
        self.device = device
        self.model.eval()
    
    def _prepare_input(self, data: Dict, exog_data: Optional[list] = None) -> torch.Tensor:
        """Prépare les données d'entrée pour la prédiction"""
        # Récupération des dernières valeurs
        values = np.array(data['values'][-self.config.window_size:], dtype=float)
        values = values.reshape(-1, 1)
        
        # Nettoyage si nécessaire
        if np.isnan(values).any():
            values = pd.Series(values.flatten()).fillna(method='ffill').fillna(method='bfill').values.reshape(-1, 1)
        
        # Normalisation
        values_norm = (values - self.norm_stats['mean']) / self.norm_stats['std']
        
        # Variables exogènes
        if exog_data is not None:
            exog_arrays = []
            for exog in exog_data:
                exog_values = np.array(exog['values'][-self.config.window_size:], dtype=float)
                exog_values = exog_values.reshape(-1, 1)
                exog_arrays.append(exog_values)
            exog = np.concatenate(exog_arrays, axis=1)
            
            # Normalisation
            exog_mean = np.array(self.norm_stats['exog_mean'])
            exog_std = np.array(self.norm_stats['exog_std'])
            exog_norm = (exog - exog_mean) / exog_std
            
            x = np.concatenate([values_norm, exog_norm], axis=1)
        else:
            x = values_norm
        
        return torch.FloatTensor(x).unsqueeze(0)
    
    def _denormalize(self, values: np.ndarray) -> np.ndarray:
        """Dénormalise les valeurs"""
        return values * self.norm_stats['std'] + self.norm_stats['mean']
    
    def _generate_future_timestamps(self, last_timestamps: list, n_steps: int) -> list:
        """Génère les timestamps futurs"""
        times = [datetime.fromisoformat(t) for t in last_timestamps]
        intervals = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        avg_interval = np.mean(intervals)
        
        last_time = times[-1]
        future_timestamps = []
        for i in range(1, n_steps + 1):
            future_time = last_time + timedelta(seconds=avg_interval * i)
            future_timestamps.append(future_time.isoformat())
        
        return future_timestamps
    
    def predict_future(
        self,
        data: Dict,
        exog_data: Optional[list] = None,
        n_future_steps: Optional[int] = None
    ) -> Dict:
        """
        Fait une prédiction dans le futur
        
        Args:
            data: Données historiques
            exog_data: Variables exogènes (si disponibles)
            n_future_steps: Nombre de steps à prédire (si None, utilise config.pred_steps)
        
        Returns:
            Dict avec timestamps et valeurs prédites
        """
        print("\n" + "="*60)
        print("PRÉDICTION DANS LE FUTUR")
        print("="*60)
        
        if n_future_steps is None:
            n_future_steps = self.config.pred_steps
        
        # Préparation
        x = self._prepare_input(data, exog_data).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            pred = self.model(x)
            pred = pred.squeeze(0).cpu().numpy()
        
        # Dénormalisation
        pred_denorm = self._denormalize(pred)
        
        # Timestamps futurs
        last_timestamps = data['timestamps'][-self.config.window_size:]
        future_timestamps = self._generate_future_timestamps(
            last_timestamps,
            n_future_steps
        )
        
        predictions = {
            'timestamps': future_timestamps[:n_future_steps],
            'values': pred_denorm[:n_future_steps, 0].tolist(),
            'input_window': {
                'timestamps': last_timestamps,
                'values': data['values'][-self.config.window_size:]
            }
        }
        
        # Affichage
        print("\nPrédictions futures:")
        for i, (ts, val) in enumerate(zip(predictions['timestamps'], predictions['values'])):
            print(f"  Step {i+1}: {ts} -> {val:.4f}")
        
        return predictions
    
    def plot_forecast(
        self,
        historical_data: Dict,
        predictions: Dict,
        n_history: int = 200,
        save_path: Optional[Path] = None
    ):
        """
        Visualise les prédictions futures avec l'historique
        
        Args:
            historical_data: Données historiques complètes
            predictions: Prédictions futures
            n_history: Nombre de points historiques à afficher
            save_path: Chemin de sauvegarde (optionnel)
        """
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Historique (derniers n_history points)
        hist_values = historical_data['values'][-n_history:]
        hist_times = range(len(hist_values))
        
        ax.plot(hist_times, hist_values, 'b-', label='Historique', linewidth=2, alpha=0.8)
        
        # Fenêtre d'entrée utilisée
        window_start = len(hist_values) - self.config.window_size
        window_values = historical_data['values'][-self.config.window_size:]
        window_times = range(window_start, len(hist_values))
        
        ax.plot(window_times, window_values, 'g-', linewidth=3, alpha=0.6, label='Fenêtre d\'entrée')
        
        # Prédictions futures
        pred_values = predictions['values']
        pred_start = len(hist_values)
        pred_times = range(pred_start, pred_start + len(pred_values))
        
        ax.plot(pred_times, pred_values, 'r-', marker='o', linestyle='--',
               linewidth=2, markersize=8, label='Prédiction Future', markerfacecolor='red')
        
        # Ligne de séparation
        ax.axvline(x=len(hist_values) - 1, color='orange', linestyle=':', 
                  linewidth=3, label='Présent → Futur')
        
        # Zone de prédiction
        ax.axvspan(pred_start, pred_times[-1], alpha=0.1, color='red', label='Zone de Prédiction')
        
        ax.set_xlabel('Index Temporel', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur', fontsize=12, fontweight='bold')
        ax.set_title('Prédiction Future sur Série Temporelle', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Graphique sauvegardé: {save_path}")
        
        plt.show()