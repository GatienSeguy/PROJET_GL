import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Optional
from config import ModelConfig


class Predictor:
    """Classe pour faire des prédictions"""
    
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
        """Prépare les données d'entrée"""
        # Récupération des dernières valeurs
        values = np.array(data['values'][-self.config.window_size:], dtype=float)
        values = values.reshape(-1, 1)
        
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
    
    def predict(self, data: Dict, exog_data: Optional[list] = None) -> Dict:
        """Fait une prédiction multi-step"""
        print("\n" + "="*60)
        print("Prédiction")
        print("="*60)
        
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
            self.config.pred_steps
        )
        
        predictions = {
            'timestamps': future_timestamps,
            'values': pred_denorm.flatten().tolist(),
            'input_window': {
                'timestamps': last_timestamps,
                'values': data['values'][-self.config.window_size:]
            }
        }
        
        # Affichage
        print("\nRésultats:")
        for i, (ts, val) in enumerate(zip(future_timestamps, predictions['values'])):
            print(f"  Step {i+1}: {ts} -> {val:.4f}")
        
        return predictions
    
    def plot_prediction(self, historical_data: Dict, predictions: Dict, save_path: Optional[Path] = None):
        """Visualise la prédiction"""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Historique
        hist_values = historical_data['values'][-100:]  # Dernières 100 valeurs
        hist_times = range(len(hist_values))
        
        ax.plot(hist_times, hist_values, label='Historique', color='blue', alpha=0.7, linewidth=2)
        
        # Prédictions
        pred_values = predictions['values']
        pred_start = len(hist_values)
        pred_times = range(pred_start, pred_start + len(pred_values))
        
        ax.plot(pred_times, pred_values, label='Prédiction', color='red', 
               marker='o', linestyle='--', linewidth=2, markersize=8)
        
        # Ligne de séparation
        ax.axvline(x=pred_start - 1, color='green', linestyle=':', 
                  linewidth=2, label='Début des prédictions')
        
        ax.set_xlabel('Index temporel', fontsize=12)
        ax.set_ylabel('Valeur', fontsize=12)
        ax.set_title('Prédiction de séries temporelles', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Graphique sauvegardé: {save_path}")
        
        plt.show()