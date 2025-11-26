# ============================================================================
# testing_pred_v2.py - Prédiction Multi-Horizon (Stratégie DIRECT + Rolling)
# ============================================================================
"""
Stratégie de prédiction optimale basée sur la littérature:

1. DIRECT Multi-Step : Le modèle prédit directement H pas en une fois
2. Rolling Window : Pour les longues prédictions, on "roll" par chunks de H pas
3. Recalibration : On re-ancre périodiquement sur les vraies valeurs si disponibles

Références:
- Taieb et al. (2012) - Stratégie DIRECT vs RECURSIVE
- Marcellino et al. (2006) - Rolling forecasts
- Lim et al. (2021) - Temporal Fusion Transformers pour multi-horizon

Cette approche évite l'accumulation d'erreurs de la stratégie récursive.
"""

import math
from typing import Callable, Optional, Dict, Any, List, Iterator
import torch
import numpy as np
import pandas as pd


# ============================================================================
# PRÉDICTION DIRECT MULTI-STEP
# ============================================================================

@torch.no_grad()
def predict_direct_multistep(
    model: torch.nn.Module,
    values: List[float],
    norm_stats: Dict[str, float],
    window_size: int,
    horizon: int,
    n_total_steps: int,
    device: str = "cpu",
    inverse_fn: Optional[Callable] = None,
    confidence_level: float = 0.95,
    residual_std: Optional[float] = None,
    y_true_values: Optional[List[float]] = None,
    idx_start: int = 0,
    use_true_values_for_rolling: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    Prédiction multi-horizon avec stratégie DIRECT + Rolling.
    
    Le modèle prédit H pas d'un coup. Pour prédire N > H pas total:
    - On prédit les H premiers pas
    - On avance la fenêtre de H pas
    - On répète jusqu'à couvrir N pas
    
    Args:
        model: Modèle multi-horizon entraîné (sortie = horizon pas)
        values: Série complète (pour extraire la fenêtre initiale)
        norm_stats: {"mean": ..., "std": ...}
        window_size: Taille de la fenêtre d'entrée
        horizon: Nombre de pas prédits par le modèle en une fois
        n_total_steps: Nombre total de pas à prédire
        device: cpu/cuda/mps
        inverse_fn: Fonction de dénormalisation
        confidence_level: Niveau de confiance pour le halo
        residual_std: Écart-type des résidus (pour le halo)
        y_true_values: Vraies valeurs pour comparaison
        idx_start: Index de départ dans la série complète
        use_true_values_for_rolling: Si True, utilise les vraies valeurs
            pour avancer la fenêtre (oracle mode, pour debug)
    
    Yields:
        Événements pred_start, pred_point, pred_end
    """
    model.eval()
    model = model.to(device)
    
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    
    # Z-score pour l'intervalle de confiance
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
    
    if residual_std is None:
        residual_std = std * 0.05  # 5% par défaut
    
    # Nombre de "rolls" nécessaires
    n_rolls = math.ceil(n_total_steps / horizon)
    
    yield {
        "type": "pred_start",
        "n_steps": n_total_steps,
        "horizon": horizon,
        "n_rolls": n_rolls,
        "idx_start": idx_start,
        "strategy": "DIRECT_ROLLING"
    }
    
    # Stockage des résultats
    all_predictions = []
    all_low = []
    all_high = []
    all_true = []
    
    # Extraire la fenêtre initiale
    window_start = idx_start - window_size
    if window_start < 0:
        raise ValueError(f"Pas assez de données avant idx={idx_start}")
    
    current_window = np.array(values[window_start:idx_start], dtype=float)
    
    # Gestion des NaN
    if np.isnan(current_window).any():
        current_window = pd.Series(current_window).fillna(method='ffill').fillna(method='bfill').values
    
    # Normaliser la fenêtre
    current_window_norm = (current_window - mean) / std
    
    steps_predicted = 0
    roll_idx = 0
    
    while steps_predicted < n_total_steps:
        # Préparer l'input
        x_input = torch.FloatTensor(current_window_norm).unsqueeze(0).to(device)
        
        # Prédiction multi-horizon
        pred_horizon = model(x_input)  # (1, horizon)
        pred_horizon = pred_horizon.squeeze(0).cpu().numpy()  # (horizon,)
        
        # Traiter chaque pas de l'horizon
        for h in range(horizon):
            if steps_predicted >= n_total_steps:
                break
            
            global_idx = idx_start + steps_predicted
            pred_norm = pred_horizon[h]
            
            # Dénormalisation
            if inverse_fn is not None:
                pred_tensor = torch.tensor([[pred_norm]], dtype=torch.float32)
                pred_denorm = float(inverse_fn(pred_tensor).item())
            else:
                pred_denorm = float(pred_norm * std + mean)
            
            # Calcul du halo
            # L'incertitude croît avec la distance depuis le dernier ancrage
            distance_from_anchor = h + 1  # Distance depuis le début de ce roll
            uncertainty = residual_std * math.sqrt(distance_from_anchor)
            low_bound = pred_denorm - z_score * uncertainty
            high_bound = pred_denorm + z_score * uncertainty
            
            # Vraie valeur (si disponible)
            y_true = None
            if y_true_values is not None and steps_predicted < len(y_true_values):
                y_true = y_true_values[steps_predicted]
                if y_true is not None:
                    all_true.append(float(y_true))
            
            all_predictions.append(pred_denorm)
            all_low.append(low_bound)
            all_high.append(high_bound)
            
            yield {
                "type": "pred_point",
                "idx": global_idx,
                "step": steps_predicted,
                "roll": roll_idx,
                "h": h,
                "y": y_true,
                "yhat": pred_denorm,
                "low": low_bound,
                "high": high_bound,
            }
            
            steps_predicted += 1
        
        # Préparer le prochain roll : avancer la fenêtre
        if steps_predicted < n_total_steps:
            if use_true_values_for_rolling and y_true_values is not None:
                # Mode oracle : utiliser les vraies valeurs
                new_values = []
                for h in range(min(horizon, len(y_true_values) - (steps_predicted - horizon))):
                    true_idx = steps_predicted - horizon + h
                    if true_idx < len(y_true_values) and y_true_values[true_idx] is not None:
                        new_values.append((y_true_values[true_idx] - mean) / std)
                    else:
                        new_values.append(pred_horizon[h])
                
                # Shift la fenêtre
                shift = min(horizon, len(new_values))
                current_window_norm = np.roll(current_window_norm, -shift)
                current_window_norm[-shift:] = new_values[:shift]
            else:
                # Mode normal : utiliser les prédictions
                shift = min(horizon, len(pred_horizon))
                current_window_norm = np.roll(current_window_norm, -shift)
                current_window_norm[-shift:] = pred_horizon[:shift]
        
        roll_idx += 1
    
    # Métriques finales
    metrics = {}
    if all_true:
        all_true_arr = np.array(all_true)
        all_pred_arr = np.array(all_predictions[:len(all_true)])
        
        mse = float(np.mean((all_true_arr - all_pred_arr) ** 2))
        mae = float(np.mean(np.abs(all_true_arr - all_pred_arr)))
        rmse = float(np.sqrt(mse))
        
        var_y = np.var(all_true_arr)
        r2 = float(1.0 - mse / var_y) if var_y > 1e-10 else None
        
        metrics = {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}
    
    yield {
        "type": "pred_end",
        "n_steps": steps_predicted,
        "n_rolls": roll_idx,
        "all_predictions": all_predictions,
        "all_low": all_low,
        "all_high": all_high,
        "all_true": all_true if all_true else None,
        "metrics": metrics
    }


# ============================================================================
# PRÉDICTION HYBRIDE : DIRECT + RECALIBRATION
# ============================================================================

@torch.no_grad()
def predict_with_recalibration(
    model: torch.nn.Module,
    values: List[float],
    norm_stats: Dict[str, float],
    window_size: int,
    n_total_steps: int,
    device: str = "cpu",
    inverse_fn: Optional[Callable] = None,
    recalib_every: int = 50,
    confidence_level: float = 0.95,
    residual_std: Optional[float] = None,
    y_true_values: Optional[List[float]] = None,
    idx_start: int = 0,
) -> Iterator[Dict[str, Any]]:
    """
    Prédiction avec recalibration périodique.
    
    Stratégie:
    - Prédire `recalib_every` pas
    - Recalibrer sur les vraies valeurs (si disponibles)
    - Répéter
    
    C'est une approche "rolling forecast" classique en économétrie.
    
    Args:
        model: Modèle (peut être single-step ou multi-step)
        recalib_every: Nombre de pas entre chaque recalibration
        ... autres paramètres identiques à predict_direct_multistep
    """
    model.eval()
    model = model.to(device)
    
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
    
    if residual_std is None:
        residual_std = std * 0.05
    
    # Déterminer si le modèle est multi-horizon
    # On teste avec une entrée factice
    test_input = torch.randn(1, window_size).to(device)
    test_output = model(test_input)
    
    if test_output.ndim == 1:
        horizon = test_output.shape[0]
    elif test_output.ndim == 2:
        horizon = test_output.shape[1]
    else:
        horizon = 1
    
    is_multistep = horizon > 1
    
    yield {
        "type": "pred_start",
        "n_steps": n_total_steps,
        "recalib_every": recalib_every,
        "model_horizon": horizon,
        "is_multistep": is_multistep,
        "idx_start": idx_start,
        "strategy": "RECALIBRATION"
    }
    
    all_predictions = []
    all_low = []
    all_high = []
    all_true = []
    
    # Fenêtre initiale
    window_start = idx_start - window_size
    current_window = np.array(values[window_start:idx_start], dtype=float)
    
    if np.isnan(current_window).any():
        current_window = pd.Series(current_window).fillna(method='ffill').fillna(method='bfill').values
    
    current_window_norm = (current_window - mean) / std
    
    steps_since_recalib = 0
    
    for step in range(n_total_steps):
        global_idx = idx_start + step
        
        # Prédire
        x_input = torch.FloatTensor(current_window_norm).unsqueeze(0).to(device)
        pred = model(x_input)
        
        if pred.ndim == 2:
            pred = pred[0, 0]  # Premier pas de l'horizon
        elif pred.ndim == 1:
            pred = pred[0]
        
        pred_norm = float(pred.cpu().numpy())
        
        # Dénormalisation
        if inverse_fn is not None:
            pred_tensor = torch.tensor([[pred_norm]], dtype=torch.float32)
            pred_denorm = float(inverse_fn(pred_tensor).item())
        else:
            pred_denorm = float(pred_norm * std + mean)
        
        # Halo
        uncertainty = residual_std * math.sqrt(steps_since_recalib + 1)
        low_bound = pred_denorm - z_score * uncertainty
        high_bound = pred_denorm + z_score * uncertainty
        
        # Vraie valeur
        y_true = None
        if y_true_values is not None and step < len(y_true_values):
            y_true = y_true_values[step]
            if y_true is not None:
                all_true.append(float(y_true))
        
        all_predictions.append(pred_denorm)
        all_low.append(low_bound)
        all_high.append(high_bound)
        
        yield {
            "type": "pred_point",
            "idx": global_idx,
            "step": step,
            "y": y_true,
            "yhat": pred_denorm,
            "low": low_bound,
            "high": high_bound,
            "steps_since_recalib": steps_since_recalib
        }
        
        # Mise à jour de la fenêtre
        steps_since_recalib += 1
        
        # Recalibration ?
        if steps_since_recalib >= recalib_every and y_true is not None:
            # Recalibrer sur la vraie valeur
            true_norm = (y_true - mean) / std
            current_window_norm = np.roll(current_window_norm, -1)
            current_window_norm[-1] = true_norm
            steps_since_recalib = 0
        else:
            # Continuer avec la prédiction
            current_window_norm = np.roll(current_window_norm, -1)
            current_window_norm[-1] = pred_norm
    
    # Métriques
    metrics = {}
    if all_true:
        all_true_arr = np.array(all_true)
        all_pred_arr = np.array(all_predictions[:len(all_true)])
        
        mse = float(np.mean((all_true_arr - all_pred_arr) ** 2))
        mae = float(np.mean(np.abs(all_true_arr - all_pred_arr)))
        rmse = float(np.sqrt(mse))
        
        var_y = np.var(all_true_arr)
        r2 = float(1.0 - mse / var_y) if var_y > 1e-10 else None
        
        metrics = {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}
    
    yield {
        "type": "pred_end",
        "n_steps": n_total_steps,
        "all_predictions": all_predictions,
        "all_low": all_low,
        "all_high": all_high,
        "all_true": all_true if all_true else None,
        "metrics": metrics
    }


# ============================================================================
# WRAPPER PRINCIPAL
# ============================================================================

def test_model_prediction(
    model: torch.nn.Module,
    values: List[float],
    norm_stats: Dict[str, float],
    window_size: int,
    n_pred_steps: int,
    device: str = "cpu",
    inverse_fn: Optional[Callable] = None,
    strategy: str = "recalibration",  # "direct", "recalibration", "recursive"
    horizon: int = 10,
    recalib_every: int = 10,
    confidence_level: float = 0.95,
    residual_std: Optional[float] = None,
    y_true_values: Optional[List[float]] = None,
    idx_start: int = 0,
) -> Iterator[Dict[str, Any]]:
    """
    Interface unifiée pour la prédiction multi-step.
    
    Stratégies disponibles:
    - "direct": Prédiction multi-horizon directe (nécessite modèle multi-output)
    - "recalibration": Recalibration périodique sur les vraies valeurs
    - "recursive": Prédiction récursive classique (déconseillé)
    """
    
    if strategy == "direct":
        yield from predict_direct_multistep(
            model=model,
            values=values,
            norm_stats=norm_stats,
            window_size=window_size,
            horizon=horizon,
            n_total_steps=n_pred_steps,
            device=device,
            inverse_fn=inverse_fn,
            confidence_level=confidence_level,
            residual_std=residual_std,
            y_true_values=y_true_values,
            idx_start=idx_start,
        )
    
    elif strategy == "recalibration":
        yield from predict_with_recalibration(
            model=model,
            values=values,
            norm_stats=norm_stats,
            window_size=window_size,
            n_total_steps=n_pred_steps,
            device=device,
            inverse_fn=inverse_fn,
            recalib_every=recalib_every,
            confidence_level=confidence_level,
            residual_std=residual_std,
            y_true_values=y_true_values,
            idx_start=idx_start,
        )
    
    else:
        raise ValueError(f"Stratégie inconnue: {strategy}")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Test de testing_pred_v2.py")
    
    # Modèle factice multi-horizon
    class DummyMultiHorizon(torch.nn.Module):
        def __init__(self, window_size, horizon):
            super().__init__()
            self.fc = torch.nn.Linear(window_size, horizon)
        
        def forward(self, x):
            if x.ndim == 3:
                x = x.squeeze(-1)
            return self.fc(x)
    
    window_size = 15
    horizon = 5
    model = DummyMultiHorizon(window_size, horizon)
    
    # Données
    values = np.sin(np.linspace(0, 8 * np.pi, 200)).tolist()
    norm_stats = {"mean": 0.0, "std": 0.7}
    
    print("\n--- Test stratégie RECALIBRATION ---")
    for evt in test_model_prediction(
        model=model,
        values=values,
        norm_stats=norm_stats,
        window_size=window_size,
        n_pred_steps=20,
        strategy="recalibration",
        recalib_every=5,
        y_true_values=values[100:120],
        idx_start=100
    ):
        if evt["type"] == "pred_start":
            print(f"  Début: {evt['n_steps']} pas, stratégie={evt['strategy']}")
        elif evt["type"] == "pred_end":
            print(f"  Fin: {evt['n_steps']} pas prédits")
            if evt["metrics"]:
                print(f"  Métriques: R²={evt['metrics'].get('R2', 'N/A')}")
    
    print("\n✓ Test réussi!")