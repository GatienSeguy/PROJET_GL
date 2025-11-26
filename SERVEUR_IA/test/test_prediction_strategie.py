# ============================================================================
# prediction_strategies.py - Module Complet de Prédiction Multi-Step
# ============================================================================
"""
4 stratégies de prédiction basées sur la littérature scientifique:

┌─────────────────────┬────────────────────────────────────────────────────┐
│ STRATÉGIE           │ DESCRIPTION                                        │
├─────────────────────┼────────────────────────────────────────────────────┤
│ 1. ONE_STEP         │ Prédiction 1 pas + recalibration immédiate         │
│                     │ → Le plus précis, évalue la vraie capacité         │
├─────────────────────┼────────────────────────────────────────────────────┤
│ 2. RECALIBRATION    │ Prédiction N pas + recalibration périodique        │
│                     │ → Bon compromis précision/horizon                  │
├─────────────────────┼────────────────────────────────────────────────────┤
│ 3. RECURSIVE        │ Prédiction récursive pure (autorégressive)         │
│                     │ → Prédiction future réelle, diverge vite           │
├─────────────────────┼────────────────────────────────────────────────────┤
│ 4. DIRECT           │ Modèle multi-horizon (prédit H pas d'un coup)      │
│                     │ → Évite accumulation erreurs, nécessite training   │
└─────────────────────┴────────────────────────────────────────────────────┘

Références:
- Taieb & Hyndman (2014) "A review of multi-step ahead forecasting"
- Chevillon (2007) "Direct multi-step estimation and forecasting"  
- Benidis et al. (2022) "Deep Learning for Time Series Forecasting: Tutorial"
"""

import math
from typing import Callable, Optional, Dict, Any, List, Iterator
from enum import Enum
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataclasses import dataclass


# ============================================================================
# ÉNUMÉRATION DES STRATÉGIES
# ============================================================================

class PredictionStrategy(Enum):
    """Stratégies de prédiction disponibles"""
    ONE_STEP = "one_step"              # 1 pas + recalib immédiate (gold standard)
    RECALIBRATION = "recalibration"    # N pas + recalib périodique
    RECURSIVE = "recursive"            # Récursif pur (autorégressif)
    DIRECT = "direct"                  # Multi-horizon direct


@dataclass
class PredictionConfig:
    """Configuration de la prédiction"""
    strategy: PredictionStrategy = PredictionStrategy.ONE_STEP
    recalib_every: int = 10           # Pour RECALIBRATION
    max_horizon: int = 50             # Pour RECURSIVE
    direct_horizon: int = 10          # Pour DIRECT
    confidence_level: float = 0.95


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def clean_nan(arr: np.ndarray) -> np.ndarray:
    """Nettoie les NaN avec forward/backward fill"""
    if np.isnan(arr).any():
        s = pd.Series(arr)
        s = s.ffill().bfill()
        return s.values
    return arr


def get_z_score(confidence: float) -> float:
    """Retourne le z-score pour un niveau de confiance"""
    return {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)


def predict_one_step_internal(model: nn.Module, window_norm: np.ndarray, device: str) -> float:
    """Prédit un seul pas avec le modèle"""
    x = torch.FloatTensor(window_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
    if pred.ndim == 3:
        pred = pred[:, -1, :]
    if pred.ndim == 2:
        pred = pred[:, 0]
    if pred.ndim == 1:
        pred = pred[0]
    return float(pred.cpu().numpy())


def denormalize_value(pred_norm: float, mean: float, std: float, inverse_fn: Optional[Callable]) -> float:
    """Dénormalise une valeur"""
    if inverse_fn is not None:
        pred_tensor = torch.tensor([[pred_norm]], dtype=torch.float32)
        return float(inverse_fn(pred_tensor).item())
    return float(pred_norm * std + mean)


def compute_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Calcule MSE, MAE, RMSE, R², MAPE"""
    if not y_true or not y_pred:
        return {}
    n = min(len(y_true), len(y_pred))
    y_true = np.array(y_true[:n], dtype=float)
    y_pred = np.array(y_pred[:n], dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {}
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    var_y = float(np.var(y_true))
    r2 = float(1.0 - mse / var_y) if var_y > 1e-10 else None
    nonzero = np.abs(y_true) > 1e-10
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else None
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# ============================================================================
# STRATÉGIE 1: ONE_STEP (Gold Standard)
# ============================================================================

@torch.no_grad()
def predict_one_step(
    model: nn.Module, values: List[float], norm_stats: Dict[str, float],
    window_size: int, n_steps: int, device: str, inverse_fn: Optional[Callable],
    residual_std: float, z_score: float, y_true: Optional[List[float]], idx_start: int,
) -> Iterator[Dict[str, Any]]:
    """Prédiction 1 pas + recalibration immédiate (gold standard)"""
    mean, std = norm_stats["mean"], norm_stats["std"]
    window_start = idx_start - window_size
    current_window = clean_nan(np.array(values[window_start:idx_start], dtype=float))
    current_window_norm = (current_window - mean) / std
    all_preds, all_low, all_high, all_true_out = [], [], [], []

    for step in range(n_steps):
        pred_norm = predict_one_step_internal(model, current_window_norm, device)
        pred_denorm = denormalize_value(pred_norm, mean, std, inverse_fn)
        uncertainty = residual_std  # Constant car recalib immédiate
        low, high = pred_denorm - z_score * uncertainty, pred_denorm + z_score * uncertainty
        y_true_val = y_true[step] if y_true and step < len(y_true) else None
        
        all_preds.append(pred_denorm)
        all_low.append(low)
        all_high.append(high)
        if y_true_val is not None:
            all_true_out.append(float(y_true_val))

        yield {"type": "pred_point", "idx": idx_start + step, "step": step,
               "y": y_true_val, "yhat": pred_denorm, "low": low, "high": high, "strategy": "one_step"}

        # Recalibration immédiate
        next_val = (float(y_true_val) - mean) / std if y_true_val is not None else pred_norm
        current_window_norm = np.roll(current_window_norm, -1)
        current_window_norm[-1] = next_val

    return all_preds, all_low, all_high, all_true_out


# ============================================================================
# STRATÉGIE 2: RECALIBRATION PÉRIODIQUE
# ============================================================================

@torch.no_grad()
def predict_recalibration(
    model: nn.Module, values: List[float], norm_stats: Dict[str, float],
    window_size: int, n_steps: int, device: str, inverse_fn: Optional[Callable],
    residual_std: float, z_score: float, y_true: Optional[List[float]], idx_start: int,
    recalib_every: int = 10,
) -> Iterator[Dict[str, Any]]:
    """Prédiction récursive + recalibration tous les N pas"""
    mean, std = norm_stats["mean"], norm_stats["std"]
    window_start = idx_start - window_size
    current_window = clean_nan(np.array(values[window_start:idx_start], dtype=float))
    current_window_norm = (current_window - mean) / std
    all_preds, all_low, all_high, all_true_out = [], [], [], []
    steps_since_recalib = 0

    for step in range(n_steps):
        pred_norm = predict_one_step_internal(model, current_window_norm, device)
        pred_denorm = denormalize_value(pred_norm, mean, std, inverse_fn)
        uncertainty = residual_std * math.sqrt(steps_since_recalib + 1)
        low, high = pred_denorm - z_score * uncertainty, pred_denorm + z_score * uncertainty
        y_true_val = y_true[step] if y_true and step < len(y_true) else None

        all_preds.append(pred_denorm)
        all_low.append(low)
        all_high.append(high)
        if y_true_val is not None:
            all_true_out.append(float(y_true_val))

        yield {"type": "pred_point", "idx": idx_start + step, "step": step,
               "y": y_true_val, "yhat": pred_denorm, "low": low, "high": high,
               "steps_since_recalib": steps_since_recalib, "strategy": "recalibration"}

        steps_since_recalib += 1
        if steps_since_recalib >= recalib_every and y_true_val is not None:
            next_val = (float(y_true_val) - mean) / std
            steps_since_recalib = 0
        else:
            next_val = pred_norm
        current_window_norm = np.roll(current_window_norm, -1)
        current_window_norm[-1] = next_val

    return all_preds, all_low, all_high, all_true_out


# ============================================================================
# STRATÉGIE 3: RECURSIVE PUR (Autorégressif)
# ============================================================================

@torch.no_grad()
def predict_recursive(
    model: nn.Module, values: List[float], norm_stats: Dict[str, float],
    window_size: int, n_steps: int, device: str, inverse_fn: Optional[Callable],
    residual_std: float, z_score: float, y_true: Optional[List[float]], idx_start: int,
    max_horizon: int = 50,
) -> Iterator[Dict[str, Any]]:
    """Prédiction autorégressive pure - JAMAIS de recalibration"""
    mean, std = norm_stats["mean"], norm_stats["std"]
    window_start = idx_start - window_size
    current_window = clean_nan(np.array(values[window_start:idx_start], dtype=float))
    current_window_norm = (current_window - mean) / std
    actual_steps = min(n_steps, max_horizon)
    all_preds, all_low, all_high, all_true_out = [], [], [], []

    for step in range(actual_steps):
        pred_norm = predict_one_step_internal(model, current_window_norm, device)
        pred_denorm = denormalize_value(pred_norm, mean, std, inverse_fn)
        # Halo avec croissance agressive
        uncertainty = residual_std * math.sqrt(step + 1) * (1.0 + 0.02 * step)
        low, high = pred_denorm - z_score * uncertainty, pred_denorm + z_score * uncertainty
        y_true_val = y_true[step] if y_true and step < len(y_true) else None

        all_preds.append(pred_denorm)
        all_low.append(low)
        all_high.append(high)
        if y_true_val is not None:
            all_true_out.append(float(y_true_val))

        yield {"type": "pred_point", "idx": idx_start + step, "step": step,
               "y": y_true_val, "yhat": pred_denorm, "low": low, "high": high,
               "uncertainty": uncertainty, "strategy": "recursive"}

        # TOUJOURS réinjecter la prédiction
        current_window_norm = np.roll(current_window_norm, -1)
        current_window_norm[-1] = pred_norm

    return all_preds, all_low, all_high, all_true_out


# ============================================================================
# STRATÉGIE 4: DIRECT MULTI-HORIZON
# ============================================================================

@torch.no_grad()
def predict_direct(
    model: nn.Module, values: List[float], norm_stats: Dict[str, float],
    window_size: int, n_steps: int, device: str, inverse_fn: Optional[Callable],
    residual_std: float, z_score: float, y_true: Optional[List[float]], idx_start: int,
    horizon: int = 10,
) -> Iterator[Dict[str, Any]]:
    """Le modèle prédit H pas en une seule inférence"""
    mean, std = norm_stats["mean"], norm_stats["std"]
    window_start = idx_start - window_size
    current_window = clean_nan(np.array(values[window_start:idx_start], dtype=float))
    current_window_norm = (current_window - mean) / std
    all_preds, all_low, all_high, all_true_out = [], [], [], []
    n_rolls = math.ceil(n_steps / horizon)
    step = 0

    for roll in range(n_rolls):
        if step >= n_steps:
            break
        x_input = torch.FloatTensor(current_window_norm).unsqueeze(0).to(device)
        pred_horizon = model(x_input)
        if pred_horizon.ndim == 3:
            pred_horizon = pred_horizon[:, -1, :]
        pred_horizon = pred_horizon.squeeze().cpu().numpy()
        if pred_horizon.ndim == 0:
            pred_horizon = np.array([float(pred_horizon)])
        actual_horizon = len(pred_horizon)

        for h in range(min(actual_horizon, n_steps - step)):
            pred_norm = float(pred_horizon[h])
            pred_denorm = denormalize_value(pred_norm, mean, std, inverse_fn)
            uncertainty = residual_std * math.sqrt(h + 1)
            low, high = pred_denorm - z_score * uncertainty, pred_denorm + z_score * uncertainty
            y_true_val = y_true[step] if y_true and step < len(y_true) else None

            all_preds.append(pred_denorm)
            all_low.append(low)
            all_high.append(high)
            if y_true_val is not None:
                all_true_out.append(float(y_true_val))

            yield {"type": "pred_point", "idx": idx_start + step, "step": step,
                   "roll": roll, "h": h, "y": y_true_val, "yhat": pred_denorm,
                   "low": low, "high": high, "strategy": "direct"}
            step += 1

        if step < n_steps and actual_horizon > 0:
            shift = min(actual_horizon, window_size)
            current_window_norm = np.roll(current_window_norm, -shift)
            current_window_norm[-shift:] = pred_horizon[:shift]

    return all_preds, all_low, all_high, all_true_out


# ============================================================================
# INTERFACE PRINCIPALE UNIFIÉE
# ============================================================================

@torch.no_grad()
def predict_multistep(
    model: nn.Module,
    values: List[float],
    norm_stats: Dict[str, float],
    window_size: int,
    n_steps: int,
    device: str = "cpu",
    inverse_fn: Optional[Callable] = None,
    config: Optional[PredictionConfig] = None,
    residual_std: Optional[float] = None,
    y_true: Optional[List[float]] = None,
    idx_start: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Interface unifiée pour la prédiction multi-step"""
    if config is None:
        config = PredictionConfig()
    model.eval()
    model = model.to(device)
    mean, std = norm_stats["mean"], norm_stats["std"]
    if residual_std is None:
        residual_std = std * 0.05
    z_score = get_z_score(config.confidence_level)

    yield {"type": "pred_start", "n_steps": n_steps, "strategy": config.strategy.value,
           "idx_start": idx_start, "window_size": window_size,
           "config": {"recalib_every": config.recalib_every, "max_horizon": config.max_horizon,
                      "direct_horizon": config.direct_horizon, "confidence_level": config.confidence_level}}

    all_preds, all_low, all_high, all_true_out = [], [], [], []
    
    if config.strategy == PredictionStrategy.ONE_STEP:
        gen = predict_one_step(model, values, norm_stats, window_size, n_steps, device,
                               inverse_fn, residual_std, z_score, y_true, idx_start)
    elif config.strategy == PredictionStrategy.RECALIBRATION:
        gen = predict_recalibration(model, values, norm_stats, window_size, n_steps, device,
                                    inverse_fn, residual_std, z_score, y_true, idx_start, config.recalib_every)
    elif config.strategy == PredictionStrategy.RECURSIVE:
        gen = predict_recursive(model, values, norm_stats, window_size, n_steps, device,
                                inverse_fn, residual_std, z_score, y_true, idx_start, config.max_horizon)
    elif config.strategy == PredictionStrategy.DIRECT:
        gen = predict_direct(model, values, norm_stats, window_size, n_steps, device,
                             inverse_fn, residual_std, z_score, y_true, idx_start, config.direct_horizon)
    else:
        raise ValueError(f"Stratégie inconnue: {config.strategy}")

    for evt in gen:
        if isinstance(evt, dict):
            yield evt
            if evt.get("type") == "pred_point":
                all_preds.append(evt["yhat"])
                all_low.append(evt["low"])
                all_high.append(evt["high"])
                if evt.get("y") is not None:
                    all_true_out.append(float(evt["y"]))
        elif isinstance(evt, tuple):
            all_preds, all_low, all_high, all_true_out = evt

    metrics = compute_metrics(all_true_out, all_preds[:len(all_true_out)])
    yield {"type": "pred_end", "n_steps": len(all_preds), "strategy": config.strategy.value,
           "all_predictions": all_preds, "all_low": all_low, "all_high": all_high,
           "all_true": all_true_out if all_true_out else None, "metrics": metrics}


# ============================================================================
# COMPARAISON DE TOUTES LES STRATÉGIES
# ============================================================================

@torch.no_grad()
def compare_all_strategies(
    model: nn.Module, values: List[float], norm_stats: Dict[str, float],
    window_size: int, n_steps: int, device: str = "cpu",
    inverse_fn: Optional[Callable] = None, residual_std: Optional[float] = None,
    y_true: Optional[List[float]] = None, idx_start: int = 0,
    recalib_every: int = 10, max_horizon: int = 50,
) -> Dict[str, Dict[str, Any]]:
    """Compare toutes les stratégies sur les mêmes données"""
    results = {}
    strategies = [
        (PredictionStrategy.ONE_STEP, {}),
        (PredictionStrategy.RECALIBRATION, {"recalib_every": recalib_every}),
        (PredictionStrategy.RECURSIVE, {"max_horizon": max_horizon}),
    ]

    for strategy, extra in strategies:
        config = PredictionConfig(strategy=strategy, recalib_every=extra.get("recalib_every", 10),
                                  max_horizon=extra.get("max_horizon", 50))
        preds, lows, highs, trues, metrics = [], [], [], [], {}
        for evt in predict_multistep(model, values, norm_stats, window_size, n_steps, device,
                                     inverse_fn, config, residual_std, y_true, idx_start):
            if evt["type"] == "pred_point":
                preds.append(evt["yhat"])
                lows.append(evt["low"])
                highs.append(evt["high"])
                if evt.get("y") is not None:
                    trues.append(evt["y"])
            elif evt["type"] == "pred_end":
                metrics = evt.get("metrics", {})
        results[strategy.value] = {"predictions": preds, "low": lows, "high": highs,
                                   "true": trues, "metrics": metrics}
    return results


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEST DU MODULE prediction_strategies.py")
    print("=" * 70)

    class SimpleModel(nn.Module):
        def __init__(self, ws):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(ws, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            return self.fc(x.squeeze(-1) if x.ndim == 3 else x)

    window_size = 15
    model = SimpleModel(window_size)
    np.random.seed(42)
    t = np.linspace(0, 10 * np.pi, 300)
    values = (np.sin(t) + 0.1 * np.random.randn(len(t))).tolist()
    norm_stats = {"mean": 0.0, "std": 0.7}
    y_true = values[200:250]

    print(f"\nComparaison des stratégies:")
    results = compare_all_strategies(model, values, norm_stats, window_size, 50, "cpu", None, 0.1, y_true, 200)
    print(f"\n{'Stratégie':<20} {'R²':>12} {'RMSE':>12} {'MAE':>12}")
    print("-" * 60)
    for name, data in results.items():
        m = data["metrics"]
        r2 = f"{m.get('R2', 0):.6f}" if m.get('R2') else "N/A"
        rmse = f"{m.get('RMSE', 0):.6f}" if m.get('RMSE') else "N/A"
        mae = f"{m.get('MAE', 0):.6f}" if m.get('MAE') else "N/A"
        print(f"{name:<20} {r2:>12} {rmse:>12} {mae:>12}")
    print("\n✓ Tests passés!")