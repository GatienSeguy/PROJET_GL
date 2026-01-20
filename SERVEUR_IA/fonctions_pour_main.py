# ====================================
# IMPORTs
# ====================================
from typing import List, Optional, Tuple, Callable
import numpy as np
import torch
from datetime import datetime
import json

# ====================================
# CONSTRUCTION DES TENSEURS SUPERVISÉS
# ====================================
def build_supervised_tensors(
    values: List[Optional[float]],
    window_len: int = 15,
    horizon: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construit (X, y) à partir d'une liste 'values' possiblement avec des None.
    X: [N, window_len], y: [N, 1]
    Stratégie: on ne garde que les fenêtres 100% valides (sans None) et une cible valide.
    """
    X_list, y_list = [], []
    n = len(values)
    if n < window_len + horizon:
        return torch.empty(0, window_len), torch.empty(0, 1)
 
    for i in range(0, n - window_len - horizon + 1):
        seq = values[i : i + window_len]
        tgt = values[i + window_len + horizon - 1]
        if any(v is None for v in seq) or tgt is None:
            continue
        X_list.append(seq)
        y_list.append([tgt])

    if not X_list:
        return torch.empty(0, window_len), torch.empty(0, 1)

    X = torch.tensor(np.array(X_list, dtype=np.float32))
    y = torch.tensor(np.array(y_list, dtype=np.float32))
    return X, y


# ====================================
# FILTRES DE DATES
# ====================================
def _parse_any_datetime(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.strptime(s, "%Y-%m-%d")


# ====================================
# FILTRAGE DES SÉRIES PAR DATES
# ====================================
def filter_series_by_dates(timestamps, values, dates):
    if not dates or len(dates) < 2 or dates[0] is None or dates[1] is None:
        return timestamps, values
    start = _parse_any_datetime(dates[0])
    end = _parse_any_datetime(dates[1])
    if start > end:
        start, end = end, start
    ts_out, val_out = [], []
    for t, v in zip(timestamps, values):
        if start <= t <= end:
            ts_out.append(t)
            val_out.append(v)
    return ts_out, val_out


# ====================================
# CONSTRUCTION DES TENSEURS AVEC STEP
# ====================================
def build_supervised_tensors_with_step(values, window_len=1, horizon=1, step=1):
    if step <= 0:
        step = 1
    n = len(values)
    if n == 0:
        return torch.empty(0, window_len), torch.empty(0, 1)
    max_start = n - (window_len + horizon - 1) * step
    if max_start <= 0:
        return torch.empty(0, window_len), torch.empty(0, 1)

    X_list, y_list = [], []
    for i in range(0, max_start):
        seq_idx = [i + k * step for k in range(window_len)]
        tgt_idx = i + (window_len + horizon - 1) * step
        seq = [values[j] for j in seq_idx]
        tgt = values[tgt_idx]
        if any(v is None for v in seq) or tgt is None:
            continue
        X_list.append(seq)
        y_list.append([tgt])

    if not X_list:
        return torch.empty(0, window_len), torch.empty(0, 1)

    X = torch.tensor(np.array(X_list, dtype=np.float32))
    y = torch.tensor(np.array(y_list, dtype=np.float32))
    return X, y


# ====================================
# SPLIT TRAIN/TEST (ancien, conservé)
# ====================================
def split_train_test(X, y, portion_train):
    p = portion_train if (portion_train is not None and 0.0 < portion_train < 1.0) else 0.8
    n = X.shape[0]
    if n == 0:
        return X, y, X, y
    n_train = max(1, min(n - 1, int(n * p))) if n >= 2 else n
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


# ====================================
# NOUVEAU : SPLIT TRAIN/VAL/TEST (80/10/10)
# ====================================
def split_train_val_test(
    X: torch.Tensor, 
    y: torch.Tensor, 
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Découpe (X, y) en 3 parties : train, validation, test.
    
    Args:
        X: Tenseur d'entrée (N, ...)
        y: Tenseur de sortie (N, ...)
        train_ratio: Proportion pour l'entraînement (défaut 0.8)
        val_ratio: Proportion pour la validation (défaut 0.1)
        test_ratio: Proportion pour le test prédictif (défaut 0.1)
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, info_dict
        
        info_dict contient:
            - n_train, n_val, n_test
            - idx_val_start, idx_test_start
    """
    # Normalisation des ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total
    
    n = X.shape[0]
    if n < 3:
        # Pas assez de données, tout va en train
        return X, y, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), {
            "n_train": n, "n_val": 0, "n_test": 0,
            "idx_val_start": n, "idx_test_start": n
        }
    
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val
    
    # Assurer qu'on a au moins 1 point par partie si possible
    if n_test < 1 and n > 2:
        n_test = 1
        n_val = max(1, n - n_train - n_test)
    
    idx_val_start = n_train
    idx_test_start = n_train + n_val
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    info = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "idx_val_start": idx_val_start,
        "idx_test_start": idx_test_start
    }
    
    return X_train, y_train, X_val, y_val, X_test, y_test, info


# ====================================
# SERVER-SENT EVENTS FORMAT
# ====================================
def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# ====================================
# NORMALISATION / DÉNORMALISATION
# ====================================
def normalize_data(data: torch.Tensor, method: str = "standardization") -> Tuple[torch.Tensor, dict]:
    """
    Normalise les données et retourne les paramètres de normalisation.
    """
    if method == "standardization":
        mean = data.mean()
        std = data.std()
        if std == 0:
            std = 1.0
        normalized = (data - mean) / std
        params = {"method": "standardization", "mean": mean.item(), "std": std.item()}
    
    elif method == "minmax":
        min_val = data.min()
        max_val = data.max()
        if max_val - min_val == 0:
            normalized = data
        else:
            normalized = (data - min_val) / (max_val - min_val)
        params = {"method": "minmax", "min": min_val.item(), "max": max_val.item()}
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    return normalized, params


def denormalize_data(data: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Dénormalise les données selon les paramètres fournis.
    """
    method = params.get("method")
    
    if method == "standardization":
        mean = params["mean"]
        std = params["std"]
        return data * std + mean
    
    elif method == "minmax":
        min_val = params["min"]
        max_val = params["max"]
        return data * (max_val - min_val) + min_val
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")


# ====================================
# CRÉATION FONCTION INVERSE
# ====================================
def create_inverse_function(params: dict) -> Callable:
    """
    Crée une fonction de dénormalisation à partir des paramètres.
    """
    def inverse_fn(data: torch.Tensor) -> torch.Tensor:
        return denormalize_data(data, params)
    
    return inverse_fn


# ====================================
# NOUVEAU : CALCUL VARIANCE RÉSIDUELLE
# ====================================
def compute_residual_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule l'écart-type des résidus (pour le halo de probabilité).
    """
    residuals = y_true - y_pred
    return float(np.std(residuals))