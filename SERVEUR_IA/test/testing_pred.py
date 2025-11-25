import math
from typing import Callable, Optional, Dict, Any, List, Iterator, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@torch.no_grad()
def test_model_pred(
    model: torch.nn.Module,
    X_test_or_data: Union[torch.Tensor, Dict],
    y_test_or_norm_stats: Union[torch.Tensor, Dict],
    device: str = "cpu",
    *,
    window_size: Optional[int] = None,
    pred_steps: Optional[int] = None,
    batch_size: Optional[int] = 256,
    inverse_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Test du modèle - COMPATIBLE AVEC ANCIEN ET NOUVEAU MODE
    
    MODE 1 (ANCIEN) - Évaluation classique sur test set:
        test_model(model, X_test, y_test, device, batch_size=256)
        
    MODE 2 (NOUVEAU) - Prédiction future:
        test_model(model, data, norm_stats, device, 
                   window_size=15, pred_steps=6)
    
    La fonction détecte automatiquement le mode en fonction des arguments.
    """
    
    # ========================================
    # DÉTECTION DU MODE
    # ========================================
    if isinstance(X_test_or_data, dict) and isinstance(y_test_or_norm_stats, dict):
        # MODE 2 : Nouveau (prédiction future)
        if window_size is None or pred_steps is None:
            raise ValueError(
                "Mode prédiction future : window_size et pred_steps requis"
            )
        return _test_model_prediction_mode(
            model=model,
            data=X_test_or_data,
            norm_stats=y_test_or_norm_stats,
            window_size=window_size,
            pred_steps=pred_steps,
            device=device,
            inverse_fn=inverse_fn
        )
    
    elif isinstance(X_test_or_data, torch.Tensor) and isinstance(y_test_or_norm_stats, torch.Tensor):
        # MODE 1 : Ancien (test set classique)
        return _test_model_classic_mode(
            model=model,
            X_test=X_test_or_data,
            y_test=y_test_or_norm_stats,
            device=device,
            batch_size=batch_size,
            inverse_fn=inverse_fn
        )
    
    else:
        raise TypeError(
            "Arguments invalides. Utilisez soit :\n"
            "  - Mode classique: test_model(model, X_test, y_test, device)\n"
            "  - Mode prédiction: test_model(model, data, norm_stats, device, "
            "window_size=15, pred_steps=6)"
        )


# ============================================================================
# MODE 1 : ANCIEN SYSTÈME (Test set classique)
# ============================================================================
def _test_model_classic_mode(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str,
    batch_size: int,
    inverse_fn: Optional[Callable]
) -> Iterator[Dict[str, Any]]:
    """Mode classique : évaluation sur test set"""
    
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    if y_test.ndim == 1:
        y_test = y_test.unsqueeze(1)

    N = X_test.shape[0]
    if batch_size is None or batch_size <= 0:
        batch_size = N

    dl = DataLoader(
        TensorDataset(X_test, y_test), 
        batch_size=batch_size, 
        shuffle=False
    )

    # Variance pour R²
    var = ((y_test - y_test.mean(dim=0, keepdim=True)).pow(2)).mean(dim=0)
    D = y_test.shape[1]
    nbatches = len(dl)
    
    yield {
        "type": "pred_start", 
        "n_test": N, 
        "dims": D, 
        "n_batches": nbatches
    }

    # Accumulateurs
    sqerr_sum = torch.zeros(D, device="cpu")
    abserr_sum = torch.zeros(D, device="cpu")

    seen = 0
    for bidx, (xb, yb) in enumerate(dl):
        yhat_b = model(xb.to(device))
        
        # Gestion sortie 3D (LSTM)
        if yhat_b.ndim == 3:
            yhat_b = yhat_b[:, -1, :]
        
        if yhat_b.ndim == 1:
            yhat_b = yhat_b.unsqueeze(1)

        if inverse_fn is not None:
            yb_eval = inverse_fn(yb)
            yhat_eval = inverse_fn(yhat_b)
        else:
            yb_eval = yb
            yhat_eval = yhat_b

        yb_cpu = yb_eval.detach().cpu()
        yhat_cpu = yhat_eval.detach().cpu()

        diff = yhat_cpu - yb_cpu
        
        sqerr_sum += diff.pow(2).sum(dim=0)
        abserr_sum += diff.abs().sum(dim=0)

        # Stream des paires
        for i in range(yb_cpu.shape[0]):
            yield {
                "type": "pred_pair",
                "y": yb_cpu[i].tolist(),
                "yhat": yhat_cpu[i].tolist(),
            }

        seen += yb_cpu.shape[0]
        yield {
            "type": "pred_progress", 
            "done": int(seen), 
            "total": int(N)
        }

    # Métriques finales
    mse = (sqerr_sum / N).tolist()
    mae = (abserr_sum / N).tolist()
    rmse = [float(math.sqrt(m)) for m in mse]
    
    r2_list: List[Optional[float]] = []
    for d in range(D):
        v = float(var[d].item())
        r2_list.append(
            float(1.0 - ((sqerr_sum[d].item()/N) / v)) if v > 0 else None
        )

    valid_r2 = [r for r in r2_list if r is not None]
    overall = {
        "MSE": float(sum(mse)/D),
        "MAE": float(sum(mae)/D),
        "RMSE": float(sum(rmse)/D),
        "R2": (float(sum(valid_r2)/len(valid_r2)) if valid_r2 else None),
    }

    yield {
        "type": "pred_final",
        "n_test": N,
        "dims": D,
        "metrics": {
            "per_dim": {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2_list},
            "overall_mean": overall
        },
    }
    
    yield {"type": "fin_pred", "done": 1}


# ============================================================================
# MODE 2 : NOUVEAU SYSTÈME (Prédiction future)
# ============================================================================
def _test_model_prediction_mode(
    model: torch.nn.Module,
    data: Dict,
    norm_stats: Dict,
    window_size: int,
    pred_steps: int,   # on le garde pour la signature, mais on ne s'en sert pas vraiment ici
    device: str,
    inverse_fn: Optional[Callable]
) -> Iterator[Dict[str, Any]]:
    """
    Mode prédiction future :
    - On considère 80% de la série comme historique (train)
    - On prédit autorégressivement les 20% restants
    - On renvoie des listes all_true / all_predictions de TOUTE longueur de la série
      (avec des None là où il n'y a pas de prédiction), pour que l'UI puisse tracer
      correctement et placer la séparation 80%.
    """
    model.eval()
    model = model.to(device)

    model_type = type(model).__name__.upper()

    # -----------------------------
    # Données brutes & split
    # -----------------------------
    raw_values = data["values"]
    total_data_points = len(raw_values)

    if total_data_points <= window_size + 1:
        raise ValueError(
            f"Série trop courte ({total_data_points}) pour window_size={window_size}"
        )

    # Ratio de split (à adapter si tu le passes en paramètre)
    split_ratio = 0.8
    split_idx = int(total_data_points * split_ratio)

    if split_idx <= window_size:
        raise ValueError(
            f"Index de split {split_idx} <= window_size={window_size} ; "
            f"augmente la taille de la série ou diminue la fenêtre."
        )

    # Nombre de points qu'on va prédire = 20% de fin
    n_predictions = total_data_points - split_idx

    yield {
        "type": "pred_start",
        "n_test": total_data_points,   # longueur des listes renvoyées
        "dims": 1,
        "n_batches": 1,
        "model_type": model_type,
        "window_size": window_size,
        "pred_steps": n_predictions,
        "split_index": split_idx,      # info utile si tu veux côté UI
    }

    # -----------------------------
    # Nettoyage & normalisation
    # -----------------------------
    mean = norm_stats["mean"]
    std = norm_stats["std"]

    # Série nettoyée : None -> NaN, cast en float
    clean_values = []
    for v in raw_values:
        clean_values.append(np.nan if v is None else float(v))
    all_values = np.array(clean_values, dtype=float)

    # Variance pour R² sur la partie "test" uniquement
    var_y = np.nanvar(all_values[split_idx:])

    # Listes complètes pour renvoi à l'UI
    all_true_full = all_values.copy()                 # toute la série réelle
    all_pred_full: List[Optional[float]] = [None] * total_data_points  # prédictions au bon endroit

    # Listes restreintes pour les métriques (uniquement zone prédite)
    all_y_true_metrics: List[float] = []
    all_y_pred_metrics: List[float] = []

    # Fenêtre initiale : les window_size points avant split_idx
    initial_window = all_values[split_idx - window_size: split_idx].reshape(-1, 1)

    # Comble les NaN dans cette fenêtre
    if np.isnan(initial_window).any():
        initial_window = (
            pd.Series(initial_window.flatten())
            .fillna(method="ffill")
            .fillna(method="bfill")
            .values.reshape(-1, 1)
        )

    # Normalisation
    current_window = (initial_window - mean) / std

    # -----------------------------
    # Boucle de prédiction autorégressive sur les 20% de fin
    # -----------------------------
    for pred_idx in range(n_predictions):
        global_idx = split_idx + pred_idx  # index dans la série complète

        # Préparation de l'entrée selon le type de modèle
        if model_type == "MLP":
            x_input = torch.FloatTensor(current_window.flatten()).unsqueeze(0)
        elif model_type == "LSTM":
            x_input = torch.FloatTensor(current_window).unsqueeze(0)
        elif model_type == "CNN":
            x_input = torch.FloatTensor(current_window).transpose(0, 1).unsqueeze(0)
        else:
            x_input = torch.FloatTensor(current_window).unsqueeze(0)

        x_input = x_input.to(device)

        # ----- Prédiction -----
        with torch.no_grad():
            pred = model(x_input)
            if pred.ndim == 3:
                pred = pred[:, -1, :]
            pred = pred.cpu().numpy()
            pred = float(pred.flatten()[0]) if pred.ndim > 0 else float(pred.item())

        # ----- Dénormalisation -----
        if inverse_fn is not None:
            pred_tensor = torch.tensor([[pred]], dtype=torch.float32)
            pred_denorm = float(inverse_fn(pred_tensor).item())
        else:
            pred_denorm = float(pred * std + mean)

        # ----- Vraie valeur (pour métriques) -----
        raw_val = raw_values[global_idx]
        y_true = np.nan if raw_val is None else float(raw_val)

        all_y_true_metrics.append(y_true)
        all_y_pred_metrics.append(pred_denorm)

        # Mise à jour de la série de prédictions complète
        all_pred_full[global_idx] = pred_denorm

        # Envoi pour l'UI (animation, etc.)
        y_to_send = None if np.isnan(y_true) else float(y_true)
        yield {
            "type": "pred_pair",
            "y": [y_to_send],
            "yhat": [pred_denorm],
            "pred_idx": pred_idx,
            "global_idx": global_idx,
            "step": pred_idx + 1,
        }

        # Autorégressif : on réinjecte la prédiction (normalisée) dans la fenêtre
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1, 0] = pred

    # -----------------------------
    # Métriques finales
    # -----------------------------
    all_y_true_metrics_arr = np.array(all_y_true_metrics, dtype=float)
    all_y_pred_metrics_arr = np.array(all_y_pred_metrics, dtype=float)

    mask = ~np.isnan(all_y_true_metrics_arr)
    valid_y_true = all_y_true_metrics_arr[mask]
    valid_y_pred = all_y_pred_metrics_arr[mask]

    N = len(valid_y_true)
    if N > 0:
        mse = float(np.mean((valid_y_true - valid_y_pred) ** 2))
        mae = float(np.mean(np.abs(valid_y_true - valid_y_pred)))
        rmse = float(np.sqrt(mse))
        var_y_valid = np.nanvar(valid_y_true)
        r2 = float(1.0 - mse / var_y_valid) if var_y_valid > 1e-10 else None
    else:
        mse = mae = rmse = 0.0
        r2 = None

    per_dim = {"MSE": [mse], "MAE": [mae], "RMSE": [rmse], "R2": [r2]}
    overall = {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}

    # Séries complètes pour le JSON (float ou None, jamais NaN)
    safe_true_full = [None if np.isnan(x) else float(x) for x in all_true_full]

    yield {
        "type": "pred_final",
        "n_test": total_data_points,        # même longueur que les listes renvoyées
        "dims": 1,
        "metrics": {
            "per_dim": per_dim,
            "overall_mean": overall,
        },
        "model_type": model_type,
        "all_predictions": all_pred_full,   # longueur = série complète, avec None avant split
        "all_true": safe_true_full,         # série réelle complète
        "split_index": split_idx,
    }

    yield {"type": "fin_pred", "done": 1}


def _predict_autoregressive(
    model: torch.nn.Module,
    initial_window: np.ndarray,
    n_steps: int,
    device: str,
    model_type: str
) -> np.ndarray:
    """Prédiction autorégressive"""
    predictions = []
    current_window = initial_window.copy()
    
    for _ in range(n_steps):
        if model_type == 'MLP':
            x = torch.FloatTensor(current_window.flatten()).unsqueeze(0)
        elif model_type == 'LSTM':
            x = torch.FloatTensor(current_window).unsqueeze(0)
        elif model_type == 'CNN':
            x = torch.FloatTensor(current_window).transpose(0, 1).unsqueeze(0)
        else:
            x = torch.FloatTensor(current_window).unsqueeze(0)
        
        x = x.to(device)
        
        pred = model(x)
        if pred.ndim == 3:
            pred = pred[:, -1, :]
        pred = pred.squeeze().cpu().numpy()
        
        if pred.size > 1:
            pred = pred[0]
        
        predictions.append(pred)
        
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1, 0] = pred
    
    return np.array(predictions).reshape(-1, 1)


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST DES DEUX MODES")
    print("="*60)
    
    # Modèle simple
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(15, 1)
        def forward(self, x):
            if x.ndim == 3:
                x = x[:, -1, :]
            if x.ndim == 2 and x.shape[1] > 1:
                x = x.flatten(1)
            return self.fc(x)
    
    model = DummyModel()
    
    # # MODE 1 : Ancien
    # print("\n" + "-"*60)
    # print("MODE 1 : PRED classique")
    # print("-"*60)
    # X_test = torch.randn(100, 15)
    # y_test = torch.randn(100, 1)
    
    # for update in test_model(model, X_test, y_test, 'cpu', batch_size=32):
    #     if update['type'] == 'test_start':
    #         print(f"✓ Démarré: {update['n_test']} tests")
    #     elif update['type'] == 'test_final':
    #         print(f"✓ Terminé: MSE={update['metrics']['overall_mean']['MSE']:.6f}")
    
    # MODE 2 : Nouveau
    print("\n" + "-"*60)
    print("MODE 2 : Prédiction future")
    print("-"*60)
    data = {
        'timestamps': [(datetime.now() + timedelta(hours=i)).isoformat() for i in range(100)],
        'values': np.sin(np.linspace(0, 4*np.pi, 100)).tolist()
    }
    norm_stats = {'mean': 0.0, 'std': 0.7}
    
    for update in test_model(model, data, norm_stats, 'cpu', 
                            window_size=15, pred_steps=6):
        if update['type'] == 'pred_start':
            print(f"✓ Démarré: {update['n_test']} prédictions")
        elif update['type'] == 'pred_final':
            print(f"✓ Terminé: MSE={update['metrics']['overall_mean']['MSE']:.6f}")
    
    print("\n" + "="*60)
    print("✓ LES DEUX MODES FONCTIONNENT")
    print("="*60)