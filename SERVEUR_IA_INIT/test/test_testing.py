# ============================================================================
# testing.py - Test classique (teacher forcing) pour validation
# ============================================================================
import math
from typing import Callable, Optional, Dict, Any, List, Iterator
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


@torch.no_grad()
def test_model_validation(
    model: torch.nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 256,
    inverse_fn: Optional[Callable] = None,
    idx_start: int = 0,
) -> Iterator[Dict[str, Any]]:
    """
    Test en mode validation (teacher forcing) : on utilise les vraies valeurs en entrée.
    
    Args:
        model: Modèle PyTorch entraîné
        X_val: Tenseur d'entrée de validation
        y_val: Tenseur de sortie de validation
        device: cpu/cuda/mps
        batch_size: Taille des batches
        inverse_fn: Fonction de dénormalisation
        idx_start: Index de départ dans la série complète
    
    Yields:
        Événements de type val_start, val_pair, val_end
    """
    model.eval()
    model = model.to(device)
    
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    if y_val.ndim == 1:
        y_val = y_val.unsqueeze(1)
    
    N = X_val.shape[0]
    if batch_size is None or batch_size <= 0:
        batch_size = N
    
    dl = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Variance pour R²
    var = ((y_val - y_val.mean(dim=0, keepdim=True)).pow(2)).mean(dim=0)
    D = y_val.shape[1]
    
    yield {
        "type": "val_start",
        "n_points": N,
        "dims": D,
        "idx_start": idx_start
    }
    
    # Accumulateurs pour les métriques
    sqerr_sum = torch.zeros(D, device="cpu")
    abserr_sum = torch.zeros(D, device="cpu")
    
    # Stockage des prédictions/vraies valeurs
    all_y_true = []
    all_y_pred = []
    
    seen = 0
    for xb, yb in dl:
        yhat_b = model(xb.to(device))
        
        # Gestion sortie 3D (LSTM)
        if yhat_b.ndim == 3:
            yhat_b = yhat_b[:, -1, :]
        
        if yhat_b.ndim == 1:
            yhat_b = yhat_b.unsqueeze(1)
        
        # Dénormalisation si nécessaire
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
        
        # Stream chaque paire
        for i in range(yb_cpu.shape[0]):
            global_idx = idx_start + seen + i
            y_val_point = yb_cpu[i].tolist()
            yhat_val_point = yhat_cpu[i].tolist()
            
            all_y_true.append(y_val_point[0] if len(y_val_point) == 1 else y_val_point)
            all_y_pred.append(yhat_val_point[0] if len(yhat_val_point) == 1 else yhat_val_point)
            
            yield {
                "type": "val_pair",
                "idx": global_idx,
                "y": y_val_point,
                "yhat": yhat_val_point,
            }
        
        seen += yb_cpu.shape[0]
    
    # Métriques finales
    mse = (sqerr_sum / N).tolist()
    mae = (abserr_sum / N).tolist()
    rmse = [float(math.sqrt(m)) for m in mse]
    
    r2_list: List[Optional[float]] = []
    for d in range(D):
        v = float(var[d].item())
        r2_list.append(
            float(1.0 - ((sqerr_sum[d].item() / N) / v)) if v > 0 else None
        )
    
    valid_r2 = [r for r in r2_list if r is not None]
    overall = {
        "MSE": float(sum(mse) / D),
        "MAE": float(sum(mae) / D),
        "RMSE": float(sum(rmse) / D),
        "R2": (float(sum(valid_r2) / len(valid_r2)) if valid_r2 else None),
    }
    
    yield {
        "type": "val_end",
        "n_points": N,
        "idx_start": idx_start,
        "dims": D,
        "metrics": {
            "per_dim": {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2_list},
            "overall_mean": overall
        },
        "all_predictions": all_y_pred,
        "all_true": all_y_true,
    }


def compute_residual_std_from_validation(
    all_y_true: List[float],
    all_y_pred: List[float]
) -> float:
    """
    Calcule l'écart-type des résidus à partir des résultats de validation.
    Utilisé pour dimensionner le halo de probabilité.
    """
    y_true = np.array(all_y_true, dtype=float)
    y_pred = np.array(all_y_pred, dtype=float)
    residuals = y_true - y_pred
    return float(np.std(residuals))


# ============================================================================
# ANCIENNE INTERFACE (compatibilité)
# ============================================================================
@torch.no_grad()
def test_model(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 256,
    inverse_fn: Optional[Callable] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Interface de compatibilité avec l'ancien code.
    Appelle test_model_validation avec idx_start=0.
    """
    model.eval()
    model = model.to(device)
    
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
    
    var = ((y_test - y_test.mean(dim=0, keepdim=True)).pow(2)).mean(dim=0)
    D = y_test.shape[1]
    nbatches = len(dl)
    
    yield {
        "type": "test_start",
        "n_test": N,
        "dims": D,
        "n_batches": nbatches
    }
    
    sqerr_sum = torch.zeros(D, device="cpu")
    abserr_sum = torch.zeros(D, device="cpu")
    
    seen = 0
    for xb, yb in dl:
        yhat_b = model(xb.to(device))
        
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
        
        for i in range(yb_cpu.shape[0]):
            yield {
                "type": "test_pair",
                "y": yb_cpu[i].tolist(),
                "yhat": yhat_cpu[i].tolist(),
            }
        
        seen += yb_cpu.shape[0]
    
    mse = (sqerr_sum / N).tolist()
    mae = (abserr_sum / N).tolist()
    rmse = [float(math.sqrt(m)) for m in mse]
    
    r2_list: List[Optional[float]] = []
    for d in range(D):
        v = float(var[d].item())
        r2_list.append(
            float(1.0 - ((sqerr_sum[d].item() / N) / v)) if v > 0 else None
        )
    
    valid_r2 = [r for r in r2_list if r is not None]
    overall = {
        "MSE": float(sum(mse) / D),
        "MAE": float(sum(mae) / D),
        "RMSE": float(sum(rmse) / D),
        "R2": (float(sum(valid_r2) / len(valid_r2)) if valid_r2 else None),
    }
    
    yield {
        "type": "test_final",
        "n_test": N,
        "dims": D,
        "metrics": {
            "per_dim": {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2_list},
            "overall_mean": overall
        },
    }
    
    yield {"type": "fin_test", "done": 1}


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("Test de testing.py")
    
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
    X_val = torch.randn(50, 15)
    y_val = torch.randn(50, 1)
    
    print("\nTest validation:")
    for evt in test_model_validation(model, X_val, y_val, "cpu", idx_start=100):
        if evt["type"] == "val_start":
            print(f"  Début: {evt['n_points']} points")
        elif evt["type"] == "val_end":
            print(f"  Fin: MSE={evt['metrics']['overall_mean']['MSE']:.6f}")
            
            # Calculer residual_std
            residual_std = compute_residual_std_from_validation(
                evt["all_true"], evt["all_predictions"]
            )
            print(f"  Residual std: {residual_std:.6f}")
    
    print("\n✓ Test réussi!")