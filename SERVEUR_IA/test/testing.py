import math
from typing import Callable, Optional, Dict, Any, List
import torch
from torch.utils.data import DataLoader, TensorDataset

@torch.no_grad()
def test_model(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str = "cpu",
    *,
    batch_size: Optional[int] = None,   # None = passe unique; sinon inférence en mini-batches
    inverse_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # ex: dénormalisation
) -> Dict[str, Any]:
    """
    Évalue un modèle PyTorch générique sur (X_test, y_test) et renvoie TOUTES les paires y / ŷ.

    Returns:
        {
          "n_test": N,
          "dims": D,
          "metrics": {
            "per_dim":  {"MSE":[...], "MAE":[...], "RMSE":[...], "R2":[...]},
            "overall_mean": {"MSE":float, "MAE":float, "RMSE":float, "R2":float|None}
          },
          "pairs": [ {"y":[...], "yhat":[...]}, ... ]   # N éléments, chacun liste de D valeurs
        }
    """
    model.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Harmoniser y_test en 2D: (N,) -> (N,1)
    if y_test.ndim == 1:
        y_test = y_test.unsqueeze(1)

    # --- Prédictions ---
    if batch_size is None:
        y_pred = model(X_test)
    else:
        dl = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        preds = []
        for xb, _ in dl:
            preds.append(model(xb))
        y_pred = torch.cat(preds, dim=0)

    # Assure 2D
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(1)

    # (Option) post-transformation (ex: inverse scaling) sur y et ŷ
    if inverse_fn is not None:
        y_true_eval = inverse_fn(y_test)
        y_pred_eval = inverse_fn(y_pred)
    else:
        y_true_eval = y_test
        y_pred_eval = y_pred

    # Tensors CPU plats
    y_true = y_true_eval.detach().cpu()
    y_pred = y_pred_eval.detach().cpu()

    N, D = y_true.shape[0], y_true.shape[1]

    # --- Métriques par dimension ---
    mse_list, mae_list, rmse_list, r2_list = [], [], [], []
    for d in range(D):
        t = y_true[:, d]
        p = y_pred[:, d]
        diff = p - t
        mse = float((diff.pow(2)).mean().item())
        mae = float(diff.abs().mean().item())
        rmse = float(math.sqrt(mse))
        var = float(((t - t.mean()).pow(2)).mean().item())
        r2 = float(1.0 - (mse / var)) if var > 0 else None

        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2 if r2 is not None else float("nan"))

    # Moyenne globale (ignore nan pour R2)
    overall_mse  = float(sum(mse_list) / D)
    overall_mae  = float(sum(mae_list) / D)
    overall_rmse = float(sum(rmse_list) / D)
    valid_r2 = [r for r in r2_list if not (r is None or math.isnan(r))]
    overall_r2 = float(sum(valid_r2) / len(valid_r2)) if valid_r2 else None

    # --- TOUTES les paires (y, ŷ) ---
    pairs: List[Dict[str, List[float]]] = [
        {"y": y_true[i].tolist(), "yhat": y_pred[i].tolist()}
        for i in range(N)
    ]

    return {
        "n_test": N,
        "dims": D,
        "metrics": {
            "per_dim": {
                "MSE": mse_list,
                "MAE": mae_list,
                "RMSE": rmse_list,
                "R2": [None if (math.isnan(r) or r is None) else float(r) for r in r2_list],
            },
            "overall_mean": {
                "MSE": overall_mse,
                "MAE": overall_mae,
                "RMSE": overall_rmse,
                "R2": overall_r2,
            },
        },
        "pairs": pairs,
    }