import math
from typing import Callable, Optional, Dict, Any, List, Iterator
import torch
from torch.utils.data import DataLoader, TensorDataset

@torch.no_grad()
def test_model(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str = "cpu",
    *,
    batch_size: Optional[int] = 256,
    inverse_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Stream l'évaluation: yield y/ŷ par paire + un 'test_final' avec les métriques.
    """
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    if y_test.ndim == 1:
        y_test = y_test.unsqueeze(1)

    N = X_test.shape[0]
    if batch_size is None or batch_size <= 0:
        batch_size = N

    dl = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # var(y) pour R2 (par dim)
    var = ((y_test - y_test.mean(dim=0, keepdim=True)).pow(2)).mean(dim=0)  # (D,)
    D = y_test.shape[1]
    nbatches = len(dl)
    yield {"type": "test_start", "n_test": N, "dims": D, "n_batches": nbatches}

    # CHANGEMENT: accumulateurs scalaires
    sqerr_sum = torch.zeros(D, device="cpu")
    abserr_sum = torch.zeros(D, device="cpu")

    seen = 0
    for bidx, (xb, yb) in enumerate(dl):
        yhat_b = model(xb.to(device))
        
        # AJOUT: Gérer les sorties 3D du LSTM (B, T, D) -> prendre dernière timestep
        if yhat_b.ndim == 3:
            yhat_b = yhat_b[:, -1, :]  # (B, T, D) -> (B, D)
        
        if yhat_b.ndim == 1:
            yhat_b = yhat_b.unsqueeze(1)

        if inverse_fn is not None:
            yb_eval = inverse_fn(yb)
            yhat_eval = inverse_fn(yhat_b)
        else:
            yb_eval = yb
            yhat_eval = yhat_b

        yb_cpu   = yb_eval.detach().cpu()
        yhat_cpu = yhat_eval.detach().cpu()

        diff = yhat_cpu - yb_cpu           # (B, D)
        
        # CHANGEMENT: sum sur tout (batch + dims) puis répartir par dim
        sqerr_sum  += diff.pow(2).sum(dim=0)   # Somme sur batch, garde dims
        abserr_sum += diff.abs().sum(dim=0)     # Somme sur batch, garde dims

        # ──── STREAMER y / yhat PAIRE PAR PAIRE ────
        for i in range(yb_cpu.shape[0]):
            yield {
                "type": "test_pair",
                "y": yb_cpu[i].tolist(),
                "yhat": yhat_cpu[i].tolist(),
            }

        seen += yb_cpu.shape[0]
        yield {"type": "test_progress", "done": int(seen), "total": int(N)}

    # ──── métriques finales ────
    mse  = (sqerr_sum / N).tolist()
    mae  = (abserr_sum / N).tolist()
    rmse = [float(math.sqrt(m)) for m in mse]
    r2_list: List[Optional[float]] = []
    for d in range(D):
        v = float(var[d].item())
        r2_list.append(float(1.0 - ((sqerr_sum[d].item()/N) / v)) if v > 0 else None)

    valid_r2 = [r for r in r2_list if r is not None]
    overall = {
        "MSE": float(sum(mse)/D),
        "MAE": float(sum(mae)/D),
        "RMSE": float(sum(rmse)/D),
        "R2": (float(sum(valid_r2)/len(valid_r2)) if valid_r2 else None),
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