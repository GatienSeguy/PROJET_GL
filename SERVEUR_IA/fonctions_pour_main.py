from typing import List, Optional, Tuple
import numpy as np
import torch
from typing import Optional, Tuple, Literal, List
from pydantic import BaseModel, Field, conint, confloat
from datetime import date,datetime
import json


def build_supervised_tensors(
    values: List[Optional[float]],
    window_len: int = 1,
    horizon: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construit (X, y) à partir d'une liste 'values' possiblement avec des None.
    X: [N, window_len], y: [N, 1]
    Stratégie: on ne garde que les fenêtres 100% valides (sans None) et une cible valide.
    """
    clean_vals = values  # on travaille direct, mais on filtre fenêtre par fenêtre
    X_list, y_list = [], []
    n = len(clean_vals)
    if n < window_len + horizon:
        return torch.empty(0, window_len), torch.empty(0, 1)
 
    for i in range(0, n - window_len - horizon + 1):
        seq = clean_vals[i : i + window_len]
        tgt = clean_vals[i + window_len + horizon - 1]
        # fenêtre valide ?
        if any(v is None for v in seq) or tgt is None:
            continue
        X_list.append(seq)
        y_list.append([tgt])

    if not X_list:
        return torch.empty(0, window_len), torch.empty(0, 1)

    X = torch.tensor(np.array(X_list, dtype=np.float32))
    y = torch.tensor(np.array(y_list, dtype=np.float32))
    return X, y







def _parse_any_datetime(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.strptime(s, "%Y-%m-%d")

def filter_series_by_dates(timestamps, values, dates):
    if not dates or len(dates) < 2 or dates[0] is None or dates[1] is None:
        return timestamps, values
    start = _parse_any_datetime(dates[0])
    end   = _parse_any_datetime(dates[1])
    if start > end:
        start, end = end, start
    ts_out, val_out = [], []
    for t, v in zip(timestamps, values):
        if start <= t <= end:
            ts_out.append(t)
            val_out.append(v)
    return ts_out, val_out

def build_supervised_tensors_with_step(values, window_len=1, horizon=1, step=1):
    if step <= 0: step = 1
    n = len(values)
    if n == 0:
        return torch.empty(0, window_len), torch.empty(0, 1)
    max_start = n - (window_len + horizon - 1)*step
    if max_start <= 0:
        return torch.empty(0, window_len), torch.empty(0, 1)

    X_list, y_list = [], []
    for i in range(0, max_start):
        seq_idx = [i + k*step for k in range(window_len)]
        tgt_idx = i + (window_len + horizon - 1)*step
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

def split_train_test(X, y, portion_train):
    p = portion_train if (portion_train is not None and 0.0 < portion_train < 1.0) else 0.8
    n = X.shape[0]
    if n == 0:
        return X, y, X, y
    n_train = max(1, min(n-1, int(n * p))) if n >= 2 else n
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]



def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"
