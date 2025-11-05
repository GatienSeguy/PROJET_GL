import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =======================
# Sanitation & scaling
# =======================
def sanitize_series(vals: np.ndarray) -> np.ndarray:
    vals = vals.astype(np.float32).reshape(-1)
    bad = ~np.isfinite(vals)
    if bad.any():
        vals[bad] = np.nan
    if np.isnan(vals).any():
        idx = np.arange(len(vals))
        good = ~np.isnan(vals)
        if not good.any():
            raise ValueError("Toute la série est NaN/Inf.")
        vals = np.interp(idx, idx[good], vals[good]).astype(np.float32)
    return vals


def robust_standardize(vals: np.ndarray, eps: float = 1e-8):
    m = float(np.mean(vals))
    s = float(np.std(vals))
    if not np.isfinite(s) or s < eps:
        s = 1.0
    vals_norm = (vals - m) / s
    return vals_norm.astype(np.float32), m, s


# =======================
# Device (CUDA prioritaire)
# =======================
def get_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =======================
# Dataset fenêtre glissante
# =======================
class WindowDataset(Dataset):
    """Transforme une série 1D en paires (fenêtre, cible_next_step). X: [seq_len,1], y:[1]"""
    def _init_(self, series_1d: np.ndarray, window: int):
        assert series_1d.ndim == 1, "La série doit être 1D."
        self.x, self.y = [], []
        for i in range(len(series_1d) - window):
            self.x.append(series_1d[i:i + window])
            self.y.append(series_1d[i + window])  # 1-step ahead
        self.x = np.array(self.x, dtype=np.float32)                 # [N, window]
        self.y = np.array(self.y, dtype=np.float32).reshape(-1, 1)  # [N, 1]

    def _len_(self):
        return len(self.x)

    def _getitem_(self, idx):
        # retourne tensors CPU ; pin_memory + non_blocking gèrent les transferts
        x = torch.from_numpy(self.x[idx][:, None])  # [window,1]
        y = torch.from_numpy(self.y[idx])           # [1]
        return x, y


# =======================
# Modèle RNN (LSTM / GRU)
# NOTE: on garde les poids RNN en float32
# =======================
class RNNRegressor(nn.Module):
    def _init_(self, rnn_type="lstm", input_size=1, hidden_size=128, num_layers=2, dropout=0.0):
        super()._init_()
        rnn_type = rnn_type.lower()
        self.rnn_type = rnn_type
        RNN = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, state=None):
        # IMPORTANT: on force le RNN en f32 (stable et portable)
        rnn_dtype = self.rnn.weight_ih_l0.dtype  # normalement f32
        if x.dtype != rnn_dtype:
            x = x.to(rnn_dtype)
        if state is not None:
            if isinstance(state, tuple):  # LSTM
                state = tuple(s.to(rnn_dtype) for s in state)
            else:  # GRU
                state = state.to(rnn_dtype)
        out, state = self.rnn(x, state) if state is not None else self.rnn(x)
        last = out[:, -1, :]  # [B,H]
        y = self.fc(last)     # [B,1]
        return y, state


# =======================
# Entraînement (AMP CUDA + GradScaler + Early Stopping)
# =======================
def train(
    model,
    train_loader,
    device,
    epochs=1000,
    lr=1e-3,
    log_every=50,
    early_patience=30,
    amp_dtype: str = "bf16",  # "bf16" | "fp16" | "none"
    grad_clip: float = 1.0,
    compile_model: bool = True,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Réglages CUDA
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # accélère matmul en f32 sur Ampere+
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # AMP dtype & scaler
    use_amp = (device.type == "cuda" and amp_dtype in {"bf16", "fp16"})
    amp_dtype_torch = None
    if use_amp:
        amp_dtype_torch = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == "fp16"))

    # torch.compile si possible (pas Python 3.14+)
    if compile_model and device.type == "cuda" and sys.version_info < (3, 14):
        try:
            model = torch.compile(model)  # mode par défaut ("max-autotune" sur Torch 2.5+)
            print("[INFO] torch.compile activé (CUDA).")
        except Exception as e:
            print(f"[WARN] torch.compile indisponible/échoué: {e}")

    best = float("inf")
    patience = 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            # transferts asynchrones
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype_torch):
                    yhat, _ = model(xb)
                    loss = loss_fn(yhat, yb)
                scaler.scale(loss).backward()
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                yhat, _ = model(xb)
                loss = loss_fn(yhat, yb)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                opt.step()

            tr_loss += loss.item() * xb.size(0)

        tr_loss /= max(1, len(train_loader.dataset))
        if (ep % log_every) == 0 or ep == 1:
            print(f"[Epoch {ep:04d}] train={tr_loss:.6f}")

        # Early stopping (sur train faute de val)
        if tr_loss < best - 1e-7:
            best = tr_loss
            patience = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= early_patience:
                print(f"[INFO] Early stop @ epoch {ep} (best train={best:.6f})")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


# =======================
# Prédiction auto-régressive — pré-allouée (zéro concat)
# =======================
@torch.no_grad()
def predict_autoreg_fast(model: RNNRegressor, last_window: np.ndarray, horizon: int, device):
    """
    last_window: np.array [window] (normalisée)
    horizon: int
    Retour: np.array [H] (normalisée)
    """
    model.eval()
    x_win = torch.from_numpy(last_window.astype(np.float32)).to(device, non_blocking=True)[None, :, None]  # [1,T,1]
    out_pred = torch.empty(horizon, dtype=torch.float32, device=device)

    y, state = model(x_win)        # [1,1]
    out_pred[0] = y.view(-1)[0]

    for t in range(1, horizon):
        x_step = y.view(1, 1, 1)
        y, state = model(x_step, state=state)
        out_pred[t] = y.view(-1)[0]

    return out_pred.detach().cpu().numpy().astype(np.float32)


# =======================
# Chargement des données
# =======================
def load_series_from_json(path: Path):
    data = json.loads(Path(path).read_text())
    if "values" in data:
        return np.array(data["values"], dtype=np.float32).reshape(-1)
    for key in ["y", "close", "price", "data", "series"]:
        if key in data:
            return np.array(data[key], dtype=np.float32).reshape(-1)
    if isinstance(data, dict):
        for k, v in data.items():
            if k != "timestamps" and isinstance(v, list):
                try:
                    return np.array(v, dtype=np.float32).reshape(-1)
                except Exception:
                    pass
    raise ValueError("Impossible d'identifier la série.")


# =======================
# Split 99% / 1% (temporel)
# =======================
def split_99_1(arr):
    n = len(arr)
    n_test = max(1, int(math.ceil(0.01 * n)))
    n_train = n - n_test
    return arr[:n_train], arr[n_train:]


# =======================
# Main
# =======================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument("--data", type=str, default="./Datas/EURO.json")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--plot_path", type=str, default="eurolstm_99_1.png")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--early_patience", type=int, default=30)
    p.add_argument("--rnn", type=str, default="lstm", choices=["lstm", "gru"], help="Choix du RNN de base")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"], help="AMP sur CUDA")
    p.add_argument("--no_compile", action="store_true", help="Désactive torch.compile")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.force_cpu)
    if sys.version_info >= (3, 14):
        print("[INFO] Python 3.14+ détecté : torch.compile ignoré.")
    print("Device:", device)

    # 1) charge & nettoie & standardise
    series_raw = load_series_from_json(Path(args.data))
    series_clean = sanitize_series(series_raw)
    series_scaled, mean_, std_ = robust_standardize(series_clean)

    print(f"[INFO] Serie: len={len(series_scaled)}, min={np.min(series_scaled):.3g}, max={np.max(series_scaled):.3g}")

    # 2) split 99% / 1%
    train_arr, test_arr = split_99_1(series_scaled)
    print(f"[INFO] Split -> train={len(train_arr)}, test={len(test_arr)}")

    # garde-fous
    if len(train_arr) <= args.window:
        raise ValueError(f"Fenêtre trop grande pour le train ({args.window} > {len(train_arr)}). Baissez --window.")
    if not np.isfinite(series_scaled).all():
        raise ValueError("La série normalisée contient des NaN/Inf.")

    # 3) dataset/loader — optimisations CUDA: pin_memory + prefetch
    train_ds = WindowDataset(train_arr, window=args.window)
    loader_kwargs = dict(
        batch_size=args.batch,
        shuffle=True,
        drop_last=False,
        num_workers=max(0, args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    # Enlever les clés None pour éviter les warnings
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}
    train_loader = DataLoader(train_ds, **loader_kwargs)

    # 4) modèle
    model = RNNRegressor(
        rnn_type=args.rnn,
        input_size=1,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    )

    # 5) entraînement (AMP + early stopping + compile éventuel)
    model = train(
        model, train_loader, device=device,
        epochs=args.epochs, lr=args.lr, log_every=args.log_every,
        early_patience=args.early_patience, amp_dtype=args.amp_dtype,
        compile_model=(not args.no_compile),
    )

    # 6) vraie prédiction: horizon = taille du test
    H = len(test_arr)
    last_window = train_arr[-args.window:]  # normalisée
    preds_norm = predict_autoreg_fast(model, last_window=last_window, horizon=H, device=device)

    # 7) dénormalisation
    preds = preds_norm * std_ + mean_
    truth = test_arr * std_ + mean_
    real_train = train_arr * std_ + mean_

    # 8) in-sample fit (teacher forcing 1-step)
    model.eval()
    with torch.no_grad():
        arr = torch.from_numpy(train_arr.astype(np.float32)).to(device, non_blocking=True)
        T = arr.numel()
        n_fit = T - args.window
        fitted_buf = torch.empty(n_fit, dtype=torch.float32, device=device)
        for i in range(args.window, T):
            xw = arr[i - args.window:i].view(1, -1, 1)
            yhat, _ = model(xw)
            fitted_buf[i - args.window] = yhat.view(-1)[0]
        fitted_train = fitted_buf.detach().cpu().numpy().astype(np.float32)
    fitted_train = fitted_train * std_ + mean_

    # alignement pour le plot (une seule courbe "apprise + prédiction")
    full_real = np.concatenate([real_train, truth], axis=0)
    full_pred = np.concatenate([
        np.full(len(real_train) - len(fitted_train), np.nan, dtype=np.float32),
        fitted_train,
        preds
    ], axis=0)

    # 9) métriques test
    mae = float(np.mean(np.abs(preds - truth)))
    mape = float(np.mean(np.abs((truth - preds) / np.maximum(1e-8, np.abs(truth))))) * 100.0
    print(f"[TEST] MAE={mae:.6g}  MAPE={mape:.3f}%  (horizon={H})")

    # 10) plot
    plt.figure(figsize=(11, 4))
    plt.plot(full_real, label="Série réelle")
    plt.plot(full_pred, label="Série apprise (train) + prédiction (test)")
    plt.axvline(x=len(real_train), linestyle="--", label="Début zone test")
    plt.title(f"Split 99%/1% — Test MAE={mae:.4g}, MAPE={mape:.2f}%")
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig(args.plot_path, dpi=150)
    print(f"[PLOT] Figure sauvegardée -> {args.plot_path}")
    try:
        plt.show()
    except Exception:
        pass


if _name_ == "_main_":
    main()