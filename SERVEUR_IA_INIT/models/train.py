# train.py
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_LSTM import LSTM  # Ton modèle


# -------------------------------------------------------------------
# 1. Utilitaires
# -------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_time_series_json(path: str):
    """
    Charge une série temporelle 1D depuis un fichier JSON :
    {
        "timestamps": [...],
        "values": [...]
    }

    GÈRE LES NaN PAR INTERPOLATION LINÉAIRE.

    On retourne (timestamps, values) avec values sans NaN.
    """
    with open(path, "r") as f:
        data = json.load(f)

    timestamps = data["timestamps"]
    values = np.array(data["values"], dtype=float)  # (T,)

    # --- Gestion des NaN ---
    mask = np.isnan(values)
    if mask.any():
        idx = np.arange(len(values))
        valid_idx = idx[~mask]
        valid_vals = values[~mask]

        if len(valid_idx) == 0:
            raise ValueError("Toutes les valeurs sont NaN dans le dataset, impossible d'entraîner.")

        # Interpolation linéaire pour remplir les NaN
        values[mask] = np.interp(idx[mask], valid_idx, valid_vals)
        print(f"[WARN] NaN détectés dans {path} -> remplis par interpolation linéaire.")

    return timestamps, values


def create_sequences(series: np.ndarray, seq_len: int):
    """
    Crée des paires (input_seq, target_seq) pour un LSTM seq2seq :
    - input_seq : [x_t, ..., x_{t+seq_len-1}]
    - target_seq : [x_{t+1}, ..., x_{t+seq_len}]

    series : (T,) ou (T,1)
    Retourne :
      X : (N_samples, seq_len, 1)
      Y : (N_samples, seq_len, 1)
    """
    if series.ndim == 1:
        series = series[:, None]  # (T,1)

    T = series.shape[0]
    xs, ys = [], []

    for i in range(T - seq_len):
        x = series[i : i + seq_len]
        y = series[i + 1 : i + seq_len + 1]
        xs.append(x)
        ys.append(y)

    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    return X, Y


# -------------------------------------------------------------------
# 2. Entraînement
# -------------------------------------------------------------------

def train(
    data_path: str,
    checkpoint_path: str = "lstm_checkpoint.pt",
    seq_len: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.0,   # pas utilisé dans le modèle, mais on le garde dans la signature
    batch_size: int = 32,
    lr: float = 1e-3,
    num_epochs: int = 50,
):

    device = get_device()
    # device= "cpu"  # Force CPU for compatibility
    print(f"[INFO] Device utilisé : {device}")

    # 1) Chargement de la série (JSON) + fix des NaN
    timestamps, series = load_time_series_json(data_path)  # timestamps: list, series: (T,)
    T = len(series)
    print(f"[INFO] Longueur totale de la série : {T}")

    # 2) Split 80% / 20% en chronologique
    train_size = int(0.8 * T)
    test_size = T - train_size
    print(f"[INFO] Train : {train_size} points, Test (jamais vus à l'entraînement) : {test_size} points")

    # ⚠️ On NE GARDE POUR L'ENTRAÎNEMENT QUE LES 80% DE DÉBUT
    series_train = series[:train_size]  # (train_size,)

    # 3) Normalisation (sur le TRAIN uniquement, sans regarder le 20% de fin)
    train_mean = series_train.mean()
    train_std = series_train.std() if series_train.std() > 1e-8 else 1.0

    series_train_norm = (series_train - train_mean) / train_std  # (train_size,)

    # 4) Création des séquences pour l'entraînement
    if train_size <= seq_len:
        raise ValueError(
            f"La partie train ({train_size}) est trop courte par rapport à seq_len ({seq_len})."
        )

    X_train, Y_train = create_sequences(series_train_norm, seq_len=seq_len)
    # X_train, Y_train : (N_samples, seq_len, 1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, Y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 5) Création du modèle
    in_dim = 1
    out_dim = 1

    # ⚠️ Adapté à TA classe LSTM : nb_couches au lieu de num_layers
    model = LSTM(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        nb_couches=num_layers,
        batch_first=True,
    ).to(device)

    # Si ta classe a num_parameters()
    if hasattr(model, "num_parameters"):
        print(f"[INFO] Nombre de paramètres : {model.num_parameters()}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6) Boucle d'entraînement
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device)  # (B, T, 1)
            yb = yb.to(device)  # (B, T, 1)

            optimizer.zero_grad()
            out = model(xb)  # (B, T, 1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_ds)
        print(f"Epoch {epoch:03d}/{num_epochs} - Loss: {epoch_loss:.6f}")

    # 7) Sauvegarde du checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "num_layers": num_layers,   # pour recharger -> nb_couches
        "dropout": dropout,
        "seq_len": seq_len,
        "train_mean": float(train_mean),
        "train_std": float(train_std),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Checkpoint sauvegardé dans : {checkpoint_path}")


# -------------------------------------------------------------------
# 3. main
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement LSTM sur série temporelle (80/20) JSON.")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Chemin vers le fichier JSON de série temporelle.")
    parser.add_argument("--checkpoint_path", type=str, default="lstm_checkpoint.pt",
                        help="Fichier de sortie du modèle.")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
    )