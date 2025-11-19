# prediction.py

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from model_LSTM import LSTM


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_time_series_json(path: str):
    """
    Charge :
    {
        "timestamps": [...],
        "values": [...]
    }
    Gère les NaN par interpolation linéaire.
    Retourne (timestamps_str, timestamps_dt, values)
    """
    with open(path, "r") as f:
        data = json.load(f)

    timestamps_str = data["timestamps"]
    values = np.array(data["values"], dtype=float)

    # --- Gestion des NaN (même logique que dans train.py) ---
    mask = np.isnan(values)
    if mask.any():
        idx = np.arange(len(values))
        valid_idx = idx[~mask]
        valid_vals = values[~mask]

        if len(valid_idx) == 0:
            raise ValueError("Toutes les valeurs du dataset sont NaN, impossible de prédire.")

        values[mask] = np.interp(idx[mask], valid_idx, valid_vals)
        print(f"[WARN] NaN détectés dans {path} -> remplis par interpolation linéaire.")

    # Tentative de conversion en datetime
    timestamps_dt = None
    try:
        timestamps_dt = [datetime.fromisoformat(t) for t in timestamps_str]
    except Exception:
        pass

    return timestamps_str, timestamps_dt, values


def create_teacher_forcing_predictions(
    model: torch.nn.Module,
    series_norm_t: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """
    Prédictions 'fitted' sur la série fournie (teacher forcing).

    ⚠ Ici on construit une prédiction pour CHAQUE point du train :
    - on fait glisser une fenêtre de longueur seq_len,
    - pour chaque fenêtre, on récupère TOUTE la sortie du LSTM (seq_len pas de temps),
    - on aligne ces sorties sur la série et on fait la moyenne quand plusieurs fenêtres
      recouvrent le même instant.

    series_norm_t : Tensor (T_train, 1)
    Retourne un vecteur numpy (T_train,) sans utiliser les 20% de fin.
    """
    model.eval()
    T_train = series_norm_t.shape[0]

    # On va accumuler les prédictions et compter combien de fois chaque indice est prédit
    sum_pred = np.zeros(T_train, dtype=float)
    count_pred = np.zeros(T_train, dtype=float)

    with torch.no_grad():
        for i in range(T_train - seq_len + 1):
            window = series_norm_t[i : i + seq_len].unsqueeze(0).to(device)  # (1, seq_len, 1)
            out = model(window)  # (1, seq_len, 1)
            out_np = out[0, :, 0].cpu().numpy()  # (seq_len,)

            # On aligne les prédictions de la fenêtre sur la série globale
            for j in range(seq_len):
                idx = i + j
                sum_pred[idx] += out_np[j]
                count_pred[idx] += 1.0

    # Moyenne des prédictions quand plusieurs fenêtres recouvrent le même point
    fitted = np.full(T_train, np.nan, dtype=float)
    valid = count_pred > 0
    fitted[valid] = sum_pred[valid] / count_pred[valid]

    return fitted


def create_future_forecast(
    model: torch.nn.Module,
    series_norm_t: torch.Tensor,
    train_size: int,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """
    Prédiction future sur les 20% de la série, en mode autoregressif :
    - on part des seq_len derniers points de la partie train,
    - à chaque étape on prédit le prochain point, puis on le réinjecte.
    """
    model.eval()
    T = series_norm_t.shape[0]
    test_size = T - train_size
    if test_size <= 0:
        raise ValueError("Pas de partie test (20%) détectée.")

    if train_size < seq_len:
        raise ValueError("train_size < seq_len, impossible de faire la prédiction future.")

    with torch.no_grad():
        # fenêtre initiale : derniers seq_len points du train
        window = series_norm_t[train_size - seq_len : train_size].clone().to(device)  # (seq_len, 1)
        forecast = np.zeros(test_size, dtype=float)

        for k in range(test_size):
            inp = window.unsqueeze(0)  # (1, seq_len, 1)
            out = model(inp)          # (1, seq_len, 1)
            next_val = out[0, -1, 0]  # scalaire tensor

            forecast[k] = next_val.item()

            # fenêtre glissante
            window = torch.cat([window[1:], next_val.view(1, 1)], dim=0)

    return forecast


def main(
    data_path: str,
    checkpoint_path: str = "SERVEUR_IA/models/lstm_checkpoint.pt",
    output_fig: str = "prediction_lstm.png",
):
    device = get_device()
    # device = "cpu"  # Force CPU for compatibility
    print(f"[INFO] Device utilisé : {device}")
    
      # Force CPU for compatibility
    # 1) Chargement des données JSON (+ NaN fix)
    timestamps_str, timestamps_dt, series = load_time_series_json(data_path)  # (T,)
    T = len(series)
    print(f"[INFO] Longueur totale de la série : {T}")

    # 2) Chargement du checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    in_dim = checkpoint["in_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    out_dim = checkpoint["out_dim"]
    num_layers = checkpoint["num_layers"]
    seq_len = checkpoint["seq_len"]
    train_mean = checkpoint["train_mean"]
    train_std = checkpoint["train_std"]

    # 3) Reconstruction du modèle (même signature que dans train.py)
    model = LSTM(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        nb_couches=num_layers,
        batch_first=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4) Normalisation (avec mean/std DU TRAIN, stockés dans le checkpoint)
    series_norm = (series - train_mean) / train_std
    series_norm_t = torch.tensor(series_norm, dtype=torch.float32).unsqueeze(-1)  # (T,1)

    # 5) Définition train/test (mêmes 80/20 que dans train.py)
    train_size = int(0.8 * T)
    test_size = T - train_size
    print(f"[INFO] Train : {train_size} points, Test : {test_size} points")

    # --- 5a) Fonction apprise (fitted) UNIQUEMENT sur la partie TRAIN ---
    series_norm_train_t = series_norm_t[:train_size]  # (train_size, 1)

    fitted_norm_train = create_teacher_forcing_predictions(
        model=model,
        series_norm_t=series_norm_train_t,
        seq_len=seq_len,
        device=device,
    )  # (train_size,)

    # On étend à la taille totale, mais on laisse NaN sur la partie test
    fitted_norm_full = np.full(T, np.nan, dtype=float)
    fitted_norm_full[:train_size] = fitted_norm_train

    # --- 5b) Prédiction future auto-régressive sur les 20% de fin ---
    forecast_norm = create_future_forecast(
        model=model,
        series_norm_t=series_norm_t,
        train_size=train_size,
        seq_len=seq_len,
        device=device,
    )  # (test_size,)

    forecast_norm_full = np.full(T, np.nan, dtype=float)
    forecast_norm_full[train_size:] = forecast_norm

    # 6) Dénormalisation
    series_true = series
    fitted = fitted_norm_full * train_std + train_mean
    forecast = forecast_norm_full * train_std + train_mean

    # 7) Plot final
    if timestamps_dt is not None:
        x = timestamps_dt
        x_boundary = timestamps_dt[train_size - 1]
        use_dates = True
    else:
        x = np.arange(T)
        x_boundary = train_size - 1
        use_dates = False

    plt.figure(figsize=(12, 6))
    plt.plot(x, series_true, label="Série réelle", linewidth=1.5)
    plt.plot(x, fitted, label="Fonction apprise (train seul)", linewidth=1.2)
    plt.plot(x, forecast, label="Prédiction future (20% fin)", linewidth=1.5, linestyle="--")

    # Ligne verticale frontière train/test
    plt.axvline(x_boundary, color="k", linestyle=":", label="Frontière 80/20")

    plt.xlabel("Temps" if use_dates else "Indice")
    plt.ylabel("Valeur")
    plt.title("Série temporelle : réel vs fonction apprise (train) vs prédiction future")
    plt.legend()
    plt.grid(True)

    if use_dates:
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_fig, dpi=150)
    plt.show()

    print(f"[INFO] Figure sauvegardée dans : {output_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédiction LSTM (train 80% + future 20%) sur JSON.")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Chemin vers le fichier JSON de série temporelle.")
    parser.add_argument("--checkpoint_path", type=str,
                        default="SERVEUR_IA/models/lstm_checkpoint.pt",
                        help="Checkpoint entraîné.")
    parser.add_argument("--output_fig", type=str, default="prediction_lstm.png",
                        help="Fichier image de sortie.")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        output_fig=args.output_fig,
    )