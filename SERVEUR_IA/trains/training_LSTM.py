# trains/training_LSTM.py
import inspect
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..models.optim import make_loss, make_optimizer
from ..models.model_LSTM import LSTM 
import time


def _build_lstm_safely(in_dim: int, out_dim: int, **kwargs):
    """
    Crée un LSTM en détectant la signature réelle et en mappant les alias :

    - in_dim           -> input_dim | in_features
    - out_dim          -> output_dim | out_features
    - hidden_size       <- hidden_size | width
    - nb_couches       <- n_layers | nb_couches | depth | layers | num_layers (attention: ici c'est le nb de LSTM empilés)
    - bidirectional    <- bi | bidir
    - batch_first      <- batchfirst

    Tout autre kw présent dans la signature officielle est copié tel quel.
    """
    sig = inspect.signature(LSTM.__init__)
    params = set(sig.parameters.keys())

    resolved = {}

    # --- in/out dims ---
    for cand in ("in_dim", "input_dim", "in_features"):
        if cand in params:
            resolved[cand] = in_dim
            break
    for cand in ("out_dim", "output_dim", "out_features"):
        if cand in params:
            resolved[cand] = out_dim
            break

    def put(value, *name_options):
        for name in name_options:
            if name in params:
                resolved[name] = value
                return

    # --- mapping des alias ---
    batch_first_val = kwargs.get("batch_first", True)
    if not isinstance(batch_first_val, bool):
        batch_first_val = True if str(batch_first_val).lower() == "true" else False
   
    if "batch_first" in kwargs and not isinstance(kwargs["batch_first"], bool):
        del kwargs["batch_first"]
    put(batch_first_val, "batch_first", "batchfirst")
    

    put(kwargs.get("nb_couches", 2), "num_layers", "n_layers", "nb_couches", "depth", "layers")
    put(kwargs.get("hidden_size", 128), "hidden_size", "width", "hidden_dim")
    put(kwargs.get("bidirectional", False), "bidirectional", "bi", "bidir")

   
    for k, v in kwargs.items():
        if k in params:
            resolved[k] = v

    return LSTM(**resolved)





def train_LSTM(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    # --- ARCHI ---
    hidden_size: int = 128,          # hidden_size, width
    nb_couches: int = 2,            #  n_layers, nb_couches, depth, layers, num_layers
    bidirectional: bool = False,    #  bi, bidir
    batch_first: bool = True,       #   batchfirst

    # --- LOSS / OPTIM ---
    loss_name: str = "mse",
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,

    # --- TRAIN ---
    batch_size: int = 64,
    epochs: int = 10,
    device: str = "cpu",

    # --- COMPORTEMENT SORTIE ---
    # Si y est 2D (B, out_dim) et X 3D (B, T, in_dim), on utilise la dernière sortie temporelle.
    # Si y est 3D (B, T, out_dim), on entraîne en seq->seq.
    take_last_if_needed: bool = True,
):
    """
    Entraîne un modèle LSTM sur des tenseurs déjà supervisés.

    Attendus :
    - X: (B, T, in_dim) si batch_first=True (défaut)
    - y: (B, T, out_dim) pour seq->seq
         (B, out_dim)     pour seq->one (on prend la dernière étape temporelle de la sortie)

    Retourne (model, last_avg) et yield un dict toutes les k époques comme train_MLP.
    """

    print(f"DEBUG - X.shape: {X.shape}, y.shape: {y.shape}, batch_first: {batch_first}")
    
    # Sanity checks rapides
    if batch_first:
        if X.ndim == 2:
            X = X.unsqueeze(-1)
        elif X.ndim == 1:
            X = X.unsqueeze(0).unsqueeze(-1)
        assert X.ndim == 3, "X doit être (B, T, in_dim) avec batch_first=True"
        B, T, in_dim = X.shape
    else:
        if X.ndim == 2:
            X = X.unsqueeze(-1)
        elif X.ndim == 1:
            X = X.unsqueeze(0).unsqueeze(-1)
        assert X.ndim == 3, "X doit être (T, B, in_dim) avec batch_first=False"
        T, B, in_dim = X.shape

    # Détermination de out_dim selon la forme de y
    if y.ndim == 3:
        # seq->seq
        if batch_first:
            assert y.shape[0] == X.shape[0] and y.shape[1] == X.shape[1], \
                "Incohérence B/T entre X et y (seq->seq)."
            out_dim = y.shape[2]
        else:
            assert y.shape[1] == X.shape[1] and y.shape[0] == X.shape[0], \
                "Incohérence T/B entre X et y (seq->seq)."
            out_dim = y.shape[2]
        seq_to_seq = True
    elif y.ndim == 2:
        # seq->one
        print(f"DEBUG ASSERT - batch_first={batch_first}, X.shape={X.shape}, y.shape={y.shape}")
        print(f"Comparaison: y.shape[0]={y.shape[0]} vs X.shape[0 if batch_first else 1]={X.shape[0] if batch_first else X.shape[1]}")
        
        assert y.shape[0] == (X.shape[0] if batch_first else X.shape[1]), \
            "Incohérence B entre X et y (seq->one)."
        out_dim = y.shape[1]
        seq_to_seq = False
    else:
        raise ValueError("y doit être de rang 2 (seq->one) ou 3 (seq->seq).")

    # DataLoader
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    # Modèle
    model = _build_lstm_safely(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_size=hidden_size,
        nb_couches=nb_couches,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(device)

    # Loss / Optim
    criterion = make_loss({"name": loss_name})
    optimizer = make_optimizer(
        model,
        {"name": optimizer_name, "lr": learning_rate, "weight_decay": weight_decay}
    )

    # Boucle d'entraînement
    last_avg = None
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total, n = 0.0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)  # (B, T, out_dim) si batch_first

            if seq_to_seq:
                # yb: (B, T, out_dim) → on compare sur tous les pas de temps
                loss = criterion(pred, yb)
            else:
                # seq->one : yb: (B, out_dim)
                if take_last_if_needed:
                    # on prend la dernière sortie temporelle
                    last = pred[:, -1, :] if batch_first else pred[-1, :, :]
                    loss = criterion(last, yb)
                else:
                    # fallback (peu utilisé) : moyenne des pertes sur T vs y répété
                    if batch_first:
                        # répéter yb sur T pour matcher (B, T, out_dim)
                        yb_rep = yb.unsqueeze(1).expand(-1, pred.shape[1], -1)
                    else:
                        yb_rep = yb.unsqueeze(0).expand(pred.shape[0], -1, -1)
                    loss = criterion(pred, yb_rep)

            loss.backward()
            optimizer.step()

            bs = xb.size(0) if batch_first else xb.size(1)
            total += loss.item() * bs
            n += bs

        last_avg = total / max(1, n)
        epoch_duration = time.time() - epoch_start

        k = 1
        if epoch % k == 0:
            yield {"epochs": epoch, "avg_loss": float(last_avg), "epoch_s" : epoch_duration}

        print(f"[LSTM {epoch:03d}/{epochs}] loss={last_avg:.6f}")

    yield {"done": True, "final_loss": float(last_avg)}
    return model