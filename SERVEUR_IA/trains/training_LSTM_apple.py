# trains/training_LSTM.py - VERSION OPTIMISÉE M3 Pro
import inspect
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..models.optim import make_loss, make_optimizer
from ..models.model_LSTM import LSTM 


def get_optimal_device():
    """
    Détecte automatiquement le meilleur device disponible.
    Pour M3 Pro: privilégie MPS (Metal Performance Shaders)
    """
    if torch.backends.mps.is_available():
        print("🚀 Utilisation du GPU Apple Silicon (MPS)")
        return "mps"
    elif torch.cuda.is_available():
        print("🚀 Utilisation du GPU CUDA")
        return "cuda"
    else:
        print("⚠️ Utilisation du CPU (considérez l'activation de MPS)")
        return "cpu"


def _build_lstm_safely(in_dim: int, out_dim: int, **kwargs):
    """
    Crée un LSTM en détectant la signature réelle et en mappant les alias :

    - in_dim           -> input_dim | in_features
    - out_dim          -> output_dim | out_features
    - hidden_size       <- hidden_size | width
    - nb_couches       <- n_layers | nb_couches | depth | layers | num_layers
    - bidirectional    <- bi | bidir
    - batch_first      <- batchfirst
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
    hidden_size: int = 128,
    nb_couches: int = 2,
    bidirectional: bool = False,
    batch_first: bool = True,

    # --- LOSS / OPTIM ---
    loss_name: str = "mse",
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,

    # --- TRAIN ---
    batch_size: int = 64,
    epochs: int = 10,
    device: str = None,  # Auto-détection si None
    
    # --- OPTIMISATIONS M3 Pro ---
    num_workers: int = 4,  # Parallélisation du DataLoader (profite des 12 cœurs)
    pin_memory: bool = True,  # Accélère les transferts CPU->GPU
    persistent_workers: bool = True,  # Garde les workers actifs
    prefetch_factor: int = 2,  # Précharge les batches
    
    # --- COMPORTEMENT SORTIE ---
    take_last_if_needed: bool = True,
    verbose: bool = True,
):
    """
    Entraîne un modèle LSTM sur des tenseurs déjà supervisés.
    VERSION OPTIMISÉE pour Apple Silicon M3 Pro.

    Attendus :
    - X: (B, T, in_dim) si batch_first=True (défaut)
    - y: (B, T, out_dim) pour seq->seq
         (B, out_dim)     pour seq->one

    Retourne (model, last_avg) et yield un dict toutes les k époques.
    """
    
    # Auto-détection du device optimal
    if device is None:
        device = get_optimal_device()
    
    if verbose:
        print(f"📊 X.shape: {X.shape}, y.shape: {y.shape}, batch_first: {batch_first}")
        print(f"🎯 Device: {device}, Batch size: {batch_size}, Epochs: {epochs}")
    
    # Sanity checks
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

    # Détermination de out_dim
    if y.ndim == 3:
        if batch_first:
            assert y.shape[0] == X.shape[0] and y.shape[1] == X.shape[1]
            out_dim = y.shape[2]
        else:
            assert y.shape[1] == X.shape[1] and y.shape[0] == X.shape[0]
            out_dim = y.shape[2]
        seq_to_seq = True
    elif y.ndim == 2:
        assert y.shape[0] == (X.shape[0] if batch_first else X.shape[1])
        out_dim = y.shape[1]
        seq_to_seq = False
    else:
        raise ValueError("y doit être de rang 2 (seq->one) ou 3 (seq->seq).")

    # DataLoader OPTIMISÉ pour M3 Pro
    # Note: num_workers > 0 profite des 12 cœurs CPU du M3 Pro
    # Pin memory n'est supporté que sur CUDA, pas sur MPS
    use_pin_memory = pin_memory and device == "cuda"
    
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    # Modèle
    model = _build_lstm_safely(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_size=hidden_size,
        nb_couches=nb_couches,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(device)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Modèle: {total_params:,} paramètres")

    # Loss / Optim
    criterion = make_loss({"name": loss_name})
    optimizer = make_optimizer(
        model,
        {"name": optimizer_name, "lr": learning_rate, "weight_decay": weight_decay}
    )
    
    # Note: AMP n'est pas encore stable sur MPS (PyTorch 2.x)
    # On désactive pour l'instant
    use_amp = False
    if use_amp and verbose:
        print("⚡ Activation de l'AMP (Automatic Mixed Precision)")
    
    # Boucle d'entraînement
    last_avg = None
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Plus efficace que zero_grad()
            
            # Forward pass (avec AMP si disponible)
            if use_amp:
                with torch.autocast(device_type='cpu'):  # MPS utilise cpu comme device_type
                    pred = model(xb)
                    
                    if seq_to_seq:
                        loss = criterion(pred, yb)
                    else:
                        if take_last_if_needed:
                            last = pred[:, -1, :] if batch_first else pred[-1, :, :]
                            loss = criterion(last, yb)
                        else:
                            if batch_first:
                                yb_rep = yb.unsqueeze(1).expand(-1, pred.shape[1], -1)
                            else:
                                yb_rep = yb.unsqueeze(0).expand(pred.shape[0], -1, -1)
                            loss = criterion(pred, yb_rep)
            else:
                pred = model(xb)
                
                if seq_to_seq:
                    loss = criterion(pred, yb)
                else:
                    if take_last_if_needed:
                        last = pred[:, -1, :] if batch_first else pred[-1, :, :]
                        loss = criterion(last, yb)
                    else:
                        if batch_first:
                            yb_rep = yb.unsqueeze(1).expand(-1, pred.shape[1], -1)
                        else:
                            yb_rep = yb.unsqueeze(0).expand(pred.shape[0], -1, -1)
                        loss = criterion(pred, yb_rep)

            loss.backward()
            
            # Gradient clipping (optionnel mais recommandé pour LSTM)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            bs = xb.size(0) if batch_first else xb.size(1)
            total += loss.item() * bs
            n += bs

        last_avg = total / max(1, n)

        k = 1
        if epoch % k == 0:
            yield {"epochs": epoch, "avg_loss": float(last_avg)}

        if verbose:
            print(f"[LSTM {epoch:03d}/{epochs}] loss={last_avg:.6f}")

    yield {"done": True, "final_loss": float(last_avg)}
    return model, last_avg