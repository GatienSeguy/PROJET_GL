# trains/training_MLP.py
import inspect
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..models.optim import make_loss, make_optimizer
from ..models.model_MLP import MLP
import time
import math

def _build_mlp_safely(in_dim: int, out_dim: int, **kwargs):
    """
    Crée un MLP en détectant la signature réelle et en mappant les alias :
    - hidden_size -> hidden_dim | width
    - nb_couches  -> n_layers | depth | layers
    - dropout_rate-> dropout | p_dropout
    - activation  -> act | activation_name
    - use_batchnorm -> batchnorm | bn | use_bn
    - in_dim -> input_dim | in_features
    - out_dim -> output_dim | out_features
    """
    sig = inspect.signature(MLP.__init__)
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
    put(kwargs.get("hidden_size", 128), "hidden_size", "hidden_dim", "width")
    put(kwargs.get("nb_couches", 2), "nb_couches", "n_layers", "depth", "layers")
    put(kwargs.get("dropout_rate", 0.0), "dropout_rate", "dropout", "p_dropout")
    put(kwargs.get("activation", "relu"), "activation", "act", "activation_name")
    put(kwargs.get("use_batchnorm", False), "use_batchnorm", "batchnorm", "bn", "use_bn")

    # --- Tout autre kw explicite présent dans la signature ---
    for k, v in kwargs.items():
        if k in params:
            resolved[k] = v

    # (Optionnel) Debug :
    print("[MLP kwargs résolus] ->", {
        k: resolved[k]
        for k in resolved
        if k not in ("in_dim", "input_dim", "in_features", "out_dim", "output_dim", "out_features")
    })

    return MLP(**resolved)


def train_MLP(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    # --- ARCHITECTURE ---
    hidden_size: int = 128,
    nb_couches: int = 2,
    dropout_rate: float = 0.0,
    activation: str = "relu",
    use_batchnorm: bool = False,

    # --- LOSS / OPTIM ---
    loss_name: str = "mse",
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,

    # --- TRAIN ---
    batch_size: int = 64,
    epochs: int = 10,
    device: str = "cpu",
):
    if y.ndim == 1:
        y = y.unsqueeze(1)

    # Optimisation: pin_memory pour CUDA
    pin_memory = (str(device) == "cuda" or "cuda" in str(device))
    loader = DataLoader(
        TensorDataset(X, y), 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        pin_memory=pin_memory
    )

    model = _build_mlp_safely(
        in_dim=X.shape[1],
        out_dim=y.shape[1],
        hidden_size=hidden_size,
        nb_couches=nb_couches,
        dropout_rate=dropout_rate,
        activation=activation,
        use_batchnorm=use_batchnorm,
    ).to(device)

    criterion = make_loss({"name": loss_name})
    optimizer = make_optimizer(
        model,
        {"name": optimizer_name, "lr": learning_rate, "weight_decay": weight_decay}
    )

    last_avg = 0.0
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total, n = 0.0, 0
        model.train()
        
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)  # Plus efficace
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            # Protection contre NaN/Inf
            if math.isfinite(loss_val):
                total += loss_val * xb.size(0)
            n += xb.size(0)
        
        last_avg = total / max(1, n) if n > 0 else 0.0
        epoch_duration = time.time() - epoch_start
        
        # Protection contre division par zéro et valeurs infinies
        epoch_s = 1.0 / epoch_duration if epoch_duration > 0.001 else 1000.0
        if not math.isfinite(epoch_s):
            epoch_s = 1000.0
        if not math.isfinite(last_avg):
            last_avg = 0.0
            
        yield {"type": "epoch", "epochs": epoch, "avg_loss": last_avg, "epoch_s": epoch_s}
        print(f"[MLP {epoch:03d}/{epochs}] loss={last_avg:.6f}")

    final_loss = last_avg if math.isfinite(last_avg) else 0.0
    yield {"done": True, "final_loss": final_loss}

    return model


