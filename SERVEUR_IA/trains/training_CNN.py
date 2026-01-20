# trains/training_CNN.py
import inspect
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..models.optim import make_loss, make_optimizer
from ..models.model_CNN import CNN 
import time
import math

def _build_cnn_safely(in_dim: int, out_dim: int, **kwargs):
    """
    Crée un CNN1D en détectant la signature réelle et en mappant les alias
    """
    sig = inspect.signature(CNN.__init__)
    params = set(sig.parameters.keys())
    resolved = {}

    # in/out dims
    for cand in ("in_dim", "input_dim", "in_channels"):
        if cand in params:
            resolved[cand] = in_dim
            break
    for cand in ("out_dim", "output_dim", "out_channels"):
        if cand in params:
            resolved[cand] = out_dim
            break

    def put(value, *name_options):
        for name in name_options:
            if name in params:
                resolved[name] = value
                return

    # Mapping des paramètres
    put(kwargs.get("hidden_size", 64), "hidden_dim", "hidden_size", "width")
    put(kwargs.get("nb_couches", 2), "nb_couches", "n_layers", "depth", "layers")
    put(kwargs.get("activation", "relu"), "activation", "act", "activation_name")
    put(kwargs.get("use_batchnorm", False), "use_batchnorm", "batchnorm", "bn", "use_bn")
    put(kwargs.get("kernel_size", 3), "kernel_size", "ksize")
    put(kwargs.get("stride", 1), "stride")
    put(kwargs.get("padding", 1), "padding")

    # Tout autre kw explicite
    for k, v in kwargs.items():
        if k in params:
            resolved[k] = v

    return CNN(**resolved)


def train_CNN(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    # --- ARCHI ---
    hidden_size: int = 64,
    nb_couches: int = 2,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
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
    """
    Entraîne un CNN1D.
    """
    # Reshape si nécessaire
    if X.ndim == 2:
        X = X.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    pin_memory = (str(device) == "cuda" or "cuda" in str(device))
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

    in_channels = X.shape[1]
    out_dim = y.shape[1]

    model = _build_cnn_safely(
        in_dim=in_channels,
        out_dim=out_dim,
        hidden_size=hidden_size,
        nb_couches=nb_couches,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        activation=activation,
        use_batchnorm=use_batchnorm,
    ).to(device)

    criterion = make_loss({"name": loss_name})
    optimizer = make_optimizer(model, {"name": optimizer_name, "lr": learning_rate, "weight_decay": weight_decay})

    last_avg = 0.0
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total, n = 0.0, 0
        
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            pred = model(xb)
            if pred.ndim == 3:
                pred = pred.mean(dim=2)
            if pred.ndim == 3:
                pred = pred.squeeze(1)
            
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            if math.isfinite(loss_val):
                total += loss_val * xb.size(0)
            n += xb.size(0)

        last_avg = total / max(1, n)
        epoch_duration = time.time() - epoch_start

        epoch_s = 1.0 / epoch_duration if epoch_duration > 0.001 else 1000.0
        if not math.isfinite(epoch_s):
            epoch_s = 1000.0
        if not math.isfinite(last_avg):
            last_avg = 0.0

        yield {"type": "epoch", "epochs": epoch, "avg_loss": last_avg, "epoch_s": epoch_s}
        print(f"[CNN {epoch:03d}/{epochs}] loss={last_avg:.6f}")

    final_loss = last_avg if math.isfinite(last_avg) else 0.0
    yield {"done": True, "final_loss": final_loss}

    return model