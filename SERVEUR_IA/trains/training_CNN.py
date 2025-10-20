# trains/training_cnn.py
import inspect
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..models.optim import make_loss, make_optimizer
from ..models.model_CNN import CNN 


def _build_cnn_safely(in_dim: int, out_dim: int, **kwargs):
    """
    Crée un CNN1D en détectant la signature réelle et en mappant les alias :
    - hidden_size -> hidden_dim | width
    - nb_couches  -> n_layers | depth | layers
    - kernel_size -> ksize
    - stride
    - padding
    - activation
    - use_batchnorm
    - in_dim / out_dim
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

    # map aliases from kwargs
    put(kwargs.get("hidden_size", 32), "hidden_dim", "hidden_size", "width")
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

    # debug
    print("[CNN kwargs résolus] ->", {k: resolved[k] for k in resolved if k not in ("in_dim", "out_dim")})

    return CNN(**resolved)


def train_CNN(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    # --- ARCHI ---
    hidden_size: int = 32,
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
    weight_decay: float = 0.0, #Nouveau

    # --- TRAIN ---
    batch_size: int = 64,
    epochs: int = 10,
    device: str = "cpu",
):
    if y.ndim == 1:
        y = y.unsqueeze(1)

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    print("ZEBI AVANT MODEL")
    model = _build_cnn_safely(
        in_dim=X.shape[1],
        out_dim=y.shape[1],
        hidden_size=hidden_size,
        nb_couches=nb_couches,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        activation=activation,
        use_batchnorm=use_batchnorm,
    ).to(device)

    print(" AVANT LOSS")
    criterion = make_loss({"name": loss_name})
    print(" AVANT OPTI")
    optimizer = make_optimizer(model, {"name": optimizer_name, "lr": learning_rate, "weight_decay": weight_decay})

    print("APRES OPTIM")
    last_avg = None
    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            xb = xb.unsqueeze(1)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        last_avg = total / max(1, n)

        # Progression tous les k epochs
        k = 100
        if epoch % k == 0:
            yield {"epoch": epoch, "avg_loss": float(last_avg)}

        print(f"[{epoch:03d}/{epochs}] loss={last_avg:.6f}")

    yield {"done": True, "final_loss": float(last_avg)}
    return model
