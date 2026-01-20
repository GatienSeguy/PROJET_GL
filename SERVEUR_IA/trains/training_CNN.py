# trains/training_CNN.py
"""
Module d'entraînement CNN optimisé avec parallélisation multi-cœurs/GPU.
Utilise multiprocessing pour maximiser l'utilisation CPU.
"""
import inspect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

from ..models.optim import make_loss, make_optimizer
from ..models.model_CNN import CNN
from ..hardware_config import (
    DEVICE, NUM_WORKERS, HARDWARE_INFO,
    get_optimal_dataloader, ParallelTrainer, AMPTrainer
)


def _build_cnn_safely(in_dim: int, out_dim: int, **kwargs):
    """
    Crée un CNN1D en détectant la signature réelle et en mappant les alias.
    """
    sig = inspect.signature(CNN.__init__)
    params = set(sig.parameters.keys())
    resolved = {}

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

    put(kwargs.get("hidden_size", 64), "hidden_dim", "hidden_size", "width")
    put(kwargs.get("nb_couches", 2), "nb_couches", "n_layers", "depth", "layers")
    put(kwargs.get("activation", "relu"), "activation", "act", "activation_name")
    put(kwargs.get("use_batchnorm", False), "use_batchnorm", "batchnorm", "bn", "use_bn")
    put(kwargs.get("kernel_size", 3), "kernel_size", "ksize")
    put(kwargs.get("stride", 1), "stride")
    put(kwargs.get("padding", 1), "padding")

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
    device: torch.device = None,
    
    # --- PARALLÉLISATION ---
    use_amp: bool = None,
    use_parallel: bool = True,
):
    """
    Entraîne un CNN1D avec parallélisation optimale.
    
    Optimisations:
    - DataLoader multi-workers avec pin_memory
    - Mixed Precision (AMP) sur CUDA
    - Multi-GPU avec DataParallel
    - Transferts non-bloquants
    """
    if device is None:
        device = DEVICE
    
    if use_amp is None:
        use_amp = (device.type == "cuda")
    
    print(f"[CNN TRAIN] 🚀 Device: {device}")
    print(f"[CNN TRAIN] 📊 Workers: {NUM_WORKERS}, Threads: {HARDWARE_INFO.torch_threads}")
    print(f"[CNN TRAIN] ⚡ AMP: {use_amp}, Parallel: {use_parallel}")
    
    # Reshape si nécessaire
    if X.ndim == 2:
        X = X.unsqueeze(1)  # (B, seq_len) -> (B, 1, seq_len)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    # ========== DATALOADER OPTIMISÉ ==========
    loader = get_optimal_dataloader(
        X, y,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        device=device
    )

    # ========== MODÈLE ==========
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
    )
    
    # ========== PARALLÉLISATION ==========
    if use_parallel and use_amp and device.type == "cuda":
        trainer = AMPTrainer(model, device)
        model = trainer.model
    elif use_parallel:
        trainer = ParallelTrainer(model, device)
        model = trainer.model
    else:
        model = model.to(device)
        trainer = None

    criterion = make_loss({"name": loss_name})
    optimizer = make_optimizer(model, {"name": optimizer_name, "lr": learning_rate, "weight_decay": weight_decay})
    
    scaler = None
    if use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    # ========== BOUCLE D'ENTRAÎNEMENT ==========
    last_avg = None
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total, n = 0.0, 0
        
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    if pred.ndim == 3:
                        pred = pred.mean(dim=2)
                    if pred.ndim == 3:
                        pred = pred.squeeze(1)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb)
                if pred.ndim == 3:
                    pred = pred.mean(dim=2)
                if pred.ndim == 3:
                    pred = pred.squeeze(1)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
            
            total += loss.item() * xb.size(0)
            n += xb.size(0)

        last_avg = total / max(1, n)
        epoch_duration = time.time() - epoch_start
        samples_per_sec = n / epoch_duration if epoch_duration > 0 else 0

        yield {
            "type": "epoch",
            "epochs": epoch,
            "avg_loss": float(last_avg),
            "epoch_s": 1/epoch_duration if epoch_duration > 0 else 0,
            "samples_per_sec": samples_per_sec,
            "device": str(device)
        }

        print(f"[CNN {epoch:03d}/{epochs}] loss={last_avg:.6f} ({epoch_duration:.2f}s, {samples_per_sec:.0f} samples/s)")

    yield {"done": True, "final_loss": float(last_avg)}

    if trainer and hasattr(trainer, 'get_model'):
        return trainer.get_model()
    elif isinstance(model, nn.DataParallel):
        return model.module
    return model
