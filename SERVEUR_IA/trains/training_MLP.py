# trains/training_MLP.py
"""
Module d'entraînement MLP optimisé avec parallélisation multi-cœurs/GPU.
Utilise multiprocessing pour maximiser l'utilisation CPU.
"""
import inspect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

from ..models.optim import make_loss, make_optimizer
from ..models.model_MLP import MLP
from ..hardware_config import (
    DEVICE, NUM_WORKERS, HARDWARE_INFO,
    get_optimal_dataloader, ParallelTrainer, AMPTrainer
)


def _build_mlp_safely(in_dim: int, out_dim: int, **kwargs):
    """
    Crée un MLP en détectant la signature réelle et en mappant les alias.
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

    for k, v in kwargs.items():
        if k in params:
            resolved[k] = v

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
    device: torch.device = None,
    
    # --- PARALLÉLISATION ---
    use_amp: bool = None,  # Mixed Precision (auto si None)
    use_parallel: bool = True,  # Multi-GPU si disponible
):
    """
    Entraîne un MLP avec parallélisation optimale sur tous les cœurs.
    
    Optimisations:
    - DataLoader multi-workers avec pin_memory
    - Mixed Precision (AMP) sur CUDA
    - Multi-GPU avec DataParallel
    - Transferts non-bloquants
    - zero_grad(set_to_none=True) optimisé
    """
    # Utiliser le device global optimisé
    if device is None:
        device = DEVICE
    
    # Auto-détection AMP
    if use_amp is None:
        use_amp = (device.type == "cuda")
    
    print(f"[MLP TRAIN] 🚀 Device: {device}")
    print(f"[MLP TRAIN] 📊 Workers: {NUM_WORKERS}, Threads: {HARDWARE_INFO.torch_threads}")
    print(f"[MLP TRAIN] ⚡ AMP: {use_amp}, Parallel: {use_parallel}")
    
    if y.ndim == 1:
        y = y.unsqueeze(1)

    # ========== DATALOADER OPTIMISÉ ==========
    loader = get_optimal_dataloader(
        X, y,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        device=device
    )

    # ========== MODÈLE ==========
    model = _build_mlp_safely(
        in_dim=X.shape[1],
        out_dim=y.shape[1],
        hidden_size=hidden_size,
        nb_couches=nb_couches,
        dropout_rate=dropout_rate,
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
    optimizer = make_optimizer(
        model,
        {"name": optimizer_name, "lr": learning_rate, "weight_decay": weight_decay}
    )
    
    # Scaler pour AMP
    scaler = None
    if use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    # ========== BOUCLE D'ENTRAÎNEMENT ==========
    last_avg = None
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total, n = 0.0, 0
        model.train()
        
        for xb, yb in loader:
            # Transfert non-bloquant
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            # Zero grad optimisé
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler is not None:
                # Mixed Precision
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
            
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        
        last_avg = total / max(1, n) if n > 0 else 0.0
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

        print(f"[MLP {epoch:03d}/{epochs}] loss={last_avg:.6f} ({epoch_duration:.2f}s, {samples_per_sec:.0f} samples/s)")

    yield {"done": True, "final_loss": float(last_avg)}

    # Retourner le modèle original (sans DataParallel wrapper)
    if trainer and hasattr(trainer, 'get_model'):
        return trainer.get_model()
    elif isinstance(model, nn.DataParallel):
        return model.module
    return model
