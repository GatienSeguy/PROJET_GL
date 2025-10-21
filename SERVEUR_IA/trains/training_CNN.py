# trains/training_cnn.py
import inspect
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..models.optim import make_loss, make_optimizer
from ..models.model_CNN import CNN 


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
    weight_decay: float = 0.0,

    # --- TRAIN ---
    batch_size: int = 64,
    epochs: int = 10,
    device: str = "cpu",
):
    try:
        print(f"[CNN] Début train_CNN: X.shape={X.shape}, y.shape={y.shape}")
        
        if y.ndim == 1:
            y = y.unsqueeze(1)
        
        print(f"[CNN] Après reshape y: y.shape={y.shape}")
        
        # CRITIQUE: Adapter kernel_size à la taille de l'entrée
        # X est (B, in_dim), donc la longueur de séquence = in_dim
        seq_length = X.shape[1]
        
        # Le kernel ne peut pas être plus grand que la séquence
        if kernel_size > seq_length:
            original_kernel = kernel_size
            kernel_size = min(kernel_size, seq_length)
            print(f"[CNN] ⚠️ kernel_size réduit de {original_kernel} à {kernel_size} (seq_length={seq_length})")
        
        # Adapter le padding aussi pour éviter les erreurs
        if padding >= kernel_size:
            padding = max(0, kernel_size - 1)
            print(f"[CNN] ⚠️ padding ajusté à {padding}")

        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        
        print(f"[CNN] Création du modèle avec in_dim={X.shape[1]}, out_dim={y.shape[1]}, kernel_size={kernel_size}, padding={padding}")
        
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
        
        print(f"[CNN] Modèle créé avec succès")

        criterion = make_loss({"name": loss_name})
        optimizer = make_optimizer(model, {"name": optimizer_name, "lr": learning_rate, "weight_decay": weight_decay})
        
        print(f"[CNN] Loss et optimizer créés")

        last_avg = None
        for epoch in range(1, epochs + 1):
            total, n = 0.0, 0
            for xb, yb in loader:
                try:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    
                    # IMPORTANT: CNN attend (B, C, L) où C=channels, L=length
                    # X est (B, in_dim) donc on ajoute dimension channel: (B, 1, in_dim)
                    xb = xb.unsqueeze(1)
                    
                    pred = model(xb)
                    
                    # Si pred est 3D (B, C, L), on prend la moyenne ou dernière valeur
                    if pred.ndim == 3:
                        # Utiliser adaptive pooling ou prendre la moyenne
                        pred = pred.mean(dim=2)  # Moyenne sur la dimension temporelle
                    
                    # Si pred a encore une dimension channel, on la squeeze
                    if pred.ndim == 3:
                        pred = pred.squeeze(1)
                    
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * xb.size(0)
                    n += xb.size(0)
                except Exception as e:
                    print(f"[CNN] Erreur dans batch epoch {epoch}: {e}")
                    print(f"[CNN] xb.shape={xb.shape}, yb.shape={yb.shape}, pred.shape={pred.shape if 'pred' in locals() else 'N/A'}")
                    import traceback
                    traceback.print_exc()
                    raise
                    
            last_avg = total / max(1, n)

            k = 1
            if epoch % k == 0:
                yield {"epochs": epoch, "avg_loss": float(last_avg)}

            if epoch % 10 == 0 or epoch == epochs:
                print(f"[CNN {epoch:03d}/{epochs}] loss={last_avg:.6f}")

        yield {"done": True, "final_loss": float(last_avg)}
        
        print(f"[CNN] Entraînement terminé avec succès")
        return model
        
    except Exception as e:
        print(f"[CNN] ERREUR FATALE dans train_CNN: {e}")
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": f"Erreur CNN: {str(e)}"}
        raise