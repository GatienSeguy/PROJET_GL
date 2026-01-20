# ====================================
# CONFIGURATION HARDWARE OPTIMISÉE
# ====================================
"""
Module de configuration hardware pour IRMA ML.
Version simplifiée et robuste.
"""

import os
import platform
from multiprocessing import cpu_count
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ====================================
# DATACLASS POUR LES INFOS HARDWARE
# ====================================
@dataclass
class HardwareInfo:
    """Informations sur le hardware détecté"""
    device: torch.device
    device_type: str
    num_cores: int
    num_workers: int
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    cuda_version: Optional[str] = None
    torch_threads: int = 1
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "🖥️  CONFIG HARDWARE IRMA ML",
            "=" * 50,
            f"OS: {platform.system()}",
            f"PyTorch: {torch.__version__}",
            f"CPU cores: {self.num_cores}",
            f"Workers: {self.num_workers}",
            f"Threads: {self.torch_threads}",
        ]
        
        if self.device_type == "cuda":
            lines.append(f"GPU: {self.gpu_name} ({self.gpu_memory_gb:.1f} GB)")
        elif self.device_type == "mps":
            lines.append("Device: Apple MPS")
        else:
            lines.append("Device: CPU")
        
        lines.append("=" * 50)
        return "\n".join(lines)


# ====================================
# SETUP PRINCIPAL
# ====================================
def setup_optimal_compute():
    """Configure PyTorch pour utiliser le meilleur hardware disponible."""
    
    num_cores = cpu_count()
    
    # Configurer les threads PyTorch
    try:
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(max(1, num_cores // 2))
    except Exception:
        pass
    
    # Variables d'environnement
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Initialiser les infos
    hw_info = HardwareInfo(
        device=torch.device("cpu"),
        device_type="cpu",
        num_cores=num_cores,
        num_workers=0,
        torch_threads=torch.get_num_threads(),
    )
    
    # Détection du device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        hw_info.device = device
        hw_info.device_type = "cuda"
        hw_info.gpu_count = torch.cuda.device_count()
        hw_info.gpu_name = torch.cuda.get_device_name(0)
        hw_info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        hw_info.cuda_version = torch.version.cuda
        hw_info.num_workers = min(num_cores, 4)
        
        # Optimisations CUDA
        torch.backends.cudnn.benchmark = True
        
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        hw_info.device = device
        hw_info.device_type = "mps"
        hw_info.num_workers = 0
        
    else:
        device = torch.device("cpu")
        hw_info.device = device
        hw_info.device_type = "cpu"
        # Pas de workers sur Windows pour éviter les problèmes
        hw_info.num_workers = 0 if platform.system() == "Windows" else min(num_cores, 4)
    
    return device, hw_info.num_workers, hw_info


# ====================================
# DATALOADER OPTIMISÉ
# ====================================
def get_optimal_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    device: Optional[torch.device] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """Crée un DataLoader optimisé."""
    
    if device is None:
        device = DEVICE
    if num_workers is None:
        num_workers = NUM_WORKERS
    
    dataset = TensorDataset(X, y)
    
    # Pin memory seulement pour CUDA
    pin_memory = (device.type == "cuda")
    
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    
    # Persistent workers si on a des workers
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    
    return DataLoader(dataset, **loader_kwargs)


# ====================================
# CLASSES POUR PARALLÉLISATION
# ====================================
class ParallelTrainer:
    """Wrapper pour Multi-GPU si disponible."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.device = device or DEVICE
        self.model = model.to(self.device)
        
        # Multi-GPU si disponible
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"🚀 Multi-GPU: {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
    
    def get_model(self) -> nn.Module:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model


class AMPTrainer:
    """Trainer avec Mixed Precision pour CUDA."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.device = device or DEVICE
        self.model = model.to(self.device)
        self.use_amp = (self.device.type == "cuda")
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def get_model(self) -> nn.Module:
        return self.model


# ====================================
# UTILITAIRES
# ====================================
def get_device_info() -> Dict[str, Any]:
    """Retourne les infos device pour l'API."""
    return {
        "device": str(DEVICE),
        "device_type": HARDWARE_INFO.device_type,
        "num_cores": HARDWARE_INFO.num_cores,
        "num_workers": HARDWARE_INFO.num_workers,
        "torch_threads": HARDWARE_INFO.torch_threads,
        "gpu_name": HARDWARE_INFO.gpu_name,
        "gpu_count": HARDWARE_INFO.gpu_count,
        "gpu_memory_gb": HARDWARE_INFO.gpu_memory_gb,
    }


def benchmark_device(size: int = 1500, iterations: int = 30) -> Dict[str, float]:
    """Benchmark rapide."""
    import time
    
    x = torch.randn(size, size, device=DEVICE)
    
    # Warmup
    for _ in range(3):
        _ = torch.mm(x, x)
    
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        _ = torch.mm(x, x)
    
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    return {
        "device": str(DEVICE),
        "matrix_size": size,
        "iterations": iterations,
        "total_time_s": round(elapsed, 3),
        "gflops": round((2 * size**3 * iterations) / elapsed / 1e9, 1),
    }


# ====================================
# INITIALISATION
# ====================================
try:
    DEVICE, NUM_WORKERS, HARDWARE_INFO = setup_optimal_compute()
    print(HARDWARE_INFO)
except Exception as e:
    print(f"⚠️ Erreur config hardware: {e}")
    DEVICE = torch.device("cpu")
    NUM_WORKERS = 0
    HARDWARE_INFO = HardwareInfo(
        device=DEVICE,
        device_type="cpu",
        num_cores=cpu_count(),
        num_workers=0,
        torch_threads=1
    )
