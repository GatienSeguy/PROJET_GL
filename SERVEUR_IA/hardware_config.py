# ====================================
# CONFIGURATION HARDWARE OPTIMISÉE AVEC MULTIPROCESSING
# ====================================
"""
Module de configuration hardware pour IRMA ML.
Détecte et configure automatiquement le meilleur device (CUDA, MPS, CPU)
et optimise la parallélisation sur tous les cœurs avec multiprocessing.

Usage:
    from .hardware_config import (
        DEVICE, NUM_WORKERS, HARDWARE_INFO,
        get_optimal_dataloader, setup_torch_parallelism,
        ParallelTrainer
    )
"""

import os
import sys
import platform
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
from functools import partial
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ====================================
# CONFIGURATION MULTIPROCESSING WINDOWS
# ====================================
# CRITIQUE pour Windows: éviter les erreurs de spawn
if platform.system() == "Windows":
    # Forcer spawn au lieu de fork (plus sûr sur Windows)
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Déjà configuré


# ====================================
# DATACLASS POUR LES INFOS HARDWARE
# ====================================
@dataclass
class HardwareInfo:
    """Informations sur le hardware détecté"""
    device: torch.device
    device_type: str  # "cuda", "mps", "cpu"
    num_cores: int
    num_physical_cores: int
    num_workers: int
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    cuda_version: Optional[str] = None
    cudnn_version: Optional[int] = None
    torch_threads: int = 1
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "🖥️  CONFIGURATION HARDWARE IRMA ML - PARALLÉLISATION",
            "=" * 60,
            f"📌 OS: {platform.system()} {platform.release()}",
            f"📌 Python: {platform.python_version()}",
            f"📌 PyTorch: {torch.__version__}",
            f"📌 CPU cœurs logiques: {self.num_cores}",
            f"📌 CPU cœurs physiques: {self.num_physical_cores}",
            f"📌 DataLoader workers: {self.num_workers}",
            f"📌 Torch threads: {self.torch_threads}",
        ]
        
        if self.device_type == "cuda":
            lines.extend([
                f"\n🚀 CUDA activé:",
                f"   GPU: {self.gpu_name} x{self.gpu_count}",
                f"   Mémoire: {self.gpu_memory_gb:.1f} GB",
                f"   CUDA: {self.cuda_version}",
                f"   cuDNN: {self.cudnn_version}",
            ])
        elif self.device_type == "mps":
            lines.append(f"\n🍎 MPS (Apple Silicon) activé")
        else:
            lines.append(f"\n💻 CPU activé ({self.num_cores} cœurs parallélisés)")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ====================================
# DÉTECTION CŒURS PHYSIQUES
# ====================================
def get_physical_cores() -> int:
    """Retourne le nombre de cœurs physiques (pas hyperthreading)"""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or cpu_count()
    except ImportError:
        # Estimation: généralement num_logical / 2 pour hyperthreading
        return max(1, cpu_count() // 2)


# ====================================
# SETUP PARALLÉLISME PYTORCH
# ====================================
def setup_torch_parallelism(num_threads: Optional[int] = None) -> int:
    """
    Configure PyTorch pour utiliser tous les cœurs CPU disponibles.
    
    Args:
        num_threads: Nombre de threads (auto-détecté si None)
    
    Returns:
        Nombre de threads configurés
    """
    if num_threads is None:
        num_threads = get_physical_cores()
    
    # Configuration des threads PyTorch
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, num_threads // 2))
    
    # Variables d'environnement pour les bibliothèques numériques
    env_vars = {
        "OMP_NUM_THREADS": str(num_threads),
        "MKL_NUM_THREADS": str(num_threads),
        "OPENBLAS_NUM_THREADS": str(num_threads),
        "VECLIB_MAXIMUM_THREADS": str(num_threads),
        "NUMEXPR_NUM_THREADS": str(num_threads),
        # Éviter les conflits de bibliothèques
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return num_threads


# ====================================
# SETUP PRINCIPAL
# ====================================
def setup_optimal_compute() -> Tuple[torch.device, int, HardwareInfo]:
    """
    Configure PyTorch pour utiliser au maximum le hardware disponible.
    
    Returns:
        Tuple[device, num_workers, hardware_info]
    """
    num_cores = cpu_count()
    num_physical = get_physical_cores()
    
    # Configurer le parallélisme CPU
    torch_threads = setup_torch_parallelism(num_physical)
    
    # Initialiser les infos hardware
    hw_info = HardwareInfo(
        device=torch.device("cpu"),
        device_type="cpu",
        num_cores=num_cores,
        num_physical_cores=num_physical,
        num_workers=0,
        torch_threads=torch_threads,
    )
    
    # Détection du meilleur device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        hw_info.device = device
        hw_info.device_type = "cuda"
        
        # Optimisations CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        # Infos GPU
        hw_info.gpu_count = torch.cuda.device_count()
        hw_info.gpu_name = torch.cuda.get_device_name(0)
        hw_info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        hw_info.cuda_version = torch.version.cuda
        hw_info.cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        
        # Workers pour CUDA (pas trop pour éviter le bottleneck)
        hw_info.num_workers = min(num_physical, 4)
        
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        hw_info.device = device
        hw_info.device_type = "mps"
        hw_info.num_workers = 0  # MPS ne supporte pas bien les workers
        
    else:
        device = torch.device("cpu")
        hw_info.device = device
        hw_info.device_type = "cpu"
        
        # Optimisations CPU
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        
        # Workers sur CPU (attention sur Windows)
        if platform.system() == "Windows":
            hw_info.num_workers = 0  # Windows a des problèmes avec multiprocessing
        else:
            hw_info.num_workers = min(num_physical, 4)
    
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
    """
    Crée un DataLoader optimisé pour le hardware détecté.
    """
    if device is None:
        device = DEVICE
    if num_workers is None:
        num_workers = NUM_WORKERS
    
    dataset = TensorDataset(X, y)
    
    # Pin memory pour transfert GPU plus rapide
    pin_memory = (device.type == "cuda")
    
    # Configuration selon le device
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
        loader_kwargs["prefetch_factor"] = 2
    
    return DataLoader(dataset, **loader_kwargs)


# ====================================
# CLASSE DE TRAINING PARALLÉLISÉ
# ====================================
class ParallelTrainer:
    """
    Wrapper pour l'entraînement parallélisé sur CPU multi-cœurs.
    Utilise DataParallel pour GPU multiple ou optimise les threads CPU.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.device = device or DEVICE
        self.model = model.to(self.device)
        
        # Multi-GPU si disponible
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"🚀 Multi-GPU activé: {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self._original_model = model
    
    def get_model(self) -> nn.Module:
        """Retourne le modèle original (sans DataParallel wrapper)"""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model
    
    def train_step(self, xb: torch.Tensor, yb: torch.Tensor, 
                   criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """Effectue un pas d'entraînement optimisé"""
        # Transfert non-bloquant vers le device
        xb = xb.to(self.device, non_blocking=True)
        yb = yb.to(self.device, non_blocking=True)
        
        # Zero grad optimisé
        optimizer.zero_grad(set_to_none=True)
        
        # Forward
        pred = self.model(xb)
        loss = criterion(pred, yb)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        return loss.item()


# ====================================
# FONCTIONS DE PARALLÉLISATION CPU
# ====================================
def parallel_map(func: Callable, items: List, n_jobs: int = None) -> List:
    """
    Applique une fonction en parallèle sur une liste d'items.
    
    Args:
        func: Fonction à appliquer
        items: Liste d'items
        n_jobs: Nombre de processus (auto si None)
    
    Returns:
        Liste des résultats
    """
    if n_jobs is None:
        n_jobs = get_physical_cores()
    
    # Pour les petites listes, pas besoin de paralléliser
    if len(items) < n_jobs * 2:
        return [func(item) for item in items]
    
    # Windows: utiliser ThreadPool au lieu de ProcessPool
    if platform.system() == "Windows":
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            return list(executor.map(func, items))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            return list(executor.map(func, items))


def parallel_predict(model: nn.Module, X: torch.Tensor, 
                     batch_size: int = 1024, device: torch.device = None) -> torch.Tensor:
    """
    Prédiction parallélisée par batches.
    """
    if device is None:
        device = DEVICE
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(device, non_blocking=True)
            pred = model(batch)
            predictions.append(pred.cpu())
    
    return torch.cat(predictions, dim=0)


# ====================================
# MIXED PRECISION TRAINING (CUDA)
# ====================================
class AMPTrainer:
    """
    Trainer avec Automatic Mixed Precision pour CUDA.
    Accélère l'entraînement de ~2x sur GPU compatibles.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.device = device or DEVICE
        self.model = model.to(self.device)
        self.use_amp = (self.device.type == "cuda")
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("⚡ Mixed Precision (AMP) activé")
        else:
            self.scaler = None
    
    def train_step(self, xb: torch.Tensor, yb: torch.Tensor,
                   criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """Pas d'entraînement avec AMP si disponible"""
        xb = xb.to(self.device, non_blocking=True)
        yb = yb.to(self.device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                pred = self.model(xb)
                loss = criterion(pred, yb)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            pred = self.model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        
        return loss.item()


# ====================================
# UTILITAIRES
# ====================================
def get_device_info() -> Dict[str, Any]:
    """Retourne un dict avec les infos device pour l'API"""
    return {
        "device": str(DEVICE),
        "device_type": HARDWARE_INFO.device_type,
        "num_cores": HARDWARE_INFO.num_cores,
        "num_physical_cores": HARDWARE_INFO.num_physical_cores,
        "num_workers": HARDWARE_INFO.num_workers,
        "torch_threads": HARDWARE_INFO.torch_threads,
        "gpu_name": HARDWARE_INFO.gpu_name,
        "gpu_count": HARDWARE_INFO.gpu_count,
        "gpu_memory_gb": HARDWARE_INFO.gpu_memory_gb,
    }


def benchmark_device(size: int = 2000, iterations: int = 50) -> Dict[str, float]:
    """Benchmark rapide du device"""
    import time
    
    x = torch.randn(size, size, device=DEVICE)
    
    # Warmup
    for _ in range(5):
        _ = torch.mm(x, x)
    
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        y = torch.mm(x, x)
    
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    return {
        "device": str(DEVICE),
        "matrix_size": size,
        "iterations": iterations,
        "total_time_s": round(elapsed, 3),
        "time_per_op_ms": round((elapsed / iterations) * 1000, 2),
        "gflops": round((2 * size**3 * iterations) / elapsed / 1e9, 1),
    }


# ====================================
# INITIALISATION AU CHARGEMENT
# ====================================
# Ces variables sont initialisées une seule fois
DEVICE, NUM_WORKERS, HARDWARE_INFO = setup_optimal_compute()

# Afficher les infos au démarrage (seulement si pas en mode silencieux)
if not os.environ.get("IRMA_SILENT"):
    print(HARDWARE_INFO)
