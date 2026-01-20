#!/usr/bin/env python3
"""
Script de diagnostic hardware pour IRMA ML.
Vérifie la configuration et benchmark les performances.

Usage:
    python -m SERVEUR_IA.check_hardware
"""

import sys
import platform
import multiprocessing as mp
from multiprocessing import cpu_count

print("=" * 70)
print("🔍 DIAGNOSTIC HARDWARE IRMA ML")
print("=" * 70)

# ====================================
# INFORMATIONS SYSTÈME
# ====================================
print(f"\n📌 SYSTÈME")
print(f"   OS: {platform.system()} {platform.release()} ({platform.machine()})")
print(f"   Python: {platform.python_version()}")
print(f"   Processeur: {platform.processor() or 'Non détecté'}")

# ====================================
# CPU
# ====================================
print(f"\n📌 CPU")
print(f"   Cœurs logiques: {cpu_count()}")

try:
    import psutil
    physical_cores = psutil.cpu_count(logical=False)
    print(f"   Cœurs physiques: {physical_cores}")
    print(f"   Mémoire RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
except ImportError:
    print("   ⚠️  psutil non installé (pip install psutil pour plus de détails)")
    physical_cores = max(1, cpu_count() // 2)
    print(f"   Cœurs physiques (estimé): {physical_cores}")

# ====================================
# PYTORCH
# ====================================
print(f"\n📌 PYTORCH")
try:
    import torch
    print(f"   Version: {torch.__version__}")
    print(f"   Threads actuels: {torch.get_num_threads()}")
    print(f"   Interop threads: {torch.get_num_interop_threads()}")
    
    # CUDA
    print(f"\n📌 CUDA")
    print(f"   Disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Version CUDA: {torch.version.cuda}")
        print(f"   cuDNN disponible: {torch.backends.cudnn.is_available()}")
        if torch.backends.cudnn.is_available():
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   Nombre de GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   🎮 GPU {i}: {props.name}")
            print(f"      Mémoire: {props.total_memory / 1e9:.1f} GB")
            print(f"      Compute capability: {props.major}.{props.minor}")
            print(f"      Multi-processeurs: {props.multi_processor_count}")
    
    # MPS (Apple Silicon)
    print(f"\n📌 MPS (Apple Silicon)")
    if hasattr(torch.backends, 'mps'):
        print(f"   Disponible: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print(f"   Built: {torch.backends.mps.is_built()}")
    else:
        print(f"   Non supporté dans cette version de PyTorch")
    
except ImportError:
    print("   ❌ PyTorch non installé!")
    sys.exit(1)

# ====================================
# CONFIGURATION OPTIMALE
# ====================================
print(f"\n📌 CONFIGURATION RECOMMANDÉE")

# Déterminer le device optimal
if torch.cuda.is_available():
    device = "cuda"
    print(f"   Device recommandé: CUDA (GPU)")
    print(f"   ✅ Installez PyTorch avec CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print(f"   Device recommandé: MPS (Apple Silicon)")
else:
    device = "cpu"
    print(f"   Device recommandé: CPU")
    print(f"   💡 Pour utiliser le GPU, installez PyTorch avec CUDA:")
    print(f"      pip install torch --index-url https://download.pytorch.org/whl/cu121")

# Workers
if platform.system() == "Windows":
    workers = 0
    print(f"   DataLoader workers: 0 (Windows - multiprocessing limité)")
else:
    workers = min(physical_cores, 4)
    print(f"   DataLoader workers: {workers}")

print(f"   Threads PyTorch: {physical_cores}")

# ====================================
# BENCHMARK RAPIDE
# ====================================
print(f"\n📌 BENCHMARK RAPIDE")

device_obj = torch.device(device)
sizes = [500, 1000, 2000]

for size in sizes:
    x = torch.randn(size, size, device=device_obj)
    
    # Warmup
    for _ in range(3):
        _ = torch.mm(x, x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    import time
    start = time.time()
    iterations = 20
    for _ in range(iterations):
        y = torch.mm(x, x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    gflops = (2 * size**3 * iterations) / elapsed / 1e9
    
    print(f"   Matrix {size}x{size}: {elapsed/iterations*1000:.1f} ms/op, {gflops:.1f} GFLOPS")

# ====================================
# TEST MULTIPROCESSING
# ====================================
print(f"\n📌 TEST MULTIPROCESSING")

def worker_test(x):
    return x * x

try:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    
    # Test ThreadPool
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker_test, range(100)))
    print(f"   ThreadPoolExecutor: ✅ OK")
    
    # Test ProcessPool (peut échouer sur Windows dans certains contextes)
    if platform.system() != "Windows":
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker_test, range(100)))
        print(f"   ProcessPoolExecutor: ✅ OK")
    else:
        print(f"   ProcessPoolExecutor: ⚠️ Skipped (Windows)")
        
except Exception as e:
    print(f"   ❌ Erreur multiprocessing: {e}")

# ====================================
# RÉSUMÉ
# ====================================
print("\n" + "=" * 70)
print("📊 RÉSUMÉ")
print("=" * 70)

status = "✅" if (torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())) else "⚠️"
print(f"{status} Device: {device.upper()}")
print(f"✅ CPU cores: {cpu_count()} logiques, {physical_cores} physiques")
print(f"✅ PyTorch: {torch.__version__}")

if device == "cpu" and not torch.cuda.is_available():
    print(f"\n💡 CONSEIL: Pour de meilleures performances, installez PyTorch avec CUDA:")
    print(f"   pip uninstall torch torchvision torchaudio")
    print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 70)
