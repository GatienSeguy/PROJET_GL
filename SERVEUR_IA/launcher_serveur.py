#!/usr/bin/env python3
"""
🚀 LANCEUR RAPIDE DU SERVEUR IA
================================

Script simplifié pour lancer rapidement le serveur avec les bons chemins.

Usage:
    python start_server_simple.py              # Défaut: local, CPU
    python start_server_simple.py mps          # GPU Apple Silicon
    python start_server_simple.py mps network  # GPU + réseau local
    python start_server_simple.py cpu prod     # CPU mode production
"""

import sys
import subprocess
from pathlib import Path
import socket

json_path = "/Users/gatienseguy/Documents/VSCode/PROJET_GL/SERVEUR_DATA/Datas/Boites_per_day.json"  # ton fichier JSON existant


def get_local_ip():
    """Récupère l'IP locale."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        try:
            result = subprocess.run(
                ["ipconfig", "getifaddr", "en0"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except:
            return "127.0.0.1"


def find_project_root():
    """Trouve la racine du projet contenant SERVEUR_IA/"""
    current = Path.cwd()
    
    for parent in [current] + list(current.parents):
        if (parent / "SERVEUR_IA").exists():
            return parent
    
    return None


def check_pytorch():
    """Vérifie PyTorch et les devices disponibles."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        devices = {
            "cpu": "✅",
            "cuda": "✅" if torch.cuda.is_available() else "❌",
            "mps": "✅" if (hasattr(torch.backends, 'mps') and 
                          torch.backends.mps.is_available()) else "❌"
        }
        
        return devices
    except ImportError:
        print("❌ PyTorch non installé!")
        return None


def main():
    print("\n" + "="*60)
    print("🚀 LANCEUR SERVEUR IA")
    print("="*60 + "\n")
    
    # Configuration par défaut
    device = "cpu"
    host = "127.0.0.1"
    port = 8000
    reload = True
    
    # Parsing des arguments
    args = sys.argv[1:]
    
    if len(args) >= 1:
        if args[0].lower() in ["cpu", "cuda", "mps"]:
            device = args[0].lower()
    
    if len(args) >= 2:
        mode = args[1].lower()
        if mode in ["network", "lan", "wifi", "0.0.0.0"]:
            host = "0.0.0.0"
        elif mode in ["prod", "production"]:
            reload = False
    
    # Trouver le projet
    project_root = find_project_root()
    
    if not project_root:
        print("❌ Impossible de trouver SERVEUR_IA/")
        print("💡 Lancez ce script depuis le répertoire du projet")
        sys.exit(1)
    
    print(f"📁 Projet: {project_root}")
    
    # Vérifier PyTorch
    devices = check_pytorch()
    if not devices:
        sys.exit(1)
    
    print(f"\n💻 Devices PyTorch:")
    for d, status in devices.items():
        print(f"   {d.upper()}: {status}")
    
    if devices[device] == "❌":
        print(f"\n⚠️  {device.upper()} non disponible, utilisation du CPU")
        device = "cpu"
    
    # Créer __init__.py si nécessaire
    init_file = project_root / "SERVEUR_IA" / "__init__.py"
    init_file.touch(exist_ok=True)
    
    # Afficher la config
    print(f"\n📊 Configuration:")
    print(f"   Device: {device.upper()}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {'Oui' if reload else 'Non'}")
    
    if host == "0.0.0.0":
        local_ip = get_local_ip()
        print(f"\n🌐 URLs:")
        print(f"   Local:  http://127.0.0.1:{port}")
        print(f"   Réseau: http://{local_ip}:{port}")
        print(f"   Docs:   http://{local_ip}:{port}/docs")
    else:
        print(f"\n🌐 URL: http://{host}:{port}")
        print(f"   Docs: http://{host}:{port}/docs")
    
    print("\n" + "="*60)
    print("✨ Démarrage du serveur...\n")
    
    # Construire la commande
    cmd = [
        sys.executable, "-m", "uvicorn",
        "SERVEUR_IA.main:app",
        "--host", host,
        "--port", str(port),
    ]
    
    if reload:
        cmd.append("--reload")
        cmd.extend(["--reload-dir", str(project_root)])
    
    # Lancer
    try:
        subprocess.run(cmd, cwd=str(project_root), check=True)
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("🛑 Serveur arrêté")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()