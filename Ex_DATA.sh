#!/usr/bin/env bash

# Aller dans le dossier du script (optionnel mais pratique si tu le mets à la racine du projet)
cd "$(dirname "$0")"

# Variables d'environnement (pour éviter les soucis avec libomp / MKL)
export KMP_DUPLICATE_LIB_OK=TRUEDD
export OMP_NUM_THREADS=1

# Lancement du serveur Uvicorn
python -m uvicorn SERVEUR_DATA.main2:app --host 0.0.0.0 --port 8001 --reload