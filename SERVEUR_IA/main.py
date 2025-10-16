from fastapi import FastAPI
from typing import Optional, Tuple, Literal, List
from pydantic import BaseModel, Field, conint, confloat
from datetime import date,datetime
import torch
import os, time
from models.training import train_simple
from models.model_MLP import MLP
from models.optim import make_loss,make_optimizer

from classes import (
    TimeSeriesData,
    Parametres_temporels,
    Parametres_archi_reseau,
    Parametres_choix_reseau_neurones,
    Parametres_choix_loss_fct,
    Parametres_optimisateur,
    Parametres_entrainement,
    Parametres_visualisation_suivi,
    PaquetComplet,
    TrainingPayload,
)


##### EN attendant c'est ici :
from typing import List, Optional, Tuple
import numpy as np
import torch

def build_supervised_tensors(
    values: List[Optional[float]],
    window_len: int = 1,
    horizon: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construit (X, y) à partir d'une liste 'values' possiblement avec des None.
    X: [N, window_len], y: [N, 1]
    Stratégie: on ne garde que les fenêtres 100% valides (sans None) et une cible valide.
    """
    clean_vals = values  # on travaille direct, mais on filtre fenêtre par fenêtre
    X_list, y_list = [], []
    n = len(clean_vals)
    if n < window_len + horizon:
        return torch.empty(0, window_len), torch.empty(0, 1)

    for i in range(0, n - window_len - horizon + 1):
        seq = clean_vals[i : i + window_len]
        tgt = clean_vals[i + window_len + horizon - 1]
        # fenêtre valide ?
        if any(v is None for v in seq) or tgt is None:
            continue
        X_list.append(seq)
        y_list.append([tgt])

    if not X_list:
        return torch.empty(0, window_len), torch.empty(0, 1)

    X = torch.tensor(np.array(X_list, dtype=np.float32))
    y = torch.tensor(np.array(y_list, dtype=np.float32))
    return X, y




## EN LOCAL :
# uvicorn main:app --reload   


## EN SERVEUR SUR WIFI :
# 1- Faire sur terminal de mon mac :
#  ipconfig getifaddr en0  

# 2- Faire pour lancer le serveur
# cd /Users/gatienseguy/Documents/VSCode/PROJET_GL/SERVEUR_IA
#  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

#3 - à la txrx : metxztre : URL = ""http:// IP DE L'ORDI HOST DU SERVEUR :8000" "



app = FastAPI()

last_config_tempo = None
last_config_TimeSeries = None

last_config_series = None  




# ====================================
# ROUTES
# ====================================
@app.post("/train_full")
def recevoir_paquet(paquet: PaquetComplet):
    """
    Reçoit le paquet complet envoyé depuis l'interface Tkinter.
    """

    print("\n" + "="*80)
    print("PAQUET COMPLET REÇU")
    print("="*80)

    if paquet.Parametres_temporels:
        print(" Paramètres temporels :", paquet.Parametres_temporels.model_dump())
    if paquet.Parametres_choix_reseau_neurones:
        print("Choix du modèle :", paquet.Parametres_choix_reseau_neurones.modele)
    if paquet.Parametres_archi_reseau:
        print("Architecture :", paquet.Parametres_archi_reseau.model_dump())
    if paquet.Parametres_choix_loss_fct:
        print(" Fonction de perte :", paquet.Parametres_choix_loss_fct.model_dump())
    if paquet.Parametres_optimisateur:
        print(" Optimisateur :", paquet.Parametres_optimisateur.model_dump())
    if paquet.Parametres_entrainement:
        print(" Entraînement :", paquet.Parametres_entrainement.model_dump())
    if paquet.Parametres_visualisation_suivi:
        print(" Visualisation :", paquet.Parametres_visualisation_suivi.model_dump())

    print("="*80 + "\n")


    return {"status": "OK", "message": "Paquet complet recu et valide "}


@app.post("/tempoconfig")
def recevoir_TempoConfig(config: Parametres_temporels):
    """
    Reçoit uniquement la configuration temporelle
    """
    global last_config_tempo
    last_config_tempo = config
    print("\n" + "="*70)
    print(f" CONFIGURATION REÇUE \n")
    print("="*70)
    print("\n TEMPO :")
    print(f"   - Horizon : {config.horizon} \n")
    print(f"   - Dates   : {config.dates} \n")
    print(f"   - Pas     : {config.pas_temporel} \n")
    print(f"   - Split train : {config.portion_decoupage} \n")
    print(f"################ FIN ############  \n")
    return {"status": "OK", "message": "Paquet complet recu et valide "}




@app.post("/timeseries")
def recevoir_SeriesData(series: TimeSeriesData):
    """
    Reçoit les données de séries temporelles
    """
    global last_config_series
    last_config_series = series

    n = len(series.values)
    print("\n" + "="*70)
    print(" DONNÉES SÉRIE TEMPORELLE REÇUES")
    print("="*70)
    print(f"   - Nombre de points : {n}")
    if n > 0:
        print(f"   - Premier timestamp : {series.timestamps[0]} \n")
        print(f"   - Dernier timestamp : {series.timestamps[-1]} \n")
        print(f"   - Première valeur   : {series.values[0]}\n")
        print(f"   - Dernière valeur   : {series.values[-1]} \n")
    print("="*70 + "\n")

    print(f"################ FIN ############  \n")
    return {"status": "OK", "message": "Paquet complet recu et valide "}






@app.post("/training")
def training(payload: TrainingPayload):
    series: TimeSeriesData = payload.series
    cfg: PaquetComplet = payload.config

    # ----- TEMPO (mêmes noms) -----
    horizon = 1
    if cfg and cfg.Parametres_temporels and cfg.Parametres_temporels.horizon:
        horizon = max(1, int(cfg.Parametres_temporels.horizon))
    window_len = 1  # simple: one-step input; tu peux l’exposer plus tard

    # ----- (X,y) -----
    X, y = build_supervised_tensors(series.values, window_len=window_len, horizon=horizon)
    if X.numel() == 0:
        return {"status": "ERROR", "message": "Impossible de construire (X,y) (série trop courte ou valeurs manquantes)."}

    # ----- ARCHI (mêmes noms que Parametres_archi_reseau) -----
    hidden_size = 128
    nb_couches = 2
    dropout_rate = 0.0
    activation = "relu"
    use_batchnorm = False  # pas dans tes classes actuellement

    if cfg and cfg.Parametres_archi_reseau:
        if cfg.Parametres_archi_reseau.hidden_size is not None:
            hidden_size = int(cfg.Parametres_archi_reseau.hidden_size)
        if cfg.Parametres_archi_reseau.nb_couches is not None:
            nb_couches = int(cfg.Parametres_archi_reseau.nb_couches)
        if cfg.Parametres_archi_reseau.dropout_rate is not None:
            dropout_rate = float(cfg.Parametres_archi_reseau.dropout_rate)
        if cfg.Parametres_archi_reseau.fonction_activation is not None:
            act_map = {
                "ReLU": "relu",
                "GELU": "gelu",
                "tanh": "tanh",
                "sigmoid": "sigmoid",
                "leaky_relu": "leaky_relu",
            }
            activation = act_map.get(cfg.Parametres_archi_reseau.fonction_activation, "relu")

    # ----- LOSS (mêmes noms que Parametres_choix_loss_fct) -----
    loss_name = "mse"
    if cfg and cfg.Parametres_choix_loss_fct and cfg.Parametres_choix_loss_fct.fonction_perte:
        loss_name = cfg.Parametres_choix_loss_fct.fonction_perte.lower()

    # ----- OPTIM (mêmes noms que Parametres_optimisateur) -----
    optimizer_name = "adam"
    learning_rate = 1e-3
    weight_decay = 0.0
    if cfg and cfg.Parametres_optimisateur:
        if cfg.Parametres_optimisateur.optimisateur:
            optimizer_name = cfg.Parametres_optimisateur.optimisateur.lower()
        if cfg.Parametres_optimisateur.learning_rate is not None:
            learning_rate = float(cfg.Parametres_optimisateur.learning_rate)
        if cfg.Parametres_optimisateur.decroissance is not None:
            weight_decay = float(cfg.Parametres_optimisateur.decroissance)

    # ----- TRAIN (mêmes noms que Parametres_entrainement) -----
    epochs = 10
    batch_size = 64
    if cfg and cfg.Parametres_entrainement:
        if cfg.Parametres_entrainement.nb_epochs is not None:
            epochs = int(cfg.Parametres_entrainement.nb_epochs)
        if cfg.Parametres_entrainement.batch_size is not None:
            batch_size = int(cfg.Parametres_entrainement.batch_size)

    # ----- Device -----
    device = "cpu"

    # ----- Entraînement -----
    model, last_loss = train_simple(
        X, y,
        hidden_size=hidden_size,
        num_layers=nb_couches,        # <- mapping direct du même nom logique
        dropout_rate=dropout_rate,
        activation=activation,
        use_batchnorm=use_batchnorm,
        loss_name=loss_name,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    # ----- Sauvegarde -----
    run_dir = f"./mlp_simple_run_{int(time.time())}"
    os.makedirs(run_dir, exist_ok=True)
    save_path = f"{run_dir}/model.pt"
    torch.save({"state_dict": model.state_dict()}, save_path)

    return {
        "status": "OK",
        "message": "Entraînement terminé (boucle simple).",
        "final_loss": float(last_loss) if last_loss is not None else None,
        "saved_model": save_path,
        "details": {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "window_len": window_len,
            "horizon": horizon,
            "hidden_size": hidden_size,
            "nb_couches": nb_couches,
            "dropout_rate": dropout_rate,
            "fonction_activation": cfg.Parametres_archi_reseau.fonction_activation if (cfg and cfg.Parametres_archi_reseau and cfg.Parametres_archi_reseau.fonction_activation) else None,
            "optimisateur": optimizer_name,
            "learning_rate": learning_rate,
            "decroissance": weight_decay,
            "nb_epochs": epochs,
            "batch_size": batch_size,
        }
    }


@app.get("/")
def accueil():
    """
    Retourne les dernières données reçues via /tempoconfig et /timeseries
    """
    response = {"message": "Serveur IA actif !"}

    if last_config_tempo:
        response["tempo"] = {
            "horizon": last_config_tempo.horizon,
            "dates": last_config_tempo.dates,
            "pas_temporel": last_config_tempo.pas_temporel,
            "split_train": last_config_tempo.portion_decoupage,
        }
    else:
        response["tempo"] = "Aucune configuration temporelle reçue."

    if last_config_series:
        response["series"] = {
            "nb_points": len(last_config_series.values),
            "first": {
                "timestamp": last_config_series.timestamps[0],
                "value": last_config_series.values[0]
            },
            "last": {
                "timestamp": last_config_series.timestamps[-1],
                "value": last_config_series.values[-1]
            }
        }
    else:
        response["series"] = "Aucune série reçue."

    return response