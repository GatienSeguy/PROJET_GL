from fastapi import FastAPI

from fastapi.responses import StreamingResponse
import json 
  
from typing import Optional, Tuple, Literal, List
from pydantic import BaseModel, Field, conint, confloat
from datetime import date,datetime
import torch
import os, time

from models.model_MLP import MLP
from models.optim import make_loss,make_optimizer

#Train Modele
from models.training_MLP import train_MLP
from models.training_CNN import train_CNN


from classes import (
    TimeSeriesData,
    Parametres_temporels,
    Parametres_choix_reseau_neurones,
    Parametres_choix_loss_fct,
    Parametres_optimisateur,
    Parametres_entrainement,
    Parametres_visualisation_suivi,
    Parametres_archi_reseau_MLP,
    Parametres_archi_reseau_CNN,
    PaquetComplet
)

from fonctions_pour_main import(
    build_supervised_tensors,
    _parse_any_datetime,
    filter_series_by_dates,
    build_supervised_tensors_with_step,
    split_train_test
)
##### EN attendant c'est ici :


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
def training(payload: PaquetComplet):
#     series: TimeSeriesData = 
# En attendant la requête entre serveur de data on implémente en dur dans le code la serie temporelle
    series = TimeSeriesData(
    timestamps=[
        datetime.fromisoformat("2025-01-01T00:00:00"),
        datetime.fromisoformat("2025-01-01T01:00:00"),
        datetime.fromisoformat("2025-01-01T02:00:00"),
        datetime.fromisoformat("2025-01-01T03:00:00"),
        datetime.fromisoformat("2025-01-01T04:00:00"),
        datetime.fromisoformat("2025-01-01T05:00:00"),
        datetime.fromisoformat("2025-01-01T06:00:00"),
        datetime.fromisoformat("2025-01-01T07:00:00"),
        datetime.fromisoformat("2025-01-01T08:00:00"),
        datetime.fromisoformat("2025-01-01T09:00:00"),
        datetime.fromisoformat("2025-01-01T10:00:00"),
        datetime.fromisoformat("2025-01-01T11:00:00"),
    ],
    values=[12.4, 12.7, 13.0, 12.9, 13.2, 13.5, 13.4, 13.7, 14.0, 13.9, 14.2, 14.5]
)
    
    
    #Récupération des données
    cfg: PaquetComplet = payload

    #-------------------------------
    # ----- TEMPO  -----------------
    # ------------------------------
    horizon = 1
    if cfg and cfg.Parametres_temporels and cfg.Parametres_temporels.horizon:
        horizon = max(1, int(cfg.Parametres_temporels.horizon))
    


    #-------------------------
    # ----- (X,y) ------------
    #-------------------------
    #Récupération des paramètres
    dates = cfg.Parametres_temporels.dates if (cfg and cfg.Parametres_temporels) else None
    pas_temporel = int(cfg.Parametres_temporels.pas_temporel) if (cfg and cfg.Parametres_temporels and cfg.Parametres_temporels.pas_temporel is not None) else 1
    portion_decoupage = float(cfg.Parametres_temporels.portion_decoupage) if (cfg and cfg.Parametres_temporels and cfg.Parametres_temporels.portion_decoupage is not None) else 0.8

    #Split sur les dates début et fin
    ts_filt, vals_filt = filter_series_by_dates(series.timestamps, series.values, dates)

    #On build X,y en tenseur pour toch
    X, y = build_supervised_tensors_with_step(
        vals_filt,
        horizon=horizon,
        step=pas_temporel,
    )

    if X.numel() == 0:
        def err():
            yield f"data: {json.dumps({'type':'error','message':'(X,y) vide après filtrage/découpage'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    # split séquentiel train/test via 'portion_decoupage'
    X_train, y_train, X_test, y_test = split_train_test(X, y, portion_train=portion_decoupage)

    # on entraîne sur le split train (garde X_test/y_test pour logs/éval plus tard)
    X, y = X_train, y_train



    #---------------------------------
    # ----- ARCHI --------------------
    #---------------------------------
   
    
    if cfg and cfg.Parametres_choix_reseau_neurones:
            if cfg.Parametres_choix_reseau_neurones.modele:
                model= cfg.Parametres_choix_reseau_neurones.modele.lower()
    
    #CLASS MODEL MLP
    if model =='mlp':
        hidden_size = 128
        nb_couches = 2
        dropout_rate = 0.0
        activation = "relu"
        use_batchnorm = False
        kernel_size = None
        stride = None
        padding = None
        if cfg and cfg.Parametres_archi_reseau_MLP:
            if cfg.Parametres_archi_reseau_MLP.hidden_size is not None:
                hidden_size = int(cfg.Parametres_archi_reseau_MLP.hidden_size)
            if cfg.Parametres_archi_reseau_MLP.nb_couches is not None:
                nb_couches = int(cfg.Parametres_archi_reseau_MLP.nb_couches)
            if cfg.Parametres_archi_reseau_MLP.dropout_rate is not None:
                dropout_rate = float(cfg.Parametres_archi_reseau_MLP.dropout_rate)
            if cfg.Parametres_archi_reseau_MLP.fonction_activation is not None:
                act_map = {
                    "ReLU": "relu",
                    "GELU": "gelu",
                    "tanh": "tanh",
                    "sigmoid": "sigmoid",
                    "leaky_relu": "leaky_relu",
                }
                activation = act_map.get(cfg.Parametres_archi_reseau_MLP.fonction_activation, "relu")

    #CLASS MODEL CNN
    if model =='cnn':
        hidden_size = 128
        nb_couches = 2
        dropout_rate = 0.0
        activation = "relu"
        use_batchnorm = False
        kernel_size = 3
        stride = 1
        padding = 1

        if cfg and cfg.Parametres_archi_reseau_CNN:
            if cfg.Parametres_archi_reseau_CNN.hidden_size is not None:
                hidden_size = int(cfg.Parametres_archi_reseau_CNN.hidden_size)
            
            if cfg.Parametres_archi_reseau_CNN.nb_couches is not None:
                nb_couches = int(cfg.Parametres_archi_reseau_CNN.nb_couches)

            if cfg.Parametres_archi_reseau_CNN.fonction_activation is not None:
                act_map = {
                    "ReLU": "relu",
                    "GELU": "gelu",
                    "tanh": "tanh",
                    "sigmoid": "sigmoid",
                    "leaky_relu": "leaky_relu",
                }
                activation = act_map.get(cfg.Parametres_archi_reseau_CNN.fonction_activation, "relu")
            
            if cfg.Parametres_archi_reseau_CNN.kernel_size is not None:
                kernel_size = int(cfg.Parametres_archi_reseau_CNN.kernel_size)
            
            if cfg.Parametres_archi_reseau_CNN.stride is not None:
                stride = int(cfg.Parametres_archi_reseau_CNN.stride)
            
            if cfg.Parametres_archi_reseau_CNN.padding is not None:
                padding = int(cfg.Parametres_archi_reseau_CNN.padding)
    #---------------------------------
    # ----- LOSS  --------------------
    # ---------------------------------
    loss_name = "mse"
    if cfg and cfg.Parametres_choix_loss_fct and cfg.Parametres_choix_loss_fct.fonction_perte:
        loss_name = cfg.Parametres_choix_loss_fct.fonction_perte.lower()


    # ----------------------------------
    # ----- OPTIM  --------------------
    # ----------------------------------
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


    #--------------------------------------
    # ----- TRAIN -------------------------
    #--------------------------------------
    epochs = 10
    batch_size = 64
    if cfg and cfg.Parametres_entrainement:
        if cfg.Parametres_entrainement.nb_epochs is not None:
            epochs = int(cfg.Parametres_entrainement.nb_epochs)
        if cfg.Parametres_entrainement.batch_size is not None:
            batch_size = int(cfg.Parametres_entrainement.batch_size)


    # --------------------------------------
    # ----- Device ------------------------
    # --------------------------------------
    device = "cpu"


    #--------------------------------------
    # ----- Lancement Entraînement --------
    #--------------------------------------
    def event_gen():

        if cfg and cfg.Parametres_choix_reseau_neurones:
            if cfg.Parametres_choix_reseau_neurones.modele:
                model= cfg.Parametres_choix_reseau_neurones.modele.lower()
        print(model)
        if model == "mlp":
            for msg in train_MLP(
                X, y,
                hidden_size=hidden_size,
                nb_couches=nb_couches,        
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
            ):
                yield f"data: {json.dumps(msg)}\n\n"



        if model =="cnn":
            print("CNN")

            for msg in train_CNN(
                X,y,
                hidden_size=hidden_size,
                nb_couches=nb_couches,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                use_batchnorm=use_batchnorm,
                loss_name=loss_name,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                device=device
            ):
                yield f"data: {json.dumps(msg)}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")





















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

