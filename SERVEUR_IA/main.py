from fastapi import FastAPI

from fastapi.responses import StreamingResponse
import json 

from typing import Optional, Tuple, Literal, List
from pydantic import BaseModel, Field, conint, confloat
from datetime import date,datetime
import torch
import os, time

from .models.model_MLP import MLP
from .models.optim import make_loss,make_optimizer

#Train Modele
from .trains.training_MLP import train_MLP
from .trains.training_CNN import train_CNN
from .trains.training_LSTM import train_LSTM

from .test.testing import test_model

from .classes import (
    TimeSeriesData,
    Parametres_temporels,
    Parametres_choix_reseau_neurones,
    Parametres_choix_loss_fct,
    Parametres_optimisateur,
    Parametres_entrainement,
    Parametres_visualisation_suivi,
    Parametres_archi_reseau_MLP,
    Parametres_archi_reseau_CNN,
    Parametres_archi_reseau_LSTM,
    PaquetComplet
)

from .fonctions_pour_main import(
    build_supervised_tensors,
    _parse_any_datetime,
    filter_series_by_dates,
    build_supervised_tensors_with_step,
    split_train_test,
    sse
)
##### EN attendant c'est ici :


## EN LOCAL :
# uvicorn main:app --reload   


## EN SERVEUR SUR WIFI :
# 1- Faire sur terminal de mon mac :
#  ipconfig getifaddr en0  

# 2- Faire pour lancer le serveur
# 2) va à la racine du projet
# cd /Users/gatienseguy/Documents/VSCode/PROJET_GL
# touch SERVEUR_IA/__init__.py
# python -m uvicorn SERVEUR_IA.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /Users/gatienseguy/Documents/VSCode/PROJET_GL
#3 - à la txrx : metxztre : URL = ""http:// IP DE L'ORDI HOST DU SERVEUR :8000" "



app = FastAPI()

last_config_tempo = None
last_config_TimeSeries = None

last_config_series = None  




# ====================================
# ROUTES
# ====================================

@app.post("/train_full")
def training(payload: PaquetComplet,payload_model: dict):
# def training(payload: PaquetComplet):

#     series: TimeSeriesData = 
# En attendant la requête entre serveur de data on implémente en dur dans le code la serie temporelle
#     series = TimeSeriesData(
#     timestamps=[
#         datetime.fromisoformat("2025-01-01T00:00:00"),
#         datetime.fromisoformat("2025-01-01T01:00:00"),
#         datetime.fromisoformat("2025-01-01T02:00:00"),
#         datetime.fromisoformat("2025-01-01T03:00:00"),
#         datetime.fromisoformat("2025-01-01T04:00:00"),
#         datetime.fromisoformat("2025-01-01T05:00:00"),
#         datetime.fromisoformat("2025-01-01T06:00:00"),
#         datetime.fromisoformat("2025-01-01T07:00:00"),
#         datetime.fromisoformat("2025-01-01T08:00:00"),
#         datetime.fromisoformat("2025-01-01T09:00:00"),
#         datetime.fromisoformat("2025-01-01T10:00:00"),
#         datetime.fromisoformat("2025-01-01T11:00:00"),
#     ],
#     values=[12.4, 12.7, 13.0, 12.9, 13.2, 13.5, 13.4, 13.7, 14.0, 13.9, 14.2, 14.5]
# )
    

    json_path = "/Users/gatienseguy/Documents/VSCode/PROJET_GL/SERVEUR_DATA/Datas/EURO.json"  # ton fichier JSON existant

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    series = TimeSeriesData(**data)


    #Récupération des données
    cfg: PaquetComplet = payload

    # float
    # if cfg.Parametres_optimisateur and cfg.Parametres_optimisateur.learning_rate is not None:
    #     cfg.Parametres_optimisateur.learning_rate = float(cfg.Parametres_optimisateur.learning_rate)


    model_type = cfg.Parametres_choix_reseau_neurones.modele.lower()
    
    if model_type == "mlp":
        cfg_model = Parametres_archi_reseau_MLP(**payload_model)
    elif model_type == "cnn":
        cfg_model = Parametres_archi_reseau_CNN(**payload_model)
    elif model_type == "lstm":
        cfg_model = Parametres_archi_reseau_LSTM(**payload_model)
    else:
        raise ValueError(f"Modèle inconnu: {model_type}")


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


    if model_type == "lstm":
        if X.ndim == 2:
            X = X.unsqueeze(1)  # (B, T) -> (B, T, 1)
        if X_test.ndim == 2:
            X_test = X_test.unsqueeze(1)


    elif model_type == "cnn":
        if X.ndim == 2:
            X = X.unsqueeze(1)  # (B, seq_len) -> (B, 1, seq_len)
        if X_test.ndim == 2:
            X_test = X_test.unsqueeze(1)
    
    
    def split_info():
        msg = {
            "type": "info",
            "phase": "split",
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        }
        yield f"data: {json.dumps(msg)}\n\n"

    #---------------------------------
    # ----- ARCHI --------------------
    #---------------------------------
   
    ##### FAIRE UN FONCTION !!!!!!!
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
        if cfg_model.hidden_size is not None:
            hidden_size = int(cfg_model.hidden_size)
        if cfg_model.nb_couches is not None:
            nb_couches = int(cfg_model.nb_couches)
        if cfg_model.dropout_rate is not None:
            dropout_rate = float(cfg_model.dropout_rate)
        if cfg_model.fonction_activation is not None:
            act_map = {
                "ReLU": "relu",
                "GELU": "gelu",
                "tanh": "tanh",
                "sigmoid": "sigmoid",
                "leaky_relu": "leaky_relu",
            }
            activation = act_map.get(cfg_model.fonction_activation, "relu")

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

        if cfg_model.hidden_size is not None:
            hidden_size = int(cfg_model.hidden_size)
        
        if cfg_model.nb_couches is not None:
            nb_couches = int(cfg_model.nb_couches)

        if cfg_model.fonction_activation is not None:
            act_map = {
                "ReLU": "relu",
                "GELU": "gelu",
                "tanh": "tanh",
                "sigmoid": "sigmoid",
                "leaky_relu": "leaky_relu",
            }
            activation = act_map.get(cfg_model.fonction_activation, "relu")
        
        if cfg_model.kernel_size is not None:
            kernel_size = int(cfg_model.kernel_size)
        
        if cfg_model.stride is not None:
            stride = int(cfg_model.stride)
        
        if cfg_model.padding is not None:
            padding = int(cfg_model.padding)

    #CLASS MODEL LSTM
    if model =='lstm':
        #print('LSTMMMM')
        hidden_size = 128
        nb_couches = 2
        bidirectional = False
        batch_first = True

        if cfg_model.hidden_size is not None:
            hidden_size = int(cfg_model.hidden_size)
            
        if cfg_model.nb_couches is not None:
            nb_couches = int(cfg_model.nb_couches)

        if cfg_model.bidirectional is not None:
            bidirectional = bool(cfg_model.bidirectional)

        if cfg_model.batch_first is not None:
            batch_first = bool(cfg_model.batch_first)
         


            
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
        for msg in split_info():
            yield msg
        if cfg and cfg.Parametres_choix_reseau_neurones:
            if cfg.Parametres_choix_reseau_neurones.modele:
                model_name= cfg.Parametres_choix_reseau_neurones.modele.lower()
        
         # 2) construire le générateur d'entraînement
        if model_name == "mlp":
            gen = train_MLP(
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
            )
        elif model_name == "cnn":
            gen = train_CNN(
                X, y,
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
                device=device,
            )
        elif model_name == "lstm":
            gen = train_LSTM(
                X, y,
                hidden_size=hidden_size,
                nb_couches=nb_couches,
                bidirectional=bidirectional,
                batch_first=True,
                loss_name=loss_name,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                device=device,
            )
        else:
            yield sse({"type":"error","message": f"Modèle inconnu: {model_name}"})
            return
        
        model_trained = None
        try:
            while True:
                msg = next(gen)
                
                yield sse(msg)
        except StopIteration as e:
            # Le modèle est retourné via StopIteration.value
            model_trained = e.value
            # print(f"DEBUG - model_trained type: {type(model_trained)}, is None: {model_trained is None}")

        
        # 4) Test en streaming : y / ŷ par paire + métriques finales
        if model_trained is not None:
            print(f"[DÉBUT TEST] Modèle: {type(model_trained).__name__}")

            for evt in test_model(
                model_trained, X_test, y_test,
                device=device,
                batch_size=256,
                inverse_fn=None,
            ):
                yield f"data: {json.dumps(evt)}\n\n"
                #print(f"data: {json.dumps(evt)}\n\n")


        else:
            yield sse({"type":"warn","message":"Modèle non récupéré (test sauté)."})
    
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

