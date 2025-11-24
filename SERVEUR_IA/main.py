# ====================================
# IMPORTs
# ====================================
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json 
import torch

# Train Modele
from .trains.training_MLP import train_MLP
from .trains.training_CNN import train_CNN
from .trains.training_LSTM import train_LSTM

from .test.testing import test_model

from .launcher_serveur import json_path

from typing import List, Optional, Dict, Any, Tuple

from .classes import (
    TimeSeriesData,
    Parametres_archi_reseau_MLP,
    Parametres_archi_reseau_CNN,
    Parametres_archi_reseau_LSTM,
    Tx_choix_dataset,
    PaquetComplet)

from .fonctions_pour_main import(
    filter_series_by_dates,
    build_supervised_tensors_with_step,
    split_train_test,
    sse,
    normalize_data,
    create_inverse_function
)

import os
import requests

DATA_SERVER_URL = os.getenv("DATA_SERVER_URL", "http://192.168.27.66:8001")

# python -m uvicorn SERVEUR_IA.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /Users/gatienseguy/Documents/VSCode/PROJET_GL

app = FastAPI()

last_config_tempo = None
last_config_TimeSeries = None
last_config_series = None  

payload_json = {}
# ====================================
# CLASSE GESTION DATASETS
# ====================================
class DatasetManager:
    """Gère la communication avec le serveur Data"""
    
    def __init__(self, data_server_url: str = DATA_SERVER_URL):
        """Initialise le gestionnaire de datasets"""
        self.data_server_url = data_server_url
    
    def get_available_datasets(self) -> List[str]:
        """
        Récupère la liste des noms de datasets disponibles depuis le serveur Data
        """
        try:
            url = f"{self.data_server_url}/datasets/list"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get("datasets", [])
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur lors de la récupération des datasets: {str(e)}")
    
    def fetch_dataset(self, dataset_name: str, date_start: str, date_end: str) -> TimeSeriesData:
        """
        Récupère un dataset du serveur Data pour une plage de dates donnée
        """
        try:
            url = f"{self.data_server_url}/datasets/fetch"
            
            payload = {
                "name": dataset_name,
                "dates": [date_start, date_end]
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            data_json = response.json()
            
            # Validation et conversion en TimeSeriesData
            if "timestamps" not in data_json or "values" not in data_json:
                raise ValueError(f"Format de données invalide reçu du serveur Data")
            
            time_series = TimeSeriesData(**data_json)
            return time_series
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur lors de la récupération du dataset '{dataset_name}': {str(e)}")
        except ValueError as e:
            raise Exception(f"Format de données invalide: {str(e)}")


# ====================================
# CLASSE PRINCIPALE
# ====================================
class TrainingPipeline:
    """Pipeline d'entraînement pour les réseaux de neurones"""

    def __init__(self, payload: PaquetComplet, payload_model: dict, time_series_data: Optional[TimeSeriesData] = None):
        """Initialise le pipeline avec les configurations"""
        self.cfg = payload
        self.payload_model = payload_model

        # Détection du device cpu gpu ou mps
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "mps"

        # Variables d'état
        self.series = time_series_data
        self.cfg_model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.norm_params = None
        self.inverse_fn = None
        self.model_trained = None
        
    # ====================================
    # 1) CHARGEMENT DES DONNÉES
    # ====================================
    def load_data(self) -> TimeSeriesData:
        """
        Charge les données
        
        Si un dataset a été passé au constructeur, l'utilise.
        Sinon, charge depuis le fichier JSON par défaut.
        """
        if self.series is not None:
            # Dataset déjà fourni (récupéré via /datasets/fetch)
            return self.series
        
        # Fallback : charger depuis fichier JSON TEMPORAIRE
        # json_file_path = "/Users/gatienseguy/Documents/VSCode/PROJET_GL/SERVEUR_DATA/datasets/EURO.json"
        
        # with open(json_file_path, 'r') as f:
            data_json = json.load(f)
        
        data_json = payload_json
        self.series = TimeSeriesData(**data_json)
        return self.series
    
    # ====================================
    # 2) CONFIGURATION DU MODÈLE
    # ====================================
    def setup_model_config(self):
        """Configure le modèle en fonction du type sélectionné"""
        model_type = self.cfg.Parametres_choix_reseau_neurones.modele.lower()
        
        if model_type == "mlp":
            self.cfg_model = Parametres_archi_reseau_MLP(**self.payload_model)
        elif model_type == "cnn":
            self.cfg_model = Parametres_archi_reseau_CNN(**self.payload_model)
        elif model_type == "lstm":
            self.cfg_model = Parametres_archi_reseau_LSTM(**self.payload_model)
        else:
            raise ValueError(f"Modèle inconnu: {model_type}")
        
        return self.cfg_model
    
    # ====================================
    # 3) EXTRACTION ET FILTRAGE DES DONNÉES
    # ====================================
    def preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prétraite les données : filtrage, construction de (X,y) et normalisation"""
        
        # Récupération des paramètres temporels
        # dates = self.cfg.Parametres_temporels.dates if (self.cfg and self.cfg.Parametres_temporels) else None

        # pas_temporel = int(self.cfg.Parametres_temporels.pas_temporel) if (self.cfg and self.cfg.Parametres_temporels and self.cfg.Parametres_temporels.pas_temporel is not None) else 1
        horizon = 1
        if self.cfg and self.cfg.Parametres_temporels and self.cfg.Parametres_temporels.horizon:
            horizon = max(1, int(self.cfg.Parametres_temporels.horizon))
        
        # Filtrage par dates
        # ts_filt, vals_filt = filter_series_by_dates(
        #     self.series.timestamps, 
        #     self.series.values, 
        #     dates
        # )
        
        # Construction des tenseurs (X, y)
        X, y = build_supervised_tensors_with_step(
            self.series.values,
            horizon=horizon
        )

        
        if X.numel() == 0:
            raise ValueError("(X,y) vide après filtrage/découpage")
        
        self.X = X
        self.y = y
        
        return X, y
    
    def normalize_data(self):
        """Normalise les données (X, y)"""
        all_data = torch.cat([self.X.flatten(), self.y.flatten()])
        
        # Normalisation (standardization)
        all_data_normalized, self.norm_params = normalize_data(all_data, method="standardization")
        
        # Reconstruction de X et y normalisés
        total_X = self.X.numel()
        X_normalized = all_data_normalized[:total_X].reshape(self.X.shape)
        y_normalized = all_data_normalized[total_X:].reshape(self.y.shape)
        
        # Créer la fonction inverse pour dénormalisation
        self.inverse_fn = create_inverse_function(self.norm_params)
        
        self.X = X_normalized
        self.y = y_normalized
    
    def split_data(self):
        """Divise les données en train/test"""
        portion_decoupage = float(self.cfg.Parametres_temporels.portion_decoupage) if (self.cfg and self.cfg.Parametres_temporels and self.cfg.Parametres_temporels.portion_decoupage is not None) else 0.8
        
        self.X_train, self.y_train, self.X_test, self.y_test = split_train_test(
            self.X, self.y, 
            portion_train=portion_decoupage
        )
    
    # ====================================
    # 4) ADAPTATION DU SHAPE POUR LE MODÈLE
    # ====================================
    def reshape_data_for_model(self, model_type: str):
        """Adapte la forme des données selon le type de modèle"""
        if model_type == "lstm":
            if self.X_train.ndim == 2:
                self.X_train = self.X_train.unsqueeze(1)  # (B, T) -> (B, T, 1)
            if self.X_test.ndim == 2:
                self.X_test = self.X_test.unsqueeze(1)
        
        elif model_type == "cnn":
            if self.X_train.ndim == 2:
                self.X_train = self.X_train.unsqueeze(1)  # (B, seq_len) -> (B, 1, seq_len)
            if self.X_test.ndim == 2:
                self.X_test = self.X_test.unsqueeze(1)
    
    # ====================================
    # 5) CONFIGURATION DE L'ARCHITECTURE
    # ====================================
    def setup_mlp_architecture(self) -> Dict[str, Any]:
        """Configure les paramètres de l'architecture MLP"""
        params = {
            "hidden_size": 128,
            "nb_couches": 2,
            "dropout_rate": 0.0,
            "activation": "relu",
            "use_batchnorm": False,
        }
        
        if self.cfg_model.hidden_size is not None:
            params["hidden_size"] = int(self.cfg_model.hidden_size)
        if self.cfg_model.nb_couches is not None:
            params["nb_couches"] = int(self.cfg_model.nb_couches)
        if self.cfg_model.dropout_rate is not None:
            params["dropout_rate"] = float(self.cfg_model.dropout_rate)
        if self.cfg_model.fonction_activation is not None:
            act_map = {
                "ReLU": "relu",
                "GELU": "gelu",
                "tanh": "tanh",
                "sigmoid": "sigmoid",
                "leaky_relu": "leaky_relu",
            }
            params["activation"] = act_map.get(self.cfg_model.fonction_activation, "relu")
        
        return params
    
    def setup_cnn_architecture(self) -> Dict[str, Any]:
        """Configure les paramètres de l'architecture CNN"""
        params = {
            "hidden_size": 128,
            "nb_couches": 2,
            "dropout_rate": 0.0,
            "activation": "relu",
            "use_batchnorm": False,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        }
        
        if self.cfg_model.hidden_size is not None:
            params["hidden_size"] = int(self.cfg_model.hidden_size)
        if self.cfg_model.nb_couches is not None:
            params["nb_couches"] = int(self.cfg_model.nb_couches)
        if self.cfg_model.fonction_activation is not None:
            act_map = {
                "ReLU": "relu",
                "GELU": "gelu",
                "tanh": "tanh",
                "sigmoid": "sigmoid",
                "leaky_relu": "leaky_relu",
            }
            params["activation"] = act_map.get(self.cfg_model.fonction_activation, "relu")
        if self.cfg_model.kernel_size is not None:
            params["kernel_size"] = int(self.cfg_model.kernel_size)
        if self.cfg_model.stride is not None:
            params["stride"] = int(self.cfg_model.stride)
        if self.cfg_model.padding is not None:
            params["padding"] = int(self.cfg_model.padding)
        
        return params
    
    def setup_lstm_architecture(self) -> Dict[str, Any]:
        """Configure les paramètres de l'architecture LSTM"""
        params = {
            "hidden_size": 128,
            "nb_couches": 2,
            "bidirectional": False,
            "batch_first": True,
        }
        
        if self.cfg_model.hidden_size is not None:
            params["hidden_size"] = int(self.cfg_model.hidden_size)
        if self.cfg_model.nb_couches is not None:
            params["nb_couches"] = int(self.cfg_model.nb_couches)
        if self.cfg_model.bidirectional is not None:
            params["bidirectional"] = bool(self.cfg_model.bidirectional)
        if self.cfg_model.batch_first is not None:
            params["batch_first"] = bool(self.cfg_model.batch_first)
        
        return params
    
    def setup_architecture(self, model_type: str) -> Dict[str, Any]:
        """Sélectionne la configuration d'architecture appropriée"""
        if model_type == "mlp":
            return self.setup_mlp_architecture()
        elif model_type == "cnn":
            return self.setup_cnn_architecture()
        elif model_type == "lstm":
            return self.setup_lstm_architecture()
        else:
            raise ValueError(f"Modèle inconnu: {model_type}")
    
    # ====================================
    # 6) CONFIGURATION DE LA LOSS
    # ====================================
    def setup_loss(self) -> str:
        """Configure la fonction de perte"""
        loss_name = "mse"
        if self.cfg and self.cfg.Parametres_choix_loss_fct and self.cfg.Parametres_choix_loss_fct.fonction_perte:
            loss_name = self.cfg.Parametres_choix_loss_fct.fonction_perte.lower()
        return loss_name
    
    # ====================================
    # 7) CONFIGURATION DE L'OPTIMISEUR
    # ====================================
    def setup_optimizer(self) -> Tuple[str, float, float]:
        """Configure l'optimiseur, learning rate et weight decay"""
        optimizer_name = "adam"
        learning_rate = 1e-3
        weight_decay = 0.0
        
        if self.cfg and self.cfg.Parametres_optimisateur:
            if self.cfg.Parametres_optimisateur.optimisateur:
                optimizer_name = self.cfg.Parametres_optimisateur.optimisateur.lower()
            if self.cfg.Parametres_optimisateur.learning_rate is not None:
                learning_rate = float(self.cfg.Parametres_optimisateur.learning_rate)
            if self.cfg.Parametres_optimisateur.decroissance is not None:
                weight_decay = float(self.cfg.Parametres_optimisateur.decroissance)
        
        return optimizer_name, learning_rate, weight_decay
    
    # ====================================
    # 8) CONFIGURATION DE L'ENTRAÎNEMENT
    # ====================================
    def setup_training(self) -> Tuple[int, int]:
        """Configure le nombre d'epochs et batch size"""
        epochs = 10
        batch_size = 64
        
        if self.cfg and self.cfg.Parametres_entrainement:
            if self.cfg.Parametres_entrainement.nb_epochs is not None:
                epochs = int(self.cfg.Parametres_entrainement.nb_epochs)
            if self.cfg.Parametres_entrainement.batch_size is not None:
                batch_size = int(self.cfg.Parametres_entrainement.batch_size)
        
        return epochs, batch_size
    
    # ====================================
    # 9) ENTRAÎNEMENT
    # ====================================
    def create_training_generator(self, 
                                  model_type: str,
                                  arch_params: Dict[str, Any],
                                  loss_name: str,
                                  optimizer_name: str,
                                  learning_rate: float,
                                  weight_decay: float,
                                  epochs: int,
                                  batch_size: int):
        """Crée le générateur d'entraînement en fonction du type de modèle"""
        
        if model_type == "mlp":
            return train_MLP(
                self.X_train, self.y_train,
                hidden_size=arch_params["hidden_size"],
                nb_couches=arch_params["nb_couches"],
                dropout_rate=arch_params["dropout_rate"],
                activation=arch_params["activation"],
                use_batchnorm=arch_params["use_batchnorm"],
                loss_name=loss_name,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                device=self.device,
            )
        
        elif model_type == "cnn":
            return train_CNN(
                self.X_train, self.y_train,
                hidden_size=arch_params["hidden_size"],
                nb_couches=arch_params["nb_couches"],
                kernel_size=arch_params["kernel_size"],
                stride=arch_params["stride"],
                padding=arch_params["padding"],
                activation=arch_params["activation"],
                use_batchnorm=arch_params["use_batchnorm"],
                loss_name=loss_name,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                device=self.device,
            )
        
        elif model_type == "lstm":
            return train_LSTM(
                self.X_train, self.y_train,
                hidden_size=arch_params["hidden_size"],
                nb_couches=arch_params["nb_couches"],
                bidirectional=arch_params["bidirectional"],
                batch_first=arch_params["batch_first"],
                loss_name=loss_name,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                device=self.device,
            )
        
        else:
            raise ValueError(f"Modèle inconnu: {model_type}")
    
    def run_training(self):
        """Exécute l'entraînement et retourne le modèle entraîné"""
        model_type = self.cfg.Parametres_choix_reseau_neurones.modele.lower()
        
        # Configuration
        arch_params = self.setup_architecture(model_type)
        loss_name = self.setup_loss()
        optimizer_name, learning_rate, weight_decay = self.setup_optimizer()
        epochs, batch_size = self.setup_training()
        
        # Créer le générateur d'entraînement
        gen = self.create_training_generator(
            model_type,
            arch_params,
            loss_name,
            optimizer_name,
            learning_rate,
            weight_decay,
            epochs,
            batch_size
        )
        
        # Exécuter l'entraînement
        try:
            while True:
                msg = next(gen)
                yield msg
        except StopIteration as e:
            self.model_trained = e.value
    
    # ====================================
    # 10) TEST
    # ====================================
    def run_testing(self):
        """Exécute les tests sur le modèle entraîné"""
        if self.model_trained is None:
            yield {"type": "warn", "message": "Modèle non récupéré (test sauté)."}
            return
        
        print(f"[DÉBUT TEST] Modèle: {type(self.model_trained).__name__}")
        
        for evt in test_model(
            self.model_trained, 
            self.X_test, 
            self.y_test,
            device=self.device,
            batch_size=256,
            inverse_fn=self.inverse_fn,
        ):
            yield evt
    
    # ====================================
    # 11) ORCHESTRATION COMPLÈTE
    # ====================================
    def execute_full_pipeline(self):
        """Orchestre le pipeline complet d'entraînement et test"""
        
        try:
            # Chargement des données
            self.load_data()
            
            # Configuration du modèle
            self.setup_model_config()
            
            # Prétraitement
            self.preprocess_data()
            self.normalize_data()
            self.split_data()
            
            # Reshape des données
            model_type = self.cfg.Parametres_choix_reseau_neurones.modele.lower()
            self.reshape_data_for_model(model_type)
            
            # Information de split
            msg = {
                "type": "info",
                "phase": "split",
                "n_train": int(self.X_train.shape[0]),
                "n_test": int(self.X_test.shape[0]),
            }
            yield f"data: {json.dumps(msg)}\n\n"
            
            # Entraînement
            for msg in self.run_training():
                yield sse(msg)
            
            # Test
            for msg in self.run_testing():
                yield f"data: {json.dumps(msg)}\n\n"
        
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})


# ====================================
# ROUTES
# ====================================

# ====================================
# ROUTES - ENTRAÎNEMENT COMPLET
# ====================================

@app.post("/train_full")
def training(payload: PaquetComplet, payload_model: dict):
    """Route d'entraînement complet"""
    pipeline = TrainingPipeline(payload, payload_model)
    return StreamingResponse(pipeline.execute_full_pipeline(), media_type="text/event-stream")


# ====================================
# ROUTES - GESTION DATASETS
# ====================================

# @app.get("/datasets/list")
# def get_datasets_list():
#     """
#     Récupère la liste de tous les datasets disponibles
#     Endpoint pour la UI permettant d'afficher les datasets disponibles
#     """
#     try:
#         manager = DatasetManager()
#         datasets = manager.get_available_datasets()
        
#         return {
#             "status": "success",
#             "datasets": datasets
#         }
    
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": str(e)
#         }


# @app.post("/datasets/fetch")
# def fetch_dataset(request_payload: dict):
#     """
#     Récupère un dataset spécifique avec une plage de dates
#     Endpoint pour la UI permettant de charger un dataset avec filtrage de dates
#     """
#     try:
#         # Validation de la requête
#         if "name" not in request_payload:
#             return {
#                 "status": "error",
#                 "message": "Champ 'name' manquant"
#             }
        
#         if "dates" not in request_payload or len(request_payload["dates"]) != 2:
#             return {
#                 "status": "error",
#                 "message": "Champ 'dates' manquant ou invalide (doit contenir [date_debut, date_fin])"
#             }
        
#         dataset_name = request_payload["name"]
#         date_start, date_end = request_payload["dates"]
        
#         # Récupération du dataset
#         manager = DatasetManager()
#         time_series = manager.fetch_dataset(dataset_name, date_start, date_end)
        
#         return {
#             "status": "success",
#             "data": {
#                 "timestamps": time_series.timestamps,
#                 "values": time_series.values
#             }
#         }
    
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": str(e)
#         }



# Route proxy pour récupérer la liste des datasets depuis le serveur DATA 
# UI -> SERVEUR_IA -> SERVEUR_DATA -> SERVEUR_IA -> UI
@app.post("/datasets/info_all")
def proxy_get_dataset_list(payload: dict):
    print("Message reçu depuis UI :", payload)

    if payload.get("message") != "choix dataset":
        return {"status": "error", "message": "Message invalide"}

    try:
        url = f"{DATA_SERVER_URL}/datasets/info_all"
        # print("salaupard")
        # ENVOI DU PAYLOAD AU SERVEUR DATA
        response = requests.post(url, json=payload, timeout=10)
        # print(payload)
        response.raise_for_status()
        return response.json()

    except Exception as e:
        print("Exception côté IA :", e)
        return {"status": "error", "message": str(e)}
    

@app.post("/datasets/fetch_dataset")
def proxy_fetch_dataset(payload: dict):
    print("Message reçu depuis UI pour fetch_dataset :", payload)

    try:
        url = f"{DATA_SERVER_URL}/datasets/data_solo"
        
        # ENVOI DU PAYLOAD AU SERVEUR DATA
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        # print(response.json())
        payload_json = response.json()
        return "coucou"

    except Exception as e:
        print("Exception côté IA lors du fetch_dataset :", e)
        return {"status": "error", "message": str(e)}


# ====================================
# ROUTES - CHECK SERVEUR IA
# ====================================

@app.get("/")
def root():
    """Vérification de l'état du serveur"""
    response = {"message": "Serveur IA actif !"}
    return response