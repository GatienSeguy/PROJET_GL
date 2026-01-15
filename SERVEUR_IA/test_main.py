# ====================================
# IMPORTs
#export KMP_DUPLICATE_LIB_OK=TRUEDD
#export OMP_NUM_THREADS=1
#python -m uvicorn SERVEUR_IA.test_main:app --host 0.0.0.0 --port 8000 --reload
# ====================================
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException

import json
import torch
import numpy as np

# Train Modele
from .trains.training_MLP import train_MLP
from .trains.training_CNN import train_CNN
from .trains.training_LSTM import train_LSTM

# Test
from .test.test_testing import test_model_validation, compute_residual_std_from_validation
from .test.test_prediction_strategie import (
    predict_multistep,
    compare_all_strategies,
    PredictionStrategy, 
    PredictionConfig
)

from typing import List, Optional, Dict, Any, Tuple

from .classes import (
    TimeSeriesData,
    Parametres_archi_reseau_MLP,
    Parametres_archi_reseau_CNN,
    Parametres_archi_reseau_LSTM,
    Tx_choix_dataset,
    PaquetComplet,
    newDatasetRequest,
    deleteDatasetRequest,
    AddDatasetPacket
    )

from .test_fonctions_pour_main import (
    filter_series_by_dates,
    build_supervised_tensors_with_step,
    split_train_test,
    split_train_val_test,  # NOUVEAU
    sse,
    normalize_data,
    create_inverse_function
)

import os
import requests

DATA_SERVER_URL = os.getenv("DATA_SERVER_URL", "http://192.168.1.190:8001")
# DATA_SERVER_URL = os.getenv("DATA_SERVER_URL", "http://138.231.152.52:8001")

app = FastAPI()

last_config_tempo = None
last_config_TimeSeries = None
last_config_series = None

payload_json = {"timestamps": [], "values": []}


# ====================================
# CLASSE GESTION DATASETS
# ====================================
class DatasetManager:
    """Gère la communication avec le serveur Data"""
    
    def __init__(self, data_server_url: str = DATA_SERVER_URL):
        self.data_server_url = data_server_url
    
    def get_available_datasets(self) -> List[str]:
        try:
            url = f"{self.data_server_url}/datasets/list"
            response = requests.get(url, timeout=100)
            response.raise_for_status()
            data = response.json()
            return data.get("datasets", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur lors de la récupération des datasets: {str(e)}")
    
    def fetch_dataset(self, dataset_name: str, date_start: str, date_end: str) -> TimeSeriesData:
        try:
            url = f"{self.data_server_url}/datasets/fetch"
            payload = {"name": dataset_name, "dates": [date_start, date_end]}
            response = requests.post(url, json=payload, timeout=10000)
            response.raise_for_status()
            data_json = response.json()
            if "timestamps" not in data_json or "values" not in data_json:
                raise ValueError(f"Format de données invalide reçu du serveur Data")
            time_series = TimeSeriesData(**data_json)
            return time_series
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur lors de la récupération du dataset '{dataset_name}': {str(e)}")
        except ValueError as e:
            raise Exception(f"Format de données invalide: {str(e)}")


# ====================================
# CLASSE PRINCIPALE - PIPELINE 3 PHASES
# ====================================
class TrainingPipeline:
    """Pipeline d'entraînement avec 3 phases : train / validation / test prédictif"""

    def __init__(self, payload: PaquetComplet, payload_model: dict, time_series_data: Optional[TimeSeriesData] = None):
        self.cfg = payload
        self.payload_model = payload_model
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        # Variables d'état
        self.series = time_series_data
        self.cfg_model = None
        self.X = None
        self.y = None

        # NOUVEAU : 3 splits
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.split_info = None
        
        self.norm_params = None
        self.inverse_fn = None
        self.model_trained = None
        self.residual_std = None  # NOUVEAU : pour le halo
        
        # Ratios de split (paramétrable)
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

    # ====================================
    # 1) CHARGEMENT DES DONNÉES
    # ====================================
    def load_data(self, payload_json) -> TimeSeriesData:
        data_json = payload_json
        self.series = TimeSeriesData(**data_json)
        return self.series

    # ====================================
    # 2) CONFIGURATION DU MODÈLE
    # ====================================
    def setup_model_config(self):
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
    # 3) PRÉTRAITEMENT
    # ====================================
    def preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        horizon = 1
        if self.cfg and self.cfg.Parametres_temporels and self.cfg.Parametres_temporels.horizon:
            horizon = max(1, int(self.cfg.Parametres_temporels.horizon))
        
        X, y = build_supervised_tensors_with_step(
            self.series.values,
            horizon=horizon
        )
        
        if X.numel() == 0:
            raise ValueError("(X,y) vide")
        
        self.X = X
        self.y = y
        return X, y

    def normalize_data(self):
        all_data = torch.cat([self.X.flatten(), self.y.flatten()])
        all_data_normalized, self.norm_params = normalize_data(all_data, method="standardization")
        
        total_X = self.X.numel()
        X_normalized = all_data_normalized[:total_X].reshape(self.X.shape)
        y_normalized = all_data_normalized[total_X:].reshape(self.y.shape)
        
        self.inverse_fn = create_inverse_function(self.norm_params)
        
        self.X = X_normalized
        self.y = y_normalized

    # ====================================
    # NOUVEAU : SPLIT 3 PARTIES
    # ====================================
    def split_data_three_way(self):
        """Divise les données en train/validation/test (80/10/10 par défaut)"""
        
        # Récupérer les ratios depuis la config si présents
        if self.cfg and self.cfg.Parametres_temporels and self.cfg.Parametres_temporels.portion_decoupage:
            self.train_ratio = float(self.cfg.Parametres_temporels.portion_decoupage)
            remaining = 1.0 - self.train_ratio
            self.val_ratio = remaining / 2
            self.test_ratio = remaining / 2
        
        (self.X_train, self.y_train, 
         self.X_val, self.y_val, 
         self.X_test, self.y_test, 
         self.split_info) = split_train_val_test(
            self.X, self.y,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio
        )
        
        return self.split_info

    # ====================================
    # 4) RESHAPE POUR LE MODÈLE
    # ====================================
    def reshape_data_for_model(self, model_type: str):
        if model_type == "lstm":
            if self.X_train.ndim == 2:
                self.X_train = self.X_train.unsqueeze(-1)
            if self.X_val.ndim == 2 and self.X_val.numel() > 0:
                self.X_val = self.X_val.unsqueeze(-1)
            if self.X_test.ndim == 2 and self.X_test.numel() > 0:
                self.X_test = self.X_test.unsqueeze(-1)
        
        elif model_type == "cnn":
            if self.X_train.ndim == 2:
                self.X_train = self.X_train.unsqueeze(1)
            if self.X_val.ndim == 2 and self.X_val.numel() > 0:
                self.X_val = self.X_val.unsqueeze(1)
            if self.X_test.ndim == 2 and self.X_test.numel() > 0:
                self.X_test = self.X_test.unsqueeze(1)

    # ====================================
    # 5-8) CONFIGURATION (inchangé)
    # ====================================
    def setup_mlp_architecture(self) -> Dict[str, Any]:
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
            act_map = {"ReLU": "relu", "GELU": "gelu", "tanh": "tanh", "sigmoid": "sigmoid", "leaky_relu": "leaky_relu"}
            params["activation"] = act_map.get(self.cfg_model.fonction_activation, "relu")
        return params

    def setup_cnn_architecture(self) -> Dict[str, Any]:
        params = {
            "hidden_size": 128, "nb_couches": 2, "dropout_rate": 0.0,
            "activation": "relu", "use_batchnorm": False,
            "kernel_size": 3, "stride": 1, "padding": 1,
        }
        if self.cfg_model.hidden_size is not None:
            params["hidden_size"] = int(self.cfg_model.hidden_size)
        if self.cfg_model.nb_couches is not None:
            params["nb_couches"] = int(self.cfg_model.nb_couches)
        if self.cfg_model.fonction_activation is not None:
            act_map = {"ReLU": "relu", "GELU": "gelu", "tanh": "tanh", "sigmoid": "sigmoid", "leaky_relu": "leaky_relu"}
            params["activation"] = act_map.get(self.cfg_model.fonction_activation, "relu")
        if self.cfg_model.kernel_size is not None:
            params["kernel_size"] = int(self.cfg_model.kernel_size)
        if self.cfg_model.stride is not None:
            params["stride"] = int(self.cfg_model.stride)
        if self.cfg_model.padding is not None:
            params["padding"] = int(self.cfg_model.padding)
        return params

    def setup_lstm_architecture(self) -> Dict[str, Any]:
        params = {"hidden_size": 128, "nb_couches": 2, "bidirectional": False, "batch_first": True}
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
        if model_type == "mlp":
            return self.setup_mlp_architecture()
        elif model_type == "cnn":
            return self.setup_cnn_architecture()
        elif model_type == "lstm":
            return self.setup_lstm_architecture()
        else:
            raise ValueError(f"Modèle inconnu: {model_type}")

    def setup_loss(self) -> str:
        loss_name = "mse"
        if self.cfg and self.cfg.Parametres_choix_loss_fct and self.cfg.Parametres_choix_loss_fct.fonction_perte:
            loss_name = self.cfg.Parametres_choix_loss_fct.fonction_perte.lower()
        return loss_name

    def setup_optimizer(self) -> Tuple[str, float, float]:
        optimizer_name, learning_rate, weight_decay = "adam", 1e-3, 0.0
        if self.cfg and self.cfg.Parametres_optimisateur:
            if self.cfg.Parametres_optimisateur.optimisateur:
                optimizer_name = self.cfg.Parametres_optimisateur.optimisateur.lower()
            if self.cfg.Parametres_optimisateur.learning_rate is not None:
                learning_rate = float(self.cfg.Parametres_optimisateur.learning_rate)
            if self.cfg.Parametres_optimisateur.decroissance is not None:
                weight_decay = float(self.cfg.Parametres_optimisateur.decroissance)
        return optimizer_name, learning_rate, weight_decay

    def setup_training(self) -> Tuple[int, int]:
        epochs, batch_size = 10, 64
        if self.cfg and self.cfg.Parametres_entrainement:
            if self.cfg.Parametres_entrainement.nb_epochs is not None:
                epochs = int(self.cfg.Parametres_entrainement.nb_epochs)
            if self.cfg.Parametres_entrainement.batch_size is not None:
                batch_size = int(self.cfg.Parametres_entrainement.batch_size)
        return epochs, batch_size

    # ====================================
    # 9) ENTRAÎNEMENT
    # ====================================
    def create_training_generator(self, model_type, arch_params, loss_name, 
                                  optimizer_name, learning_rate, weight_decay, epochs, batch_size):
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
        """Phase 1 : Entraînement"""
        model_type = self.cfg.Parametres_choix_reseau_neurones.modele.lower()
        arch_params = self.setup_architecture(model_type)
        loss_name = self.setup_loss()
        optimizer_name, learning_rate, weight_decay = self.setup_optimizer()
        epochs, batch_size = self.setup_training()
        
        gen = self.create_training_generator(
            model_type, arch_params, loss_name,
            optimizer_name, learning_rate, weight_decay, epochs, batch_size
        )
        
        try:
            while True:
                msg = next(gen)
                yield msg
        except StopIteration as e:
            self.model_trained = e.value

    # ====================================
    # NOUVEAU : PHASE 2 - VALIDATION
    # ====================================
    def run_validation(self):
        """Phase 2 : Validation (teacher forcing) sur les 10% après train"""
        if self.model_trained is None:
            yield {"type": "warn", "message": "Modèle non récupéré (validation sautée)."}
            return
        
        if self.X_val is None or self.X_val.numel() == 0:
            yield {"type": "warn", "message": "Pas de données de validation."}
            return
        
        idx_val_start = self.split_info["idx_val_start"]
        
        val_y_true = []
        val_y_pred = []
        
        for evt in test_model_validation(
            self.model_trained,
            self.X_val,
            self.y_val,
            device=self.device,
            batch_size=256,
            inverse_fn=self.inverse_fn,
            idx_start=idx_val_start
        ):
            # Collecter les prédictions pour calculer residual_std
            if evt["type"] == "val_end":
                val_y_true = evt.get("all_true", [])
                val_y_pred = evt.get("all_predictions", [])
                
                # Calculer l'écart-type des résidus
                self.residual_std = compute_residual_std_from_validation(val_y_true, val_y_pred)
                evt["residual_std"] = self.residual_std
            
            yield evt

    # ====================================
    # PHASE 3 - TEST PRÉDICTIF (Toutes stratégies)
    # ====================================
    def run_prediction_test(self, strategy: str = "one_step"):
        """
        Phase 3 : Test prédictif avec choix de la stratégie.
        
        Stratégies disponibles:
        - "one_step": Prédiction 1 pas + recalibration immédiate (RECOMMANDÉ)
        - "recalibration": Prédiction récursive + recalibration tous les N pas
        - "recursive": Prédiction autorégressive pure (diverge vite)
        - "direct": Multi-horizon (nécessite modèle spécial)
        
        Basé sur Taieb & Hyndman (2014).
        """
        if self.model_trained is None:
            yield {"type": "warn", "message": "Modèle non récupéré (test pred sauté)."}
            return
        
        idx_test_start = self.split_info["idx_test_start"]
        n_test = self.split_info["n_test"]
        
        if n_test == 0:
            yield {"type": "warn", "message": "Pas de données pour test prédictif."}
            return
        
        # Taille de la fenêtre
        window_size = self.X.shape[1] if self.X.ndim >= 2 else 1
        
        # Vraies valeurs
        y_true_values = self.series.values[idx_test_start:idx_test_start + n_test]
        
        # Configuration selon la stratégie
        strategy_map = {
            "one_step": PredictionStrategy.ONE_STEP,
            "recalibration": PredictionStrategy.RECALIBRATION,
            "recursive": PredictionStrategy.RECURSIVE,
            "direct": PredictionStrategy.DIRECT,
        }
        
        pred_strategy = strategy_map.get(strategy, PredictionStrategy.RECALIBRATION)
        
        config = PredictionConfig(
            strategy=pred_strategy,
            recalib_every=min(10, max(1, n_test // 20)) if n_test > 20 else 5,
            max_horizon=min(50, n_test),
            direct_horizon=10,
            confidence_level=0.95
        )
        
        print(f"[PRED] Stratégie: {pred_strategy.value.upper()}")
        
        for evt in predict_multistep(
            model=self.model_trained,
            values=self.series.values,
            norm_stats=self.norm_params,
            window_size=window_size,
            n_steps=n_test,
            device=self.device,
            inverse_fn=self.inverse_fn,
            config=config,
            residual_std=self.residual_std,
            y_true=y_true_values,
            idx_start=idx_test_start,
        ):
            yield evt
    
    # ====================================
    # COMPARAISON DE TOUTES LES STRATÉGIES
    # ====================================
    def run_all_strategies_comparison(self):
        """
        Compare toutes les stratégies de prédiction.
        
        Retourne les métriques de chaque stratégie pour comparaison.
        """
        if self.model_trained is None:
            yield {"type": "warn", "message": "Modèle non récupéré."}
            return
        
        idx_test_start = self.split_info["idx_test_start"]
        n_test = self.split_info["n_test"]
        window_size = self.X.shape[1] if self.X.ndim >= 2 else 1
        y_true_values = self.series.values[idx_test_start:idx_test_start + n_test]
        
        yield {"type": "comparison_start", "n_strategies": 3}
        
        results = compare_all_strategies(
            model=self.model_trained,
            values=self.series.values,
            norm_stats=self.norm_params,
            window_size=window_size,
            n_steps=n_test,
            device=self.device,
            inverse_fn=self.inverse_fn,
            residual_std=self.residual_std,
            y_true=y_true_values,
            idx_start=idx_test_start,
            recalib_every=10,
            max_horizon=min(50, n_test),
        )
        
        yield {
            "type": "comparison_end",
            "results": results,
            "idx_start": idx_test_start,
            "n_steps": n_test
        }

    # ====================================
    # 11) ORCHESTRATION COMPLÈTE
    # ====================================
    def execute_full_pipeline(self):
        """Orchestre le pipeline complet : train → validation → test prédictif"""
        
        try:
            # Chargement des données
            self.load_data(payload_json=payload_json)
            
            # Configuration du modèle
            self.setup_model_config()
            
            # Prétraitement
            self.preprocess_data()
            self.normalize_data()
            
            # NOUVEAU : Split 3 parties
            split_info = self.split_data_three_way()
            
            # Reshape des données
            model_type = self.cfg.Parametres_choix_reseau_neurones.modele.lower()
            self.reshape_data_for_model(model_type)
            
            # Information de split (NOUVEAU format)
            yield sse({
                "type": "split_info",
                "n_train": split_info["n_train"],
                "n_val": split_info["n_val"],
                "n_test": split_info["n_test"],
                "idx_val_start": split_info["idx_val_start"],
                "idx_test_start": split_info["idx_test_start"],
            })
            
            # Envoyer la série complète pour l'affichage
            yield sse({
                "type": "serie_complete",
                "values": self.series.values,
            })
            
            # ========== PHASE 1 : ENTRAÎNEMENT ==========
            yield sse({"type": "phase", "phase": "train", "status": "start"})
            
            for msg in self.run_training():
                yield sse(msg)
            
            yield sse({"type": "phase", "phase": "train", "status": "end"})
            
            # ========== PHASE 2 : VALIDATION ==========
            yield sse({"type": "phase", "phase": "validation", "status": "start"})
            
            val_predictions = []
            val_true = []
            val_metrics = None
            
            for msg in self.run_validation():
                if msg["type"] == "val_pair":
                    val_predictions.append(msg["yhat"])
                    val_true.append(msg["y"])
                elif msg["type"] == "val_end":
                    val_metrics = msg.get("metrics")
                yield sse(msg)
            
            yield sse({"type": "phase", "phase": "validation", "status": "end"})
            
            # ========== PHASE 3 : TEST PRÉDICTIF ==========
            yield sse({"type": "phase", "phase": "prediction", "status": "start"})
            
            pred_predictions = []
            pred_low = []
            pred_high = []
            pred_true = []
            pred_metrics = None
            
            for msg in self.run_prediction_test():
                if msg["type"] == "pred_point":
                    pred_predictions.append(msg["yhat"])
                    pred_low.append(msg["low"])
                    pred_high.append(msg["high"])
                    if msg["y"] is not None:
                        pred_true.append(msg["y"])
                elif msg["type"] == "pred_end":
                    pred_metrics = msg.get("metrics")
                yield sse(msg)
            
            yield sse({"type": "phase", "phase": "prediction", "status": "end"})
            
            # ========== DONNÉES FINALES POUR L'AFFICHAGE ==========
            yield sse({
                "type": "final_plot_data",
                "series_complete": self.series.values,
                "val_predictions": val_predictions,
                "val_true": val_true,
                "pred_predictions": pred_predictions,
                "pred_low": pred_low,
                "pred_high": pred_high,
                "pred_true": pred_true,
                "idx_val_start": split_info["idx_val_start"],
                "idx_test_start": split_info["idx_test_start"],
                "val_metrics": val_metrics,
                "pred_metrics": pred_metrics,
            })
            
            yield sse({"type": "fin_pipeline", "done": 1})
        
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})


# ====================================
# ROUTES
# ====================================
@app.post("/datasets/info_all")
def proxy_get_dataset_list(payload: dict):
    print("Message reçu depuis UI :", payload)
    if payload.get("message") != "choix dataset":
        return {"status": "error", "message": "Message invalide"}
    try:
        url = f"{DATA_SERVER_URL}/datasets/info_all"
        response = requests.post(url, json=payload, timeout=1000)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("Exception côté IA :", e)
        return {"status": "error", "message": str(e)}


def extract_timestamps_values(obj):
    if isinstance(obj, dict):
        if "timestamps" in obj and "values" in obj:
            return obj["timestamps"], obj["values"]
        for v in obj.values():
            ts, vals = extract_timestamps_values(v)
            if ts is not None and vals is not None:
                return ts, vals
    elif isinstance(obj, list):
        for item in obj:
            ts, vals = extract_timestamps_values(item)
            if ts is not None and vals is not None:
                return ts, vals
    return None, None


@app.post("/datasets/fetch_dataset")
def proxy_fetch_dataset(payload: dict):
    global payload_json
    print("Message reçu depuis UI pour fetch_dataset :", payload)
    try:
        url = f"{DATA_SERVER_URL}/datasets/data_solo"
        response = requests.post(url, json=payload, timeout=1000)
        response.raise_for_status()
        data = response.json()
        print("Réponse brute DATA_SERVER :", data)
        ts, vals = extract_timestamps_values(data)
        if ts is None or vals is None:
            return {
                "status": "error",
                "message": "Format de données inattendu depuis DATA_SERVER"
            }
        payload_json = {"timestamps": ts, "values": vals}
        print(f"[IA] Dataset chargé dans payload_json : {len(ts)} points")
        return {"status": "success", "data": payload_json}
    except Exception as e:
        print("Exception côté IA lors du fetch_dataset :", e)
        return {"status": "error", "message": str(e)}



@app.post("/datasets/data_add_proxy")
def proxy_add_dataset(payload: dict):
    url = f"{DATA_SERVER_URL}/datasets/data_add"
    data_to_send = jsonable_encoder(payload)
    print("=== ENVOYÉ AU SERVEUR DATA ===")
    # print(data_to_send)

    try:
        response = requests.post(url, json=data_to_send, timeout=1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de connexion au serveur DATA: {e}")

    print("=== RÉPONSE SERVEUR DATA ===")
    # print("Status:", response.status_code)
    # print("Body  :", response.text)

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Erreur serveur DATA: {response.text}"
        )

    return response.json()


@app.post("/datasets/add_dataset")
def add_dataset_proxy(packet: AddDatasetPacket):
    url = f"{DATA_SERVER_URL}/datasets/add_dataset"
    print("PAQUET D'AJOUTER DS IA", packet)
    print("FORWARD URL =", url)

    try:
        out_json = packet.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Serialization error in IA: {repr(e)}")

    try:
        print("➡️ Forwarding to DATA server...")
        resp = requests.post(url, json=out_json, timeout=60)
        print("⬅️ DATA server status:", resp.status_code)
        print("⬅️ DATA server headers:", dict(resp.headers))
        print("⬅️ DATA server body (first 1000 chars):", resp.text[:1000])
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Dataset server unreachable: {repr(e)}")

    if not resp.ok:
        # propage l'erreur DATA
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)

    # ✅ ICI: parse JSON safe (sinon 500 muet)
    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "DATA server returned non-JSON response",
                "exception": repr(e),
                "status_code": resp.status_code,
                "content_type": resp.headers.get("content-type"),
                "body_preview": resp.text[:1000],
            },
        )



@app.post("/datasets/data_suppression_proxy")
def proxy_suppression_dataset(payload:deleteDatasetRequest):
    # print("Message reçu depuis UI :", payload.name)
    url = f"{DATA_SERVER_URL}/datasets/data_supression"
    response = requests.post(url, json=payload, timeout=1000)
    response.raise_for_status()
    return response.json()



@app.post("/train_full")
def training(payload: PaquetComplet, payload_model: dict):
    """Route d'entraînement complet avec le nouveau pipeline 3 phases"""
    pipeline = TrainingPipeline(payload, payload_model)
    return StreamingResponse(pipeline.execute_full_pipeline(), media_type="text/event-stream")


@app.get("/")
def root():
    return {"message": "Serveur IA actif !"}