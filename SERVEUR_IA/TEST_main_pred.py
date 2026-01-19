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

# Test - test_model_validation remplacé par implémentation inline
# from .test.test_testing import test_model_validation, compute_residual_std_from_validation
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
# VARIABLE GLOBALE POUR LE MODÈLE ENTRAÎNÉ
# ====================================
trained_model_state = {
    "model": None,
    "norm_params": None,
    "inverse_fn": None,
    "window_size": None,
    "residual_std": None,
    "model_type": None,
    "device": "cpu",
    "is_trained": False,
}


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
        
        self.stop_flag = False

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
        
        # Taille de fenêtre depuis les paramètres ou valeur par défaut
        window_len = 15  # valeur par défaut
        if self.cfg and self.cfg.Parametres_temporels:
            # Debug: afficher tout le contenu de Parametres_temporels
            print(f"[PREPROCESS DEBUG] Parametres_temporels = {self.cfg.Parametres_temporels}")
            
            # Essayer plusieurs méthodes pour récupérer window_size
            window_size_param = getattr(self.cfg.Parametres_temporels, 'window_size', None)
            
            # Si c'est un dict (via model_dump), essayer aussi
            if window_size_param is None:
                try:
                    params_dict = self.cfg.Parametres_temporels.model_dump()
                    window_size_param = params_dict.get('window_size')
                    print(f"[PREPROCESS DEBUG] window_size from model_dump = {window_size_param}")
                except:
                    pass
            
            if window_size_param is not None and window_size_param > 0:
                window_len = int(window_size_param)
                print(f"[PREPROCESS] Using window_size from config: {window_len}")
            else:
                print(f"[PREPROCESS] window_size_param={window_size_param}, using default: {window_len}")
        
        print(f"[PREPROCESS] window_size={window_len}, horizon={horizon}")
        
        X, y = build_supervised_tensors_with_step(
            self.series.values,
            window_len=window_len,
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
        print(f"[DEBUG RESHAPE] model_type={model_type}")
        print(f"[DEBUG RESHAPE] AVANT - X_train.shape={self.X_train.shape}")
        
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
        
        print(f"[DEBUG RESHAPE] APRÈS - X_train.shape={self.X_train.shape}")

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
        # batch_first=True est requis pour notre pipeline
        params = {"hidden_size": 128, "nb_couches": 2, "bidirectional": False, "batch_first": True}
        if self.cfg_model.hidden_size is not None:
            params["hidden_size"] = int(self.cfg_model.hidden_size)
        if self.cfg_model.nb_couches is not None:
            params["nb_couches"] = int(self.cfg_model.nb_couches)
        if self.cfg_model.bidirectional is not None:
            params["bidirectional"] = bool(self.cfg_model.bidirectional)
        # Toujours forcer batch_first=True pour la compatibilité du pipeline
        params["batch_first"] = True
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
            # FORCE le reshape pour CNN : (B, T) -> (B, 1, T)
            X_cnn = self.X_train
            if X_cnn.ndim == 2:
                X_cnn = X_cnn.unsqueeze(1)
            print(f"[CNN CALL] X_cnn.shape={X_cnn.shape}, y.shape={self.y_train.shape}")
            
            return train_CNN(
                X_cnn, self.y_train,
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
            # FORCE le reshape pour LSTM : (B, T) -> (B, T, 1)
            X_lstm = self.X_train
            if X_lstm.ndim == 2:
                X_lstm = X_lstm.unsqueeze(-1)
            print(f"[LSTM CALL] X_lstm.shape={X_lstm.shape}, y.shape={self.y_train.shape}")
            
            return train_LSTM(
                X_lstm, self.y_train,
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
        global stop_training_flag

        model_type = self.cfg.Parametres_choix_reseau_neurones.modele.lower()
        arch_params = self.setup_architecture(model_type)
        self.arch_params = arch_params  # Sauvegarder pour utilisation ultérieure
        loss_name = self.setup_loss()
        optimizer_name, learning_rate, weight_decay = self.setup_optimizer()
        epochs, batch_size = self.setup_training()
        
        gen = self.create_training_generator(
            model_type, arch_params, loss_name,
            optimizer_name, learning_rate, weight_decay, epochs, batch_size
        )
        
        try:
            while True:
                if stop_training_flag:
                    stop_training_flag = False
                    yield {"type": "stopped", "message": "Entraînement arrêté"}
                    break

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
        
        # CORRECTION: convertir l'indice de fenêtre en indice de série
        window_size = self.X.shape[1] if self.X.ndim >= 2 else 1
        idx_val_start_windows = self.split_info["idx_val_start"]
        idx_val_start_series = idx_val_start_windows + window_size
        
        model = self.model_trained
        model_type = trained_model_state.get("model_type", "mlp")
        device = self.device
        
        n_points = self.X_val.shape[0]
        yield {"type": "val_start", "n_points": n_points, "dims": 1, "idx_start": idx_val_start_series}
        
        all_predictions = []
        all_true = []
        
        model.eval()
        with torch.no_grad():
            for i in range(n_points):
                x_sample = self.X_val[i:i+1]  # Déjà reshapé: (1, T) pour MLP, (1, 1, T) pour CNN, (1, T, 1) pour LSTM
                y_true_norm = self.y_val[i].item()
                
                # Debug pour comprendre la shape
                if i == 0:
                    print(f"[VAL DEBUG] model_type={model_type}, x_sample.shape={x_sample.shape}")
                
                # X_val est DÉJÀ reshapé par reshape_data_for_model()
                # Donc on n'a pas besoin de faire un autre unsqueeze !
                x_input = x_sample
                
                if i == 0:
                    print(f"[VAL DEBUG] x_input.shape={x_input.shape}")
                
                x_input = x_input.to(device)
                output = model(x_input)
                
                # Extraire la prédiction selon le type de modèle
                if model_type == "lstm":
                    if output.ndim == 3:
                        y_pred_norm = output[0, -1, 0].cpu().item()
                    else:
                        y_pred_norm = output.flatten()[0].cpu().item()
                elif model_type == "cnn":
                    if output.ndim == 3:
                        y_pred_norm = output[0, 0, -1].cpu().item()
                    else:
                        y_pred_norm = output.flatten()[-1].cpu().item()
                else:
                    y_pred_norm = output.flatten()[0].cpu().item()
                
                # Dénormaliser
                y_pred = self.inverse_fn(torch.tensor([y_pred_norm])).item()
                y_true = self.inverse_fn(torch.tensor([y_true_norm])).item()
                
                all_predictions.append(y_pred)
                all_true.append(y_true)
                
                yield {"type": "val_pair", "idx": idx_val_start_series + i, "y": y_true, "yhat": y_pred}
        
        # Calculer les métriques
        y_true_arr = np.array(all_true)
        y_pred_arr = np.array(all_predictions)
        mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
        
        # Calculer l'écart-type des résidus
        self.residual_std = float(np.std(y_true_arr - y_pred_arr))
        
        # Format attendu par l'interface
        yield {
            "type": "val_end",
            "metrics": {
                "overall_mean": {"MSE": mse, "MAE": mae},
                "mse": mse,
                "mae": mae
            },
            "all_true": all_true,
            "all_predictions": all_predictions,
            "residual_std": self.residual_std
        }

    # ====================================
    # PHASE 3 - TEST PRÉDICTIF (Toutes stratégies)
    # ====================================
    def run_prediction_test(self, strategy: str = "one_step"):
        """
        Phase 3 : Test prédictif avec reshape correct pour LSTM/CNN.
        """
        if self.model_trained is None:
            yield {"type": "warn", "message": "Modèle non récupéré (test pred sauté)."}
            return
        
        # CORRECTION: idx_test_start est l'indice dans les fenêtres X
        # L'indice réel dans la série est idx_test_start + window_size
        window_size = self.X.shape[1] if self.X.ndim >= 2 else 1
        idx_test_start_windows = self.split_info["idx_test_start"]
        n_test = self.split_info["n_test"]
        
        # Indice réel dans la série originale
        idx_test_start_series = idx_test_start_windows + window_size
        
        if n_test == 0:
            yield {"type": "warn", "message": "Pas de données pour test prédictif."}
            return
        
        y_true_values = self.series.values[idx_test_start_series:idx_test_start_series + n_test]
        model_type = self.cfg.Parametres_choix_reseau_neurones.modele.lower()
        
        print(f"[PRED] model_type={model_type}, window_size={window_size}, n_test={n_test}")
        print(f"[PRED] idx_test_start_windows={idx_test_start_windows}, idx_test_start_series={idx_test_start_series}")
        
        # Implémentation inline de la prédiction avec reshape correct
        yield {
            "type": "pred_start",
            "n_steps": n_test,
            "strategy": strategy,
            "idx_start": idx_test_start_series,  # Indice dans la série
            "window_size": window_size,
            "config": {"model_type": model_type}
        }
        
        try:
            from scipy import stats
            
            model = self.model_trained
            model.eval()
            
            # Normalisation params
            norm_params = self.norm_params
            method = norm_params.get("method", "standardization")
            
            # Préparer les données
            values = self.series.values
            series_array = np.array(values, dtype=np.float32)
            
            if method == "minmax":
                min_val, max_val = norm_params["min"], norm_params["max"]
                series_norm = (series_array - min_val) / (max_val - min_val + 1e-8)
            else:
                mean_val, std_val = norm_params["mean"], norm_params["std"]
                series_norm = (series_array - mean_val) / (std_val + 1e-8)
            
            # Contexte initial (les window_size points AVANT la zone de test)
            context_start = idx_test_start_series - window_size
            if context_start < 0:
                context_start = 0
            context = series_norm[context_start:idx_test_start_series].copy()
            
            # Si pas assez de contexte, padding
            if len(context) < window_size:
                pad = np.zeros(window_size - len(context))
                context = np.concatenate([pad, context])
            
            predictions = []
            pred_low = []
            pred_high = []
            
            z_score = stats.norm.ppf(0.975)  # 95% CI
            residual_std = self.residual_std if self.residual_std else 0.1
            
            with torch.no_grad():
                for step in range(n_test):
                    # Créer l'input avec le bon reshape
                    x_input = torch.tensor(context[-window_size:], dtype=torch.float32).unsqueeze(0)
                    
                    # RESHAPE SELON LE TYPE DE MODÈLE
                    if model_type == "lstm":
                        x_input = x_input.unsqueeze(-1)  # (1, T) -> (1, T, 1)
                    elif model_type == "cnn":
                        x_input = x_input.unsqueeze(1)   # (1, T) -> (1, 1, T)
                    # MLP: pas de reshape, reste (1, T)
                    
                    x_input = x_input.to(self.device)
                    
                    # Prédiction
                    output = model(x_input)
                    
                    # Extraire la valeur prédite selon le type de modèle
                    if model_type == "lstm":
                        # LSTM retourne (B, T, out_dim), prendre le dernier timestep
                        if output.ndim == 3:
                            y_pred_norm = output[0, -1, 0].cpu().item()
                        elif output.ndim == 2:
                            y_pred_norm = output[0, 0].cpu().item()
                        else:
                            y_pred_norm = output.cpu().item()
                    elif model_type == "cnn":
                        # CNN retourne (B, out_dim, T), prendre le dernier point temporel
                        if output.ndim == 3:
                            y_pred_norm = output[0, 0, -1].cpu().item()
                        elif output.ndim == 2:
                            y_pred_norm = output[0, -1].cpu().item()
                        else:
                            y_pred_norm = output.cpu().item()
                    else:
                        # MLP retourne (B, out_dim)
                        if output.ndim >= 2:
                            y_pred_norm = output[0, 0].cpu().item()
                        else:
                            y_pred_norm = output.cpu().item()
                    
                    # Dénormaliser
                    if method == "minmax":
                        y_pred = y_pred_norm * (max_val - min_val) + min_val
                    else:
                        y_pred = y_pred_norm * std_val + mean_val
                    
                    # Intervalle de confiance (constant pour one-step avec recalib)
                    uncertainty = residual_std * z_score
                    low = float(y_pred - uncertainty)
                    high = float(y_pred + uncertainty)
                    
                    predictions.append(float(y_pred))
                    pred_low.append(low)
                    pred_high.append(high)
                    
                    # Valeur réelle pour comparaison
                    y_true = y_true_values[step] if step < len(y_true_values) else None
                    
                    yield {
                        "type": "pred_point",
                        "step": step + 1,
                        "idx": idx_test_start_series + step,
                        "yhat": float(y_pred),
                        "y": y_true,
                        "low": low,
                        "high": high
                    }
                    
                    # ONE-STEP avec recalibration : utiliser la VRAIE valeur pour le contexte
                    # Cela évalue la vraie capacité du modèle à prédire 1 pas
                    if y_true is not None:
                        # Normaliser la vraie valeur
                        if method == "minmax":
                            next_val = (float(y_true) - min_val) / (max_val - min_val + 1e-8)
                        else:
                            next_val = (float(y_true) - mean_val) / (std_val + 1e-8)
                    else:
                        # Si pas de vraie valeur, utiliser la prédiction (mode récursif)
                        next_val = y_pred_norm
                    
                    # Roulement du contexte : garder window_size éléments
                    context = np.roll(context, -1)
                    context[-1] = next_val
            
            # Calculer les métriques
            y_true_arr = np.array([v for v in y_true_values if v is not None], dtype=np.float32)
            y_pred_arr = np.array(predictions[:len(y_true_arr)], dtype=np.float32)
            
            if len(y_true_arr) > 0 and len(y_pred_arr) > 0:
                mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
                mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
            else:
                mse, mae = 0.0, 0.0
            
            yield {
                "type": "pred_end",
                "predictions": predictions,
                "pred_low": pred_low,
                "pred_high": pred_high,
                "y_true": list(y_true_values),
                "metrics": {"MSE": mse, "MAE": mae},
                "idx_start": idx_test_start_series
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"type": "error", "message": str(e)}
    
    # ====================================
    # COMPARAISON DE TOUTES LES STRATÉGIES
    # ====================================
    def run_all_strategies_comparison(self):
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
        global stop_training_flag, trained_model_state
        stop_training_flag = False
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
            
            # Calculer les indices réels dans la série (pas dans les fenêtres)
            window_size = self.X.shape[1] if self.X.ndim >= 2 else 1
            idx_val_start_series = split_info["idx_val_start"] + window_size
            idx_test_start_series = split_info["idx_test_start"] + window_size
            
            print(f"[SPLIT] n_total_series={len(self.series.values)}")
            print(f"[SPLIT] window_size={window_size}")
            print(f"[SPLIT] n_train={split_info['n_train']}, n_val={split_info['n_val']}, n_test={split_info['n_test']}")
            print(f"[SPLIT] idx_val_start (fenêtres)={split_info['idx_val_start']}, idx_val_start (série)={idx_val_start_series}")
            print(f"[SPLIT] idx_test_start (fenêtres)={split_info['idx_test_start']}, idx_test_start (série)={idx_test_start_series}")
            
            # Information de split (avec indices dans la SÉRIE, pas dans les fenêtres)
            yield sse({
                "type": "split_info",
                "n_train": split_info["n_train"],
                "n_val": split_info["n_val"],
                "n_test": split_info["n_test"],
                "idx_val_start": idx_val_start_series,
                "idx_test_start": idx_test_start_series,
                "window_size": window_size,
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
            
            # ========== SAUVEGARDER LE MODÈLE ==========
            window_size = self.X.shape[1] if self.X.ndim >= 2 else 1
            trained_model_state["model"] = self.model_trained
            trained_model_state["norm_params"] = self.norm_params
            trained_model_state["inverse_fn"] = self.inverse_fn
            trained_model_state["window_size"] = window_size
            trained_model_state["residual_std"] = self.residual_std if self.residual_std else 0.1
            trained_model_state["model_type"] = model_type
            trained_model_state["device"] = str(self.device)
            trained_model_state["is_trained"] = True
            trained_model_state["arch_params"] = self.arch_params  # NOUVEAU: sauvegarder l'architecture
            trained_model_state["pipeline"] = self  # Référence au pipeline pour accès à cfg
            print(f"✅ Modèle sauvegardé (type={model_type}, window={window_size}, arch={self.arch_params})")
            
            # ========== DONNÉES FINALES ==========
            yield sse({
                "type": "final_plot_data",
                "series_complete": self.series.values,
                "val_predictions": val_predictions,
                "val_true": val_true,
                "pred_predictions": pred_predictions,
                "pred_low": pred_low,
                "pred_high": pred_high,
                "pred_true": pred_true,
                "idx_val_start": idx_val_start_series,
                "idx_test_start": idx_test_start_series,
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
        # Stocker aussi le nom du dataset
        dataset_name = payload.get("name", "unknown")
        payload_json = {"timestamps": ts, "values": vals, "name": dataset_name}
        print(f"[IA] Dataset '{dataset_name}' chargé dans payload_json : {len(ts)} points")
        return {"status": "success", "data": payload_json}
    except Exception as e:
        print("Exception côté IA lors du fetch_dataset :", e)
        return {"status": "error", "message": str(e)}


@app.post("/datasets/data_add_proxy")
def proxy_add_dataset(payload: dict):
    url = f"{DATA_SERVER_URL}/datasets/data_add"
    data_to_send = jsonable_encoder(payload)
    print("=== ENVOYÉ AU SERVEUR DATA ===")

    try:
        response = requests.post(url, json=data_to_send, timeout=1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de connexion au serveur DATA: {e}")

    print("=== RÉPONSE SERVEUR DATA ===")

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
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)

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
def proxy_suppression_dataset(payload: deleteDatasetRequest):
    print("Message reçu depuis UI pour suppression:", payload.name)
    url = f"{DATA_SERVER_URL}/datasets/data_supression"
    
    payload_dict = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
    
    try:
        response = requests.post(url, json=payload_dict, timeout=1000)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la suppression: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur Data: {str(e)}")


@app.post("/train_full")
def training(payload: PaquetComplet, payload_model: dict):
    """Route d'entraînement complet avec le nouveau pipeline 3 phases"""
    pipeline = TrainingPipeline(payload, payload_model)
    return StreamingResponse(pipeline.execute_full_pipeline(), media_type="text/event-stream")


stop_training_flag = False

@app.post("/stop_training")
def stop_training():
    """Arrête l'entraînement en cours"""
    global stop_training_flag
    stop_training_flag = True
    print("🛑 Arrêt de l'entraînement demandé")
    return {"status": "ok", "message": "Arrêt demandé"}


@app.get("/")
def root():
    return {"message": "Serveur IA actif !"}


# ====================================
# ENDPOINT PRÉDICTION FUTURE
# ====================================
from pydantic import BaseModel as PydanticBaseModel
from scipy import stats

class PredictRequest(PydanticBaseModel):
    horizon: int = 10
    confidence_level: float = 0.95


@app.post("/predict")
def predict_future(request: PredictRequest):
    """Prédit H pas dans le FUTUR"""
    
    def prediction_generator():
        global trained_model_state, payload_json
        
        if not trained_model_state["is_trained"]:
            yield sse({"type": "error", "message": "Aucun modèle entraîné. Entraînez d'abord via l'onglet Training."})
            return
        
        if not payload_json.get("values"):
            yield sse({"type": "error", "message": "Aucune donnée. Sélectionnez d'abord un dataset."})
            return
        
        try:
            model = trained_model_state["model"]
            norm_params = trained_model_state["norm_params"]
            inverse_fn = trained_model_state["inverse_fn"]
            window_size = trained_model_state["window_size"]
            residual_std = trained_model_state["residual_std"] or 0.1
            model_type = trained_model_state["model_type"]
            device = torch.device(trained_model_state["device"])
            
            series_values = payload_json["values"]
            horizon = request.horizon
            
            print(f"[PREDICT] horizon={horizon}, n_data={len(series_values)}, window={window_size}")
            
            yield sse({"type": "pred_start", "n_steps": horizon, "series_length": len(series_values)})
            
            # Normalisation
            series_array = np.array(series_values, dtype=np.float32)
            method = norm_params.get("method", "standardization")
            
            if method == "minmax":
                min_val, max_val = norm_params["min"], norm_params["max"]
                series_norm = (series_array - min_val) / (max_val - min_val + 1e-8)
            else:
                mean_val, std_val = norm_params["mean"], norm_params["std"]
                series_norm = (series_array - mean_val) / (std_val + 1e-8)
            
            context = series_norm[-window_size:].copy()
            predictions, pred_low, pred_high = [], [], []
            
            print(f"[PREDICT] Contexte initial (derniers {window_size} points normalisés):")
            print(f"[PREDICT] context = {context}")
            print(f"[PREDICT] Dernières valeurs réelles: {series_array[-5:]}")
            
            model.eval()
            z_score = stats.norm.ppf((1 + request.confidence_level) / 2)
            
            with torch.no_grad():
                for step in range(horizon):
                    x_input = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
                    
                    if model_type == "lstm":
                        x_input = x_input.unsqueeze(-1)  # (1, T) -> (1, T, 1)
                    elif model_type == "cnn":
                        x_input = x_input.unsqueeze(1)   # (1, T) -> (1, 1, T)
                    
                    x_input = x_input.to(device)
                    output = model(x_input)
                    
                    # Extraire correctement la prédiction selon le type de modèle
                    if model_type == "lstm":
                        # LSTM retourne (B, T, out_dim), prendre le dernier timestep
                        if output.ndim == 3:
                            y_pred_norm = output[0, -1, 0].cpu().item()
                        elif output.ndim == 2:
                            y_pred_norm = output[0, 0].cpu().item()
                        else:
                            y_pred_norm = output.cpu().item()
                    elif model_type == "cnn":
                        # CNN retourne (B, out_dim, T'), prendre le dernier timestep
                        if output.ndim == 3:
                            # (B, C, T) -> prendre output[0, 0, -1] (dernier point temporel)
                            y_pred_norm = output[0, 0, -1].cpu().item()
                        elif output.ndim == 2:
                            # (B, T) -> prendre le dernier
                            y_pred_norm = output[0, -1].cpu().item()
                        else:
                            y_pred_norm = output.cpu().item()
                    else:
                        # MLP retourne (B, out_dim)
                        y_pred_norm = output.flatten()[0].cpu().item()
                    
                    print(f"[PREDICT] step={step}, output.shape={output.shape}, y_pred_norm={y_pred_norm:.4f}")
                    
                    # Dénormaliser pour la sortie
                    if method == "minmax":
                        y_pred = y_pred_norm * (max_val - min_val) + min_val
                    else:  # standardization
                        y_pred = y_pred_norm * std_val + mean_val
                    
                    print(f"[PREDICT] step={step}, y_pred_norm={y_pred_norm:.4f}, y_pred={y_pred:.4f}")
                    
                    # IC avec incertitude croissante
                    uncertainty = residual_std * z_score * np.sqrt(step + 1)
                    low, high = float(y_pred - uncertainty), float(y_pred + uncertainty)
                    
                    predictions.append(float(y_pred))
                    pred_low.append(low)
                    pred_high.append(high)
                    
                    yield sse({"type": "pred_point", "step": step+1, "yhat": float(y_pred), "low": low, "high": high})
                    
                    # Autorégression : la prédiction normalisée devient le nouvel input
                    context = np.roll(context, -1)
                    context[-1] = y_pred_norm
            
            yield sse({
                "type": "pred_end",
                "predictions": predictions,
                "pred_low": pred_low,
                "pred_high": pred_high,
                "idx_start": len(series_values),
                "series_complete": series_values,
                "horizon": horizon
            })
            yield sse({"type": "fin_prediction", "done": 1})
            
        except Exception as e:
            import traceback
            print(f"[PREDICT] ERREUR: {e}")
            traceback.print_exc()
            yield sse({"type": "error", "message": str(e)})
    
    return StreamingResponse(prediction_generator(), media_type="text/event-stream")


@app.get("/model/status")
def model_status():
    """Statut du modèle et des données"""
    global trained_model_state, payload_json
    return {
        "model": {
            "is_trained": trained_model_state["is_trained"],
            "model_type": trained_model_state["model_type"],
            "window_size": trained_model_state["window_size"],
        },
        "data": {
            "is_loaded": bool(payload_json.get("values")),
            "n_points": len(payload_json.get("values", []))
        }
    }


# ====================================
# SAUVEGARDE ET CHARGEMENT DE MODÈLES
# ====================================
import base64
import io
from pathlib import Path

# DATA_SERVER_URL déjà défini en haut du fichier


class SaveModelRequest(PydanticBaseModel):
    name: str
    
class LoadModelRequest(PydanticBaseModel):
    name: str


@app.post("/model/save")
def save_model(request: SaveModelRequest):
    """
    Sauvegarde le modèle entraîné et son contexte sur le serveur DATA.
    - Modèle (.pth) encodé en base64 -> /models/model_add
    - Contexte (paramètres, norm_params, etc.) -> /contexte/add_solo
    """
    global trained_model_state, payload_json
    
    if not trained_model_state["is_trained"]:
        raise HTTPException(status_code=400, detail="Aucun modèle entraîné à sauvegarder")
    
    model = trained_model_state["model"]
    model_type = trained_model_state["model_type"]
    name = request.name
    
    try:
        # 1. Sauvegarder le modèle (.pth) en base64
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Envoyer au serveur DATA
        response = requests.post(
            f"{DATA_SERVER_URL}/models/model_add",
            json={"name": name, "data": model_base64},
            timeout=30
        )
        if response.status_code != 200:
            raise Exception(f"Erreur sauvegarde modèle: {response.text}")
        
        print(f"[SAVE] Modèle '{name}' sauvegardé sur le serveur DATA")
        
        # 2. Préparer le contexte complet
        # Récupérer la config depuis le pipeline (si disponible)
        config_dict = {}
        if hasattr(trained_model_state.get("pipeline"), "cfg") and trained_model_state["pipeline"].cfg:
            cfg = trained_model_state["pipeline"].cfg
            config_dict = {
                "Parametres_temporels": cfg.Parametres_temporels.model_dump() if cfg.Parametres_temporels else None,
                "Parametres_choix_loss_fct": cfg.Parametres_choix_loss_fct.model_dump() if cfg.Parametres_choix_loss_fct else None,
                "Parametres_optimisateur": cfg.Parametres_optimisateur.model_dump() if cfg.Parametres_optimisateur else None,
                "Parametres_entrainement": cfg.Parametres_entrainement.model_dump() if cfg.Parametres_entrainement else None,
            }
        
        # Construire le payload pour /contexte/add_solo
        payload = {
            "Parametres_temporels": config_dict.get("Parametres_temporels", {"horizon": 1, "portion_decoupage": 0.8}),
            "Parametres_choix_reseau_neurones": {"modele": model_type.upper()},
            "Parametres_choix_loss_fct": config_dict.get("Parametres_choix_loss_fct", {"loss": "mse"}),
            "Parametres_optimisateur": config_dict.get("Parametres_optimisateur", {"optimisateur": "adam", "learning_rate": 0.001}),
            "Parametres_entrainement": config_dict.get("Parametres_entrainement", {"epochs": 100, "batch_size": 32}),
            "Parametres_visualisation_suivi": {"afficher_courbe": True},
        }
        
        # Architecture du modèle
        arch_params = trained_model_state.get("arch_params", {})
        payload_model = {"Parametres_archi_reseau": arch_params}
        
        # Dataset info
        payload_dataset = {
            "name": payload_json.get("name", "unknown"),
            "n_points": len(payload_json.get("values", [])),
        }
        
        # Nom du modèle
        payload_name_model = {"name": name}
        
        # Contexte complet avec les infos supplémentaires pour le chargement
        paquet_complet = {
            "payload": payload,
            "payload_model": payload_model,
            "payload_dataset": payload_dataset,
            "payload_name_model": payload_name_model,
        }
        
        response = requests.post(
            f"{DATA_SERVER_URL}/contexte/add_solo",
            json=paquet_complet,
            timeout=30
        )
        if response.status_code != 200:
            raise Exception(f"Erreur sauvegarde contexte: {response.text}")
        
        print(f"[SAVE] Contexte '{name}' sauvegardé sur le serveur DATA")
        
        # 3. Sauvegarder les paramètres de normalisation et autres infos dans un fichier JSON séparé
        # On utilise une route supplémentaire ou on l'inclut dans le contexte
        extra_context = {
            "norm_params": trained_model_state["norm_params"],
            "window_size": trained_model_state["window_size"],
            "residual_std": trained_model_state["residual_std"],
            "model_type": model_type,
            "arch_params": arch_params,
            "dataset_name": payload_json.get("name", "unknown"),
        }
        
        # Sauvegarder localement aussi (backup)
        import json
        from pathlib import Path
        backup_dir = Path("./saved_models")
        backup_dir.mkdir(exist_ok=True)
        
        with open(backup_dir / f"{name}_context.json", "w") as f:
            json.dump(extra_context, f, indent=2)
        
        torch.save(model.state_dict(), backup_dir / f"{name}.pth")
        
        print(f"[SAVE] Backup local créé dans {backup_dir}")
        
        return {
            "status": "ok",
            "message": f"Modèle '{name}' sauvegardé avec succès",
            "model_type": model_type,
            "window_size": trained_model_state["window_size"],
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/list")
def list_models():
    """Liste tous les modeles disponibles avec leurs metadonnees"""
    try:
        response = requests.post(
            f"{DATA_SERVER_URL}/models/model_all",
            json={"message": "choix_models"},
            timeout=30
        )
        if response.status_code != 200:
            raise Exception(f"Erreur: {response.text}")
        
        models = response.json()
        
        # Pour chaque modele, essayer de recuperer les metadonnees depuis le backup local
        result = []
        for name, info in models.items():
            model_info = {
                "name": name,
                "nom": info.get("nom", name),
                "model_type": "?",
                "dataset_name": "?",
                "window_size": "?",
            }
            
            # Essayer de lire le backup local pour plus d'infos
            backup_context_path = Path(f"./saved_models/{name}_context.json")
            if backup_context_path.exists():
                try:
                    import json
                    with open(backup_context_path, "r") as f:
                        extra_context = json.load(f)
                    model_info["model_type"] = extra_context.get("model_type", "?")
                    model_info["dataset_name"] = extra_context.get("dataset_name", "?")
                    model_info["window_size"] = extra_context.get("window_size", "?")
                except:
                    pass
            
            result.append(model_info)
        
        return {"models": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/load")
def load_model(request: LoadModelRequest):
    """
    Charge un modèle sauvegardé depuis le serveur DATA.
    Restaure le modèle et son contexte pour faire des prédictions.
    """
    global trained_model_state
    
    name = request.name
    
    try:
        # 1. Récupérer le contexte
        response = requests.post(
            f"{DATA_SERVER_URL}/contexte/obtenir_solo",
            json={"name": name},
            timeout=30
        )
        if response.status_code != 200:
            raise Exception(f"Contexte non trouvé: {response.text}")
        
        contexte = response.json()
        print(f"[LOAD] Contexte récupéré: {contexte.keys()}")
        
        # Déterminer le type de modèle
        model_type_raw = contexte.get("Parametres_choix_reseau_neurones", "mlp")
        if isinstance(model_type_raw, dict):
            model_type = model_type_raw.get("modele", "MLP").lower()
        else:
            model_type = model_type_raw.lower()
        
        print(f"[LOAD] Type de modèle: {model_type}")
        
        # 2. Récupérer le modèle
        response = requests.post(
            f"{DATA_SERVER_URL}/models/model_all",
            json={"message": "choix_models"},
            timeout=30
        )
        if response.status_code != 200:
            raise Exception(f"Erreur récupération modèles: {response.text}")
        
        models = response.json()
        if name not in models:
            raise Exception(f"Modèle '{name}' non trouvé")
        
        model_base64 = models[name].get("model_state_dict")
        if not model_base64:
            raise Exception(f"Données du modèle '{name}' invalides")
        
        # Décoder le modèle
        model_bytes = base64.b64decode(model_base64)
        buffer = io.BytesIO(model_bytes)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
        
        # 3. Reconstruire l'architecture du modèle
        # Récupérer les paramètres d'architecture depuis le contexte
        if model_type == "cnn":
            arch_key = "Parametres_archi_reseau_CNN"
        elif model_type == "lstm":
            arch_key = "Parametres_archi_reseau_LSTM"
        else:
            arch_key = "Parametres_archi_reseau_MLP"
        
        arch_params = contexte.get(arch_key, {})
        print(f"[LOAD] Architecture: {arch_params}")
        
        # Récupérer window_size depuis le backup local ou estimer depuis state_dict
        backup_context_path = Path(f"./saved_models/{name}_context.json")
        if backup_context_path.exists():
            import json
            with open(backup_context_path, "r") as f:
                extra_context = json.load(f)
            window_size = extra_context.get("window_size", 15)
            norm_params = extra_context.get("norm_params", {"method": "standardization", "mean": 0.0, "std": 1.0})
            residual_std = extra_context.get("residual_std", 0.1)
            arch_params = extra_context.get("arch_params", arch_params)
            print(f"[LOAD] Backup trouvé! arch_params depuis backup: {arch_params}")
        else:
            print(f"[LOAD] ⚠️ Backup non trouvé à {backup_context_path}")
            # Estimer depuis le state_dict
            window_size = 15  # Valeur par défaut
            norm_params = {"method": "standardization", "mean": 0.0, "std": 1.0}
            residual_std = 0.1
            
            # Essayer de déduire les paramètres depuis les poids
            for key, value in state_dict.items():
                print(f"[LOAD] state_dict key: {key}, shape: {value.shape}")
                if "fc_in.weight" in key and model_type == "mlp":
                    # Pour MLP: fc_in.weight a shape (hidden_size, in_dim)
                    arch_params["hidden_size"] = value.shape[0]
                    window_size = value.shape[1]
                elif "lstm_in.weight_ih_l0" in key and model_type == "lstm":
                    # Pour LSTM: weight_ih a shape (4*hidden_size, in_dim)
                    arch_params["hidden_size"] = value.shape[0] // 4
                elif "conv_in.weight" in key and model_type == "cnn":
                    # Pour CNN: conv_in.weight a shape (hidden_size, in_dim, kernel_size)
                    arch_params["hidden_size"] = value.shape[0]
                    arch_params["kernel_size"] = value.shape[2]
            
            # Compter le nombre de couches
            if model_type == "mlp":
                backbone_layers = [k for k in state_dict.keys() if "backbone" in k and "weight" in k]
                arch_params["nb_couches"] = len(backbone_layers) + 1  # +1 pour fc_in
            elif model_type == "lstm":
                backbone_layers = [k for k in state_dict.keys() if "backbone" in k and "weight_ih" in k]
                arch_params["nb_couches"] = len(backbone_layers) + 1
            
            print(f"[LOAD] arch_params déduits depuis state_dict: {arch_params}")
        
        print(f"[LOAD] window_size={window_size}, norm_params={norm_params}")
        
        # Créer le modèle selon le type
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_type == "mlp":
            from .models.model_MLP import MLP
            hidden_size = arch_params.get("hidden_size", 128)
            nb_couches = arch_params.get("nb_couches", 3)
            activation = arch_params.get("activation", "relu").lower()
            
            model = MLP(
                in_dim=window_size,
                hidden_size=hidden_size,
                out_dim=1,
                num_blocks=nb_couches,
                activation=activation
            )
            
        elif model_type == "lstm":
            from .models.model_LSTM import LSTM
            hidden_size = arch_params.get("hidden_size", 64)
            nb_couches = arch_params.get("nb_couches", 2)
            bidirectional = arch_params.get("bidirectional", False)
            batch_first = arch_params.get("batch_first", True)
            
            model = LSTM(
                in_dim=1,
                hidden_dim=hidden_size,
                out_dim=1,
                nb_couches=nb_couches,
                bidirectional=bidirectional,
                batch_first=batch_first
            )
            
        elif model_type == "cnn":
            from .models.model_CNN import CNN
            hidden_size = arch_params.get("hidden_size", 64)
            nb_couches = arch_params.get("nb_couches", 2)
            kernel_size = arch_params.get("kernel_size", 3)
            activation = arch_params.get("activation", "relu").lower()
            padding = arch_params.get("padding", 1)
            stride = arch_params.get("stride", 1)
            
            model = CNN(
                in_dim=1,
                hidden_dim=hidden_size,
                out_dim=1,
                num_blocks=nb_couches,
                activation=activation,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride
            )
        else:
            raise Exception(f"Type de modèle inconnu: {model_type}")
        
        # Charger les poids
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # 4. Mettre à jour trained_model_state
        trained_model_state["model"] = model
        trained_model_state["model_type"] = model_type
        trained_model_state["is_trained"] = True
        trained_model_state["norm_params"] = norm_params
        trained_model_state["window_size"] = window_size
        trained_model_state["residual_std"] = residual_std
        trained_model_state["device"] = str(device)
        trained_model_state["arch_params"] = arch_params
        
        # Créer la fonction inverse
        def inverse_fn(data):
            if norm_params.get("method") == "minmax":
                return data * (norm_params["max"] - norm_params["min"]) + norm_params["min"]
            else:
                return data * norm_params["std"] + norm_params["mean"]
        
        trained_model_state["inverse_fn"] = inverse_fn
        
        print(f"[LOAD] Modèle '{name}' chargé avec succès!")
        
        return {
            "status": "ok",
            "message": f"Modèle '{name}' chargé avec succès",
            "model_type": model_type,
            "window_size": window_size,
            "is_ready_for_prediction": True,
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/model/delete/{name}")
def delete_model(name: str):
    """Supprime un modèle du serveur DATA"""
    try:
        # Supprimer le modèle
        response = requests.post(
            f"{DATA_SERVER_URL}/models/model_delete",
            json={"name": name},
            timeout=30
        )
        if response.status_code != 200:
            print(f"Warning: Erreur suppression modèle: {response.text}")
        
        # Supprimer le backup local si existant
        from pathlib import Path
        backup_dir = Path("./saved_models")
        model_path = backup_dir / f"{name}.pth"
        context_path = backup_dir / f"{name}_context.json"
        
        if model_path.exists():
            model_path.unlink()
        if context_path.exists():
            context_path.unlink()
        
        return {"status": "ok", "message": f"Modèle '{name}' supprimé"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))