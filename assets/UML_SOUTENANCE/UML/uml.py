#!/usr/bin/env python3
"""Script de génération des diagrammes UML pour MLApp - Version complète"""

import subprocess
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Génération des diagrammes UML MLApp")
    parser.add_argument("--output", "-o", type=str, default=".", help="Dossier de sortie")
    return parser.parse_args()

DIAGRAMS = {
    # ========================================================================
    # DIAGRAMME DE CAS D'UTILISATION
    # ========================================================================
    "UC_use_case": """@startuml
left to right direction
skinparam backgroundColor #FEFEFE
title Diagramme de Cas d'Utilisation - MLApp

actor Utilisateur as U

rectangle "MLApp" {
  package "Gestion Données" {
    usecase "UC1: Importer Dataset" as UC1
    usecase "UC2: Sélectionner Dataset" as UC2
    usecase "UC11: Supprimer Dataset" as UC11
    usecase "UC12: Filtrer par dates" as UC12
  }
  
  package "Configuration" {
    usecase "UC3: Configurer Modèle (MLP/CNN/LSTM)" as UC3
    usecase "UC4: Configurer Entraînement" as UC4
  }
  
  package "Entraînement" {
    usecase "UC5: Lancer Entraînement" as UC5
    usecase "UC6: Arrêter Entraînement" as UC6
    usecase "UC7: Visualiser Loss" as UC7
  }
  
  package "Évaluation" {
    usecase "UC8: Visualiser Métriques" as UC8
    usecase "UC9: Voir Tests (Validation + Prédiction)" as UC9
  }
  
  package "Prédiction" {
    usecase "UC13: Prédire sur Horizon H" as UC13
  }
  
  package "Persistance" {
    usecase "UC10: Sauvegarder Modèle" as UC10
    usecase "UC15: Charger Modèle" as UC15
  }
}

U --> UC1
U --> UC2
U --> UC3
U --> UC4
U --> UC5
U --> UC6
U --> UC8
U --> UC9
U --> UC10
U --> UC13
U --> UC15

UC5 ..> UC2 : <<include>>
UC5 ..> UC3 : <<include>>
UC5 ..> UC4 : <<include>>
UC5 .> UC7 : <<include>>
UC2 .> UC12 : <<extend>>

note right of UC9
  Onglet Testing:
  Courbe verte (validation)
  Courbe rouge + IC (test)
end note
@enduml""",

    # ========================================================================
    # DIAGRAMME DE COMPOSANTS
    # ========================================================================
    "DC_composants": """@startuml
skinparam backgroundColor #FEFEFE
title Diagramme de Composants - Architecture MLApp

package "Client" #FFF8E1 {
  [Interface Tkinter/CTk] as UI
  component "Matplotlib" as MPL
  component "Threading" as THR
  component "Requests" as REQ
  UI --> MPL : visualisation
  UI --> THR : async SSE
  UI --> REQ : HTTP client
}

note right of UI
  Streaming SSE pour
  mise à jour temps réel
end note

package "Serveur IA (Port 8000)" #E3F2FD {
  [FastAPI App] as IA_API
  component "TrainingPipeline" as TP
  component "PyTorch Models" as PT
  
  package "Modules Entraînement" {
    [train_MLP]
    [train_CNN]
    [train_LSTM]
  }
  
  package "Modules Test" {
    [run_validation]
    [run_prediction_test]
  }
  
  package "Prédiction Future" {
    [predict_future]
  }
  
  package "Proxies vers Data" {
    [proxy_fetch_dataset]
    [proxy_add_dataset]
    [proxy_suppression]
    [proxy_info_all]
  }
  
  IA_API --> TP
  IA_API --> [predict_future]
  IA_API --> [proxy_fetch_dataset]
  TP --> PT
  TP --> [train_MLP]
  TP --> [run_validation]
  TP --> [run_prediction_test]
}

package "Serveur Data (Port 8001)" #E8F5E9 {
  [FastAPI App] as DATA_API
  component "DatasetService" as DS
  component "ModelService" as MS
  component "ContexteService" as CS
  database "datasets/" as DB_DS
  database "models/" as DB_MD
  database "contextes/" as DB_CTX
  DATA_API --> DS
  DATA_API --> MS
  DATA_API --> CS
  DS --> DB_DS
  MS --> DB_MD
  CS --> DB_CTX
}

UI --> IA_API : HTTP/SSE
IA_API --> DATA_API : HTTP REST (via Proxies)
@enduml""",

    # ========================================================================
    # DIAGRAMMES DE CLASSES
    # ========================================================================
    "DC_IA": """@startuml
skinparam backgroundColor #FEFEFE
title Diagramme de Classes - Serveur IA

class FastAPIApp {
  +POST /train_full
  +POST /stop_training
  +POST /predict
  +GET /model/status
  +POST /model/save
  +POST /model/load
  +POST /datasets/info_all (proxy)
  +POST /datasets/fetch_dataset (proxy)
}

class TrainingPipeline {
  -stop_flag: bool
  -cfg: PaquetComplet
  -device: torch.device
  -X_train, y_train: Tensor
  -X_val, y_val: Tensor
  -X_test, y_test: Tensor
  -norm_params: dict
  -residual_std: float
  -model_trained: nn.Module
  +load_data()
  +preprocess_data()
  +normalize_data()
  +split_data_three_way()
  +run_training(): Generator
  +run_validation(): Generator
  +run_prediction_test(): Generator
  +execute_full_pipeline(): Generator
}

class trained_model_state <<dict>> {
  +model: nn.Module
  +norm_params: dict
  +window_size: int
  +residual_std: float
  +model_type: str
  +is_trained: bool
}

FastAPIApp --> TrainingPipeline
FastAPIApp --> trained_model_state
TrainingPipeline --> trained_model_state
@enduml""",

    "DC_DATA": """@startuml
skinparam backgroundColor #FEFEFE
title Diagramme de Classes - Serveur Data

class FastAPIApp {
  +POST /datasets/info_all
  +POST /datasets/data_solo
  +POST /datasets/add_dataset
  +POST /datasets/data_supression
  +POST /models/model_add
  +POST /models/model_all
  +POST /contexte/add_solo
  +POST /contexte/all
}

class DatasetService {
  -base_path: Path
  +list_datasets()
  +get_dataset_data()
  +add_dataset()
  +delete_dataset()
}

class ModelService {
  -base_path: Path
  +list_models()
  +get_model()
  +add_model()
}

class ContexteService {
  -base_path: Path
  +list_contextes()
  +get_contexte()
  +add_contexte()
}

FastAPIApp --> DatasetService
FastAPIApp --> ModelService
FastAPIApp --> ContexteService
@enduml""",

    "DC_pydantic": """@startuml
skinparam backgroundColor #FEFEFE
title Diagramme de Classes - DTOs Pydantic

class TimeSeriesData <<BaseModel>> {
  +timestamps: List[datetime]
  +values: List[float]
}

class PaquetComplet <<BaseModel>> {
  +Parametres_temporels
  +Parametres_choix_reseau_neurones
  +Parametres_choix_loss_fct
  +Parametres_optimisateur
  +Parametres_entrainement
}

class Parametres_temporels {
  +horizon: int = 1
  +window_size: int = 15
  +portion_decoupage: float = 0.8
}

class Parametres_archi_reseau_MLP {
  +nb_couches: int = 2
  +hidden_size: int = 128
  +dropout_rate: float = 0.0
}

class Parametres_archi_reseau_CNN {
  +nb_couches: int = 2
  +hidden_size: int = 64
  +kernel_size: int = 3
}

class Parametres_archi_reseau_LSTM {
  +nb_couches: int = 2
  +hidden_size: int = 64
  +bidirectional: bool = False
}

PaquetComplet *-- Parametres_temporels
@enduml""",

    # ========================================================================
    # DS1: CONFIGURATION DU MODÈLE
    # ========================================================================
    "DS1_configuration": """@startuml
skinparam backgroundColor #FEFEFE
title DS1: Configuration du Modèle

actor Utilisateur as U
participant "UI\\nFenetre_Params" as UI
participant "Formatter_JSON" as FMT

U -> UI: Ouvre Fenêtre Paramètres
activate UI

== Configuration Modèle ==
U -> UI: Sélectionne type modèle\\n(MLP / CNN / LSTM)
UI -> UI: Affiche paramètres spécifiques

alt MLP sélectionné
  U -> UI: Configure:\\n- nb_couches\\n- hidden_size\\n- dropout_rate\\n- activation
else CNN sélectionné
  U -> UI: Configure:\\n- nb_couches\\n- hidden_size\\n- kernel_size\\n- stride, padding
else LSTM sélectionné
  U -> UI: Configure:\\n- nb_couches\\n- hidden_size\\n- bidirectional
end

== Configuration Entraînement ==
U -> UI: Configure paramètres temporels:\\n- window_size\\n- horizon

U -> UI: Configure optimiseur:\\n- type (Adam/SGD/RMSprop)\\n- learning_rate\\n- weight_decay

U -> UI: Configure entraînement:\\n- nb_epochs\\n- batch_size\\n- loss (MSE/MAE/Huber)

== Création Payload ==
U -> UI: Clic "Valider"
UI -> FMT: Formatter_JSON_global()
FMT -> FMT: Créer PaquetComplet
FMT -> FMT: Créer payload_model

FMT --> UI: {payload, payload_model}
UI --> U: Configuration prête\\nBouton "Start" activé

deactivate UI
@enduml""",

    # ========================================================================
    # DS2: LANCEMENT ENTRAÎNEMENT
    # ========================================================================
    "DS2_entrainement": """@startuml
skinparam backgroundColor #FEFEFE
title DS2: Lancement de l'Entraînement (Phase 1)

actor Utilisateur as U
participant "UI" as UI
participant "Serveur IA\\n/train_full" as IA
participant "TrainingPipeline" as TP
participant "train_MLP/CNN/LSTM" as TRAIN

U -> UI: Clic "Start"
activate UI

UI -> IA: POST /train_full\\n{payload, payload_model}
activate IA

IA -> TP: new TrainingPipeline(cfg)
activate TP

== Initialisation ==
TP -> TP: load_data(payload_json)
TP -> TP: setup_model_config()
TP -> TP: preprocess_data()
TP -> TP: normalize_data()
TP -> TP: split_data_three_way()\\n(80% train / 10% val / 10% test)

TP -->> UI: SSE: {type: "split_info",\\nn_train, n_val, n_test}
TP -->> UI: SSE: {type: "serie_complete",\\nvalues: [...]}

== Phase 1: Entraînement ==
TP -->> UI: SSE: {type: "phase",\\nphase: "train", status: "start"}

TP -> TRAIN: Créer générateur train_*()
activate TRAIN

loop Pour chaque epoch (1 à nb_epochs)
  TRAIN -> TRAIN: Forward pass (batches)
  TRAIN -> TRAIN: Calcul loss
  TRAIN -> TRAIN: Backward + optimizer.step()
  
  TRAIN -->> TP: yield {epoch, avg_loss}
  TP -->> UI: SSE: {type: "epoch",\\nepoch: n, avg_loss: 0.0234}
  
  UI -> UI: update_loss_plot()
  
  alt stop_flag == True
    TP -->> UI: SSE: {type: "stopped"}
    TP -> TP: break
  end
end

TRAIN --> TP: model_trained
deactivate TRAIN

TP -->> UI: SSE: {type: "phase",\\nphase: "train", status: "end"}

note right of TP
  model_trained sauvegardé
  dans trained_model_state
end note

deactivate TP
deactivate IA
deactivate UI
@enduml""",

    # ========================================================================
    # DS3: PHASE DE TEST (VALIDATION + TEST PRÉDICTIF)
    # ========================================================================
    "DS3_test": """@startuml
skinparam backgroundColor #FEFEFE
title DS3: Phase de Test (Validation + Test Prédictif)

participant "UI\\nCadre_Testing" as UI
participant "TrainingPipeline" as TP
participant "model_trained" as MODEL

activate TP

== Phase 2: Validation (10%) - Courbe Verte ==
TP -->> UI: SSE: {type: "phase",\\nphase: "validation", status: "start"}
TP -->> UI: SSE: {type: "val_start",\\nn_points: 500, idx_start: 4015}

loop Pour chaque point de validation
  TP -> MODEL: model(X_val[i])
  MODEL --> TP: y_hat_norm
  
  TP -> TP: y_hat = inverse_normalize(y_hat_norm)
  TP -> TP: Stocker (y_true, y_hat)
  
  TP -->> UI: SSE: {type: "val_pair",\\nidx: i, y: 1.234, yhat: 1.228}
  UI -> UI: plot_validation(idx, y, yhat)
end

TP -> TP: residual_std = std(y_true - y_pred)
TP -->> UI: SSE: {type: "val_end",\\nresidual_std: 0.035,\\nmetrics: {MSE, MAE}}

note right of TP
  residual_std utilisé pour
  les intervalles de confiance
end note

== Phase 3: Test Prédictif (10%) - Courbe Rouge + IC ==
TP -->> UI: SSE: {type: "phase",\\nphase: "prediction", status: "start"}
TP -->> UI: SSE: {type: "pred_start",\\nn_steps: 500, idx_start: 4515}

loop Pour chaque point de test
  TP -> MODEL: model(context)
  MODEL --> TP: y_hat_norm
  
  TP -> TP: y_hat = inverse_normalize(y_hat_norm)
  TP -> TP: low = y_hat - 1.96 × residual_std
  TP -> TP: high = y_hat + 1.96 × residual_std
  
  TP -->> UI: SSE: {type: "pred_point",\\nidx: i, yhat: 1.089,\\ny: 1.092, low: 1.02, high: 1.16}
  UI -> UI: plot_prediction(idx, yhat, low, high)
  
  TP -> TP: Recalibrer context\\navec y_true (ONE_STEP)
end

TP -->> UI: SSE: {type: "pred_end",\\nmetrics: {MSE, MAE}}

== Finalisation ==
TP -> TP: Sauvegarder trained_model_state
TP -->> UI: SSE: {type: "final_plot_data",\\nval_predictions, pred_predictions,\\npred_low, pred_high}
TP -->> UI: SSE: {type: "fin_pipeline", done: 1}

deactivate TP

note over UI
  Onglet Testing affiche:
  - Série réelle (bleu)
  - Validation (vert)
  - Test + IC 95% (rouge)
end note
@enduml""",

    # ========================================================================
    # DS4: MÉTRIQUES
    # ========================================================================
    "DS4_metrics": """@startuml
skinparam backgroundColor #FEFEFE
title DS4: Calcul et Affichage des Métriques

participant "UI\\nCadre_Metrics" as UI
participant "TrainingPipeline" as TP

== Fin Phase 2: Validation ==
TP -> TP: Calculer métriques validation
note right of TP
  y_true_arr = [y1, y2, ...]
  y_pred_arr = [yhat1, yhat2, ...]
  
  MSE = mean((y - yhat)²)
  MAE = mean(|y - yhat|)
  residual_std = std(y - yhat)
end note

TP -->> UI: SSE: {type: "val_end",\\nmetrics: {MSE: 0.0012, MAE: 0.028},\\nresidual_std: 0.035}

== Fin Phase 3: Test Prédictif ==
TP -> TP: Calculer métriques test
note right of TP
  MSE = mean((y - yhat)²)
  MAE = mean(|y - yhat|)
  RMSE = sqrt(MSE)
  
  SS_res = sum((y - yhat)²)
  SS_tot = sum((y - mean(y))²)
  R² = 1 - SS_res/SS_tot
  
  MAPE = mean(|y-yhat|/|y|) × 100
end note

TP -->> UI: SSE: {type: "pred_end",\\nmetrics: {MSE: 0.0015, MAE: 0.031}}

== Fin Pipeline ==
TP -->> UI: SSE: {type: "final_plot_data",\\nval_metrics: {MSE, MAE},\\npred_metrics: {MSE, MAE}}

UI -> UI: Calculer métriques dérivées:\\n- RMSE = sqrt(MSE)\\n- R²\\n- MAPE

UI -> UI: update_metrics_display()

actor Utilisateur as U
U -> UI: Consulte Onglet Metrics

note over UI
  Affichage:
  ┌─────────────────────┐
  │ MSE:   0.0015       │
  │ MAE:   0.031        │
  │ RMSE:  0.039        │
  │ R²:    0.94         │
  │ MAPE:  2.8%         │
  │ σ_res: 0.035        │
  └─────────────────────┘
end note
@enduml""",

    # ========================================================================
    # DS5: PRÉDICTION FUTURE
    # ========================================================================
    "DS5_prediction": """@startuml
skinparam backgroundColor #FEFEFE
title DS5: Prédiction Future (Onglet Prediction)

actor Utilisateur as U
participant "UI\\nCadre_Prediction" as UI
participant "Serveur IA\\n/predict" as API
participant "trained_model_state" as STATE
participant "model" as MODEL

U -> UI: Configure horizon H
U -> UI: Clic "Prédire"
activate UI

UI -> API: POST /predict\\n{horizon: H, confidence_level: 0.95}
activate API

API -> STATE: Vérifier is_trained
STATE --> API: True

API -> STATE: Récupérer:\\n- model\\n- norm_params\\n- window_size\\n- residual_std

API -> API: Préparer contexte initial\\ncontext = series[-window_size:]\\ncontext_norm = normalize(context)

API -->> UI: SSE: {type: "pred_start",\\nn_steps: H, series_length: N}

loop Pour step = 1 à H
  API -> MODEL: model(context_norm)
  MODEL --> API: y_pred_norm
  
  API -> API: y_pred = denormalize(y_pred_norm)
  
  note right of API
    Incertitude croissante:
    σ = residual_std × z × √(step+1)
    
    z = 1.96 pour IC 95%
  end note
  
  API -> API: uncertainty = σ × √(step+1)
  API -> API: low = y_pred - uncertainty
  API -> API: high = y_pred + uncertainty
  
  API -->> UI: SSE: {type: "pred_point",\\nstep: step, yhat: y_pred,\\nlow: low, high: high}
  
  UI -> UI: plot_future_prediction()
  
  API -> API: AUTORÉGRESSION:\\ncontext_norm = roll(context_norm, -1)\\ncontext_norm[-1] = y_pred_norm
end

API -->> UI: SSE: {type: "pred_end",\\npredictions: [...],\\npred_low: [...], pred_high: [...]}
API -->> UI: SSE: {type: "fin_prediction", done: 1}

deactivate API

note over UI
  Mode RECURSIVE:
  Pas de y_true disponible
  → utilise ses propres prédictions
  → IC croît avec √(step+1)
end note

deactivate UI
@enduml""",

    # ========================================================================
    # DS6: IMPORT DATASET
    # ========================================================================
    "DS6_import_dataset": """@startuml
skinparam backgroundColor #FEFEFE
title DS6: Import Dataset

actor Utilisateur as U
participant "UI\\nGestion_Datasets" as UI
participant "Serveur IA\\n(Proxy)" as IA
participant "Serveur Data" as DATA

U -> UI: Clic "Importer"
U -> UI: Sélectionne fichier\\nCSV ou JSON
activate UI

UI -> UI: Parser fichier\\n→ timestamps[]\\n→ values[]

UI -> UI: Demander nom du dataset

UI -> IA: POST /datasets/add_dataset\\n{name, timestamps, values}
activate IA

note right of IA
  Proxy: forward vers
  Serveur Data
end note

IA -> DATA: POST /datasets/add_dataset\\n{name, timestamps, values}
activate DATA

DATA -> DATA: Valider format:\\n- len(timestamps) == len(values)\\n- timestamps parsables\\n- values numériques

DATA -> DATA: Écrire fichier\\ndatasets/{name}.json

DATA --> IA: {status: "ok",\\nn_points: 5000}
deactivate DATA

IA --> UI: {status: "ok",\\nn_points: 5000}
deactivate IA

UI -> UI: Rafraîchir liste datasets
UI -> UI: Afficher confirmation:\\n"Dataset importé (5000 points)"

deactivate UI
@enduml""",

    # ========================================================================
    # DS7: FETCH/LOAD DATASET
    # ========================================================================
    "DS7_fetch_dataset": """@startuml
skinparam backgroundColor #FEFEFE
title DS7: Chargement Dataset (Fetch)

actor Utilisateur as U
participant "UI" as UI
participant "Serveur IA\\n(Proxy)" as IA
participant "payload_json" as PJ
participant "Serveur Data" as DATA

== Liste des datasets ==
U -> UI: Ouvre application
activate UI

UI -> IA: POST /datasets/info_all\\n{message: "choix_datasets"}
activate IA
IA -> DATA: POST /datasets/info_all
DATA --> IA: {datasets: ["ds1", "ds2", ...]}
IA --> UI: {datasets: [...]}
deactivate IA

UI -> UI: Afficher dans Treeview

== Sélection et chargement ==
U -> UI: Sélectionne dataset\\ndans Treeview
U -> UI: Configure dates (optionnel):\\n- date_start\\n- date_end
U -> UI: Clic "Charger"

UI -> IA: POST /datasets/fetch_dataset\\n{name: "ds1",\\ndates: ["2020-01-01", "2025-12-31"]}
activate IA

IA -> DATA: POST /datasets/data_solo\\n{name, dates}
activate DATA

DATA -> DATA: Lire datasets/{name}.json
DATA -> DATA: Filtrer par dates si spécifié

DATA --> IA: {timestamps: [...],\\nvalues: [...]}
deactivate DATA

IA -> PJ: Stocker globalement:\\npayload_json = {\\n  timestamps: [...],\\n  values: [...],\\n  name: "ds1"\\n}

IA --> UI: {status: "success",\\ndata: {timestamps, values},\\nn_points: 4500}
deactivate IA

UI -> UI: Afficher série dans\\nonglet Représentation
UI -> UI: Activer bouton "Start"
UI --> U: "Dataset chargé (4500 points)"

deactivate UI
@enduml""",

    # ========================================================================
    # DS8: SAVE MODEL
    # ========================================================================
    "DS8_save_model": """@startuml
skinparam backgroundColor #FEFEFE
title DS8: Sauvegarde du Modèle

actor Utilisateur as U
participant "UI" as UI
participant "Serveur IA\\n/model/save" as IA
participant "trained_model_state" as STATE
participant "Serveur Data" as DATA

U -> UI: Clic "Sauvegarder Modèle"
U -> UI: Entre nom: "mon_modele"
activate UI

UI -> IA: POST /model/save\\n{name: "mon_modele"}
activate IA

IA -> STATE: Vérifier is_trained
STATE --> IA: True

== Sauvegarde du modèle (.pth) ==
IA -> STATE: Récupérer model
IA -> IA: buffer = BytesIO()
IA -> IA: torch.save(model.state_dict(), buffer)
IA -> IA: model_base64 = base64.encode(buffer)

IA -> DATA: POST /models/model_add\\n{name: "mon_modele",\\ndata: "<base64>"}
activate DATA
DATA -> DATA: Décoder base64
DATA -> DATA: Écrire models/mon_modele.pth
DATA --> IA: {status: "ok"}
deactivate DATA

== Sauvegarde du contexte (.json) ==
IA -> STATE: Récupérer:\\n- norm_params\\n- window_size\\n- residual_std\\n- arch_params

IA -> IA: Construire contexte complet

IA -> DATA: POST /contexte/add_solo\\n{payload, payload_model,\\npayload_dataset, payload_name_model}
activate DATA
DATA -> DATA: Écrire contextes/mon_modele.json
DATA --> IA: {status: "ok"}
deactivate DATA

== Backup local ==
IA -> IA: Écrire saved_models/mon_modele.pth
IA -> IA: Écrire saved_models/mon_modele_context.json

IA --> UI: {status: "ok",\\nmessage: "Modèle sauvegardé",\\nmodel_type: "mlp"}
deactivate IA

UI -> UI: Afficher confirmation
UI -> UI: Rafraîchir liste modèles

deactivate UI
@enduml""",

    # ========================================================================
    # DS9: LOAD MODEL
    # ========================================================================
    "DS9_load_model": """@startuml
skinparam backgroundColor #FEFEFE
title DS9: Chargement du Modèle

actor Utilisateur as U
participant "UI" as UI
participant "Serveur IA\\n/model/load" as IA
participant "trained_model_state" as STATE
participant "Serveur Data" as DATA

U -> UI: Sélectionne modèle\\ndans liste
U -> UI: Clic "Charger Modèle"
activate UI

UI -> IA: POST /model/load\\n{name: "mon_modele"}
activate IA

== Récupération du contexte ==
IA -> DATA: POST /contexte/all
activate DATA
DATA --> IA: {contextes: {...}}
deactivate DATA

IA -> IA: Extraire contexte "mon_modele":\\n- model_type\\n- arch_params\\n- norm_params\\n- window_size

== Récupération du modèle ==
IA -> DATA: POST /models/model_all
activate DATA
DATA --> IA: {models: {mon_modele: "<base64>"}}
deactivate DATA

IA -> IA: state_dict = base64.decode(data)

== Reconstruction du modèle ==
IA -> IA: Créer architecture selon model_type

alt model_type == "mlp"
  IA -> IA: model = MLP(in_dim, hidden_size,\\nout_dim, num_blocks)
else model_type == "cnn"
  IA -> IA: model = CNN(in_dim, hidden_dim,\\nout_dim, kernel_size, ...)
else model_type == "lstm"
  IA -> IA: model = LSTM(in_dim, hidden_dim,\\nout_dim, nb_couches, ...)
end

IA -> IA: model.load_state_dict(state_dict)
IA -> IA: model.eval()
IA -> IA: model.to(device)

== Mise à jour état global ==
IA -> STATE: trained_model_state = {\\n  model: model,\\n  is_trained: True,\\n  norm_params: {...},\\n  window_size: 15,\\n  residual_std: 0.035,\\n  model_type: "mlp"\\n}

IA --> UI: {status: "ok",\\nmodel_type: "mlp",\\nis_ready_for_prediction: true}
deactivate IA

UI -> UI: Afficher confirmation
UI -> UI: Activer onglet Prediction

deactivate UI
@enduml""",

    # ========================================================================
    # DS10: DELETE DATASET
    # ========================================================================
    "DS10_delete_dataset": """@startuml
skinparam backgroundColor #FEFEFE
title DS10: Suppression Dataset

actor Utilisateur as U
participant "UI" as UI
participant "Serveur IA\\n(Proxy)" as IA
participant "Serveur Data" as DATA

U -> UI: Sélectionne dataset
U -> UI: Clic "Supprimer"
activate UI

UI -> UI: Demander confirmation:\\n"Supprimer dataset X ?"

U -> UI: Confirme

UI -> IA: POST /datasets/data_suppression_proxy\\n{name: "dataset_x"}
activate IA

IA -> DATA: POST /datasets/data_supression\\n{name: "dataset_x"}
activate DATA

DATA -> DATA: Vérifier existence
DATA -> DATA: Supprimer datasets/{name}.json

DATA --> IA: {status: "ok"}
deactivate DATA

IA --> UI: {status: "ok"}
deactivate IA

UI -> UI: Rafraîchir liste datasets
UI -> UI: Afficher confirmation

deactivate UI
@enduml""",

    # ========================================================================
    # DIAGRAMME D'ACTIVITÉ
    # ========================================================================
    "DA_pipeline": """@startuml
skinparam backgroundColor #FEFEFE
title Diagramme d'Activité - Pipeline Complet

start
:Utilisateur clique "Start";

partition "Initialisation" {
  :load_data(payload_json);
  :setup_model_config();
  :preprocess_data();
  :normalize_data();
  :split_data_three_way()\\n(80% / 10% / 10%);
}

partition "Phase 1: Entraînement (80%)" #E3F2FD {
  :SSE: phase=train;
  
  while (epoch < nb_epochs?) is (oui)
    :Forward pass;
    :Calcul loss;
    :Backward pass;
    :optimizer.step();
    :SSE: epoch + avg_loss;
    
    if (stop_flag?) then (oui)
      :SSE: stopped;
      stop
    endif
  endwhile (non)
  
  :Récupérer model_trained;
}

partition "Phase 2: Validation (10%)" #E8F5E9 {
  :SSE: phase=validation;
  
  while (i < n_val?) is (oui)
    :y_hat = model(X_val[i]);
    :Dénormaliser;
    :SSE: val_pair (idx, y, yhat);
  endwhile (non)
  
  :residual_std = std(résidus);
  :SSE: val_end + metrics;
}

partition "Phase 3: Test Prédictif (10%)" #FFF8E1 {
  :SSE: phase=prediction;
  
  while (step < n_test?) is (oui)
    :y_hat = model(context);
    :Calculer IC 95%;
    :SSE: pred_point (yhat, low, high);
    :Recalibrer context avec y_true;
  endwhile (non)
  
  :SSE: pred_end + metrics;
}

:Sauvegarder trained_model_state;
:SSE: final_plot_data;
:SSE: fin_pipeline;

stop
@enduml""",

    # ========================================================================
    # DIAGRAMME DE DÉPLOIEMENT
    # ========================================================================
    "DD_deploiement": """@startuml
skinparam backgroundColor #FEFEFE
title Diagramme de Déploiement - MLApp

node "Machine Client" {
  artifact "Python 3.8+" as PY_C
  artifact "Tkinter/CTk" as TK
  artifact "Matplotlib" as MPL
  artifact "Requests" as REQ
  
  PY_C --> TK
  TK --> MPL
  TK --> REQ
}

node "Serveur IA (Port 8000)" {
  artifact "Python 3.8+" as PY_IA
  artifact "FastAPI" as FAST_IA
  artifact "PyTorch" as TORCH
  artifact "NumPy/SciPy" as NP
  
  PY_IA --> FAST_IA
  FAST_IA --> TORCH
  FAST_IA --> NP
}

node "Serveur Data (Port 8001)" {
  artifact "Python 3.8+" as PY_D
  artifact "FastAPI" as FAST_D
  
  database "datasets/" as DS
  database "models/" as MD
  database "contextes/" as CTX
  
  PY_D --> FAST_D
  FAST_D --> DS
  FAST_D --> MD
  FAST_D --> CTX
}

"Machine Client" --> "Serveur IA (Port 8000)" : HTTP/SSE\\nPort 8000
"Serveur IA (Port 8000)" --> "Serveur Data (Port 8001)" : HTTP REST\\nPort 8001

note right of "Serveur IA (Port 8000)"
  GPU optionnel:
  CUDA / MPS
end note

note right of "Serveur Data (Port 8001)"
  Stockage fichiers:
  JSON + .pth
end note
@enduml"""
}

def generate_puml_files(output_dir):
    print("\n📝 Génération des fichiers .puml...")
    puml_dir = output_dir / "puml"
    puml_dir.mkdir(exist_ok=True)
    for name, code in DIAGRAMS.items():
        puml_path = puml_dir / f"{name}.puml"
        with open(puml_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"  ✓ {name}.puml")
    return puml_dir

def generate_png_with_plantuml(puml_dir, output_dir):
    print("\n🖼️  Génération des PNG avec PlantUML...")
    success, failed = 0, 0
    for puml_file in sorted(puml_dir.glob("*.puml")):
        try:
            subprocess.run(["plantuml", "-tpng", "-o", str(output_dir.absolute()), str(puml_file)],
                          capture_output=True, timeout=60)
            if (output_dir / f"{puml_file.stem}.png").exists():
                print(f"  ✓ {puml_file.stem}.png")
                success += 1
            else:
                print(f"  ✗ {puml_file.stem}")
                failed += 1
        except FileNotFoundError:
            print("  ✗ PlantUML non trouvé. Installer avec: brew install plantuml")
            failed += 1
            break
        except Exception as e:
            print(f"  ✗ {puml_file.stem}: {e}")
            failed += 1
    return success, failed

def main():
    args = parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"🔧 Génération UML MLApp - {len(DIAGRAMS)} diagrammes")
    print("=" * 60)
    
    print("\n📋 Diagrammes:")
    for i, name in enumerate(DIAGRAMS.keys(), 1):
        print(f"   {i:2}. {name}")
    
    puml_dir = generate_puml_files(output_dir)
    success, failed = generate_png_with_plantuml(puml_dir, output_dir)
    
    print("\n" + "=" * 60)
    print(f"✅ Terminé: {success} succès, {failed} échecs")
    if failed > 0:
        print(f"\n💡 Pour générer manuellement:")
        print(f"   plantuml -tpng {puml_dir}/*.puml -o {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
