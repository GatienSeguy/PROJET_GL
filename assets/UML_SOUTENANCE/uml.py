#!/usr/bin/env python3
"""
Script de génération des diagrammes UML pour MLApp
Génère des fichiers PNG via le serveur PlantUML public ou local.

Usage:
    python uml.py                    # Génère dans ./diagrams/
    python uml.py --output /mon/path # Génère dans /mon/path/
    python uml.py --plantuml-jar /path/to/plantuml.jar  # Utilise JAR local
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Génération des diagrammes UML MLApp")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./diagrams",
        help="Dossier de sortie pour les PNG (défaut: ./diagrams)"
    )
    parser.add_argument(
        "--plantuml-jar",
        type=str,
        default=None,
        help="Chemin vers plantuml.jar (si non spécifié, utilise le serveur web)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="png",
        choices=["png", "svg", "pdf"],
        help="Format de sortie (défaut: png)"
    )
    return parser.parse_args()

# ============================================================================
# MÉTHODES DE GÉNÉRATION
# ============================================================================

def generate_with_jar(name: str, code: str, output_dir: Path, jar_path: str, fmt: str) -> bool:
    """Génère un diagramme en utilisant plantuml.jar localement"""
    try:
        # Créer fichier temporaire .puml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.puml', delete=False) as f:
            f.write(code)
            puml_path = f.name
        
        # Exécuter PlantUML
        cmd = ["java", "-jar", jar_path, f"-t{fmt}", "-o", str(output_dir), puml_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Renommer le fichier de sortie
        generated = Path(puml_path).with_suffix(f".{fmt}")
        if generated.exists():
            final_path = output_dir / f"{name}.{fmt}"
            generated.rename(final_path)
            print(f"✓ {name}.{fmt} généré")
            return True
        
        # Vérifier si le fichier a été généré directement dans output_dir
        direct_output = output_dir / Path(puml_path).with_suffix(f".{fmt}").name
        if direct_output.exists():
            final_path = output_dir / f"{name}.{fmt}"
            direct_output.rename(final_path)
            print(f"✓ {name}.{fmt} généré")
            return True
            
        print(f"✗ {name}: fichier non généré")
        if result.stderr:
            print(f"  Erreur: {result.stderr[:200]}")
        return False
        
    except subprocess.TimeoutExpired:
        print(f"✗ {name}: timeout")
        return False
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False
    finally:
        # Nettoyer le fichier temporaire
        try:
            os.unlink(puml_path)
        except:
            pass

def generate_with_server(name: str, code: str, output_dir: Path, fmt: str) -> bool:
    """Génère un diagramme via le serveur PlantUML public"""
    try:
        import requests
        import zlib
        
        # Encodage PlantUML
        compressed = zlib.compress(code.encode('utf-8'))[2:-4]
        base64_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
        encoded = ""
        
        for i in range(0, len(compressed), 3):
            if i + 2 < len(compressed):
                b1, b2, b3 = compressed[i], compressed[i+1], compressed[i+2]
            elif i + 1 < len(compressed):
                b1, b2, b3 = compressed[i], compressed[i+1], 0
            else:
                b1, b2, b3 = compressed[i], 0, 0
            
            encoded += base64_chars[b1 >> 2]
            encoded += base64_chars[((b1 & 0x3) << 4) | (b2 >> 4)]
            encoded += base64_chars[((b2 & 0xF) << 2) | (b3 >> 6)]
            encoded += base64_chars[b3 & 0x3F]
        
        url = f"http://www.plantuml.com/plantuml/{fmt}/{encoded}"
        
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        })
        response.raise_for_status()
        
        output_path = output_dir / f"{name}.{fmt}"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ {name}.{fmt} généré ({len(response.content)} bytes)")
        return True
        
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

def generate_puml_files(output_dir: Path) -> None:
    """Génère uniquement les fichiers .puml pour utilisation manuelle"""
    print("\nGénération des fichiers .puml...")
    puml_dir = output_dir / "puml"
    puml_dir.mkdir(exist_ok=True)
    
    for name, code in DIAGRAMS.items():
        puml_path = puml_dir / f"{name}.puml"
        with open(puml_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"  ✓ {name}.puml")
    
    print(f"\nFichiers .puml générés dans: {puml_dir}")
    print("\nPour générer les PNG manuellement:")
    print("  1. Installer PlantUML: brew install plantuml")
    print(f"  2. Exécuter: plantuml -tpng {puml_dir}/*.puml")
    print("  Ou utiliser l'extension VS Code 'PlantUML'")

# ============================================================================
# DIAGRAMMES UML
# ============================================================================

DIAGRAMS = {
    # Diagramme de Cas d'Utilisation
    "use_case": """@startuml
left to right direction
skinparam packageStyle rectangle
skinparam actorStyle awesome
skinparam backgroundColor #FEFEFE

title Diagramme de Cas d'Utilisation - MLApp

actor Utilisateur as U

rectangle "MLApp" {
  package "Gestion Données" {
    usecase "Importer Dataset" as UC1
    usecase "Sélectionner Dataset" as UC2
    usecase "Supprimer Dataset" as UC11
    usecase "Filtrer par dates" as UC12
  }
  
  package "Configuration" {
    usecase "Configurer Modèle\\n(MLP/CNN/LSTM)" as UC3
    usecase "Configurer Entraînement" as UC4
    usecase "Choisir Métriques" as UC13
  }
  
  package "Entraînement" {
    usecase "Lancer Entraînement" as UC5
    usecase "Arrêter Entraînement" as UC6
    usecase "Visualiser Loss\\nen temps réel" as UC7
  }
  
  package "Évaluation" {
    usecase "Visualiser Métriques" as UC8
    usecase "Effectuer Prédiction" as UC9
    usecase "Choisir Stratégie\\nde Prédiction" as UC14
  }
  
  package "Persistance" {
    usecase "Sauvegarder Modèle" as UC10
    usecase "Charger Modèle" as UC15
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
U --> UC11
U --> UC15

UC5 ..> UC2 : <<include>>
UC5 ..> UC3 : <<include>>
UC5 ..> UC4 : <<include>>
UC5 .> UC7 : <<include>>
UC9 ..> UC5 : <<include>>
UC9 .> UC14 : <<extend>>
UC10 ..> UC5 : <<include>>
UC2 .> UC12 : <<extend>>

@enduml""",

    # Diagramme de Classes - Interface Utilisateur
    "class_ui": """@startuml
skinparam classAttributeIconSize 0
skinparam backgroundColor #FEFEFE
skinparam class {
    BackgroundColor #F8F9FA
    BorderColor #2C3E50
    ArrowColor #2C3E50
}

title Diagramme de Classes - Interface Utilisateur (Tkinter/CTk)

package "Interface Principale" #E8F6F3 {
  class Fenetre_Acceuil <<CTk>> {
    - JSON_Datasets: dict
    - stop_training: bool
    - Payload: dict
    - Results_notebook: CTkTabview
    - cadre: CTkFrame
    --
    + __init__()
    + obtenir_datasets(): int
    + EnvoyerConfig(): void
    + annuler_entrainement(): void
    + Formatter_JSON_global(): dict
    + Formatter_JSON_dataset(): dict
    + Formatter_JSON_specif(): dict
    + Parametrer_modele(): void
    + Parametrer_horizon(): void
    + Gestion_Datasets(): void
  }
}

package "Onglets Résultats" #EBF5FB {
  class Cadre_Entrainement <<CTkFrame>> {
    - is_training: bool
    - epochs_data: List[int]
    - loss_data: List[float]
    - figure: Figure
    - canvas: FigureCanvasTkAgg
    - ax: Axes
    --
    + start_training(): void
    + add_data_point(epoch, loss, epoch_s): void
    + update_plot(): void
    + reset(): void
  }

  class Cadre_Testing <<CTkFrame>> {
    - y_true_data: List[float]
    - y_pred_data: List[float]
    - figure: Figure
    - canvas: FigureCanvasTkAgg
    --
    + add_prediction(y, yhat): void
    + display_metrics(metrics: dict): void
    + update_plot(): void
    + reset(): void
  }

  class Cadre_Metrics <<CTkFrame>> {
    - mse_label: CTkLabel
    - mae_label: CTkLabel
    - rmse_label: CTkLabel
    - r2_label: CTkLabel
    --
    + display_metrics(mse, mae, rmse, r2): void
    + clear(): void
  }

  class Cadre_Prediction <<CTkFrame>> {
    - predictions: List[float]
    - low_bounds: List[float]
    - high_bounds: List[float]
    - figure: Figure
    --
    + add_prediction_point(yhat, low, high): void
    + display_final(data: dict): void
    + update_plot(): void
  }
}

package "Fenêtres de Configuration" #FDF2E9 {
  class Gestion_Datasets <<CTkToplevel>> {
    - Dataset_tree: Treeview
    - Selected_Dataset: dict
    - payload_add_dataset: TimeSeriesData
    --
    + rafraichir_liste_datasets(): void
    + charger_fichier_json(): void
    + supprimer_dataset(): void
    + Select_Dataset(): void
  }

  class Fenetre_Params <<CTkToplevel>> {
    - model_type: StringVar
    - params_frame: CTkFrame
    --
    + create_mlp_params(): void
    + create_cnn_params(): void
    + create_lstm_params(): void
    + save_params(): void
    + on_model_change(): void
  }

  class Fenetre_Params_horizon <<CTkToplevel>> {
    - horizon_entry: CTkEntry
    - portion_entry: CTkEntry
    - calendars: List[Calendar]
    --
    + validate_dates(): bool
    + save_params(): void
  }
}

Fenetre_Acceuil "1" *-- "1" Cadre_Entrainement
Fenetre_Acceuil "1" *-- "1" Cadre_Testing
Fenetre_Acceuil "1" *-- "1" Cadre_Metrics
Fenetre_Acceuil "1" *-- "1" Cadre_Prediction
Fenetre_Acceuil ..> Gestion_Datasets : ouvre
Fenetre_Acceuil ..> Fenetre_Params : ouvre
Fenetre_Acceuil ..> Fenetre_Params_horizon : ouvre

@enduml""",

    # Diagramme de Classes - Serveur IA
    "class_serveur_ia": """@startuml
skinparam classAttributeIconSize 0
skinparam backgroundColor #FEFEFE
skinparam class {
    BackgroundColor #F8F9FA
    BorderColor #2C3E50
}

title Diagramme de Classes - Serveur IA (FastAPI + PyTorch)

package "API FastAPI" #E8F6F3 {
  class FastAPIApp <<Controller>> {
    + train_full(payload, payload_model): StreamingResponse
    + stop_training(): dict
    + proxy_get_dataset_list(payload): dict
    + proxy_fetch_dataset(payload): dict
    + add_dataset_proxy(packet): dict
    + proxy_suppression_dataset(payload): dict
    + root(): dict
  }
}

package "Pipeline Entrainement" #EBF5FB {
  class TrainingPipeline {
    - cfg: PaquetComplet
    - payload_model: dict
    - device: torch.device
    - series: TimeSeriesData
    - X, y: Tensor
    - X_train, y_train: Tensor
    - X_val, y_val: Tensor
    - X_test, y_test: Tensor
    - split_info: dict
    - norm_params: dict
    - inverse_fn: Callable
    - model_trained: nn.Module
    - residual_std: float
    - stop_flag: bool
    --
    + load_data(payload_json): TimeSeriesData
    + preprocess_data(): Tuple[Tensor, Tensor]
    + normalize_data(): void
    + split_data_three_way(): dict
    + create_training_generator(): Generator
    + run_training(): Generator
    + run_validation(): Generator
    + run_prediction_test(strategy): Generator
    + execute_full_pipeline(): Generator
  }

  class DatasetManager {
    - data_server_url: str
    --
    + get_available_datasets(): List[str]
    + fetch_dataset(name, start, end): TimeSeriesData
  }

  class ModelManager {
    - models_dir: str
    --
    + save_model(model, config, norm_params, metrics): str
    + load_model(model_path): nn.Module
    + list_models(): List[dict]
  }
}

package "Modules Entrainement" #FDF2E9 {
  class train_MLP <<Generator>> {
    + __call__(X, y, hidden_size, nb_couches,
      dropout_rate, activation, loss_name,
      optimizer_name, learning_rate,
      batch_size, epochs, device): Generator
  }
  
  class train_CNN <<Generator>> {
    + __call__(X, y, hidden_size, nb_couches,
      kernel_size, stride, padding,
      activation, loss_name, ...): Generator
  }
  
  class train_LSTM <<Generator>> {
    + __call__(X, y, hidden_size, nb_couches,
      bidirectional, batch_first,
      loss_name, ...): Generator
  }
}

package "Modules de Test" #FDEDEC {
  class test_model_validation <<Generator>> {
    + __call__(model, X_val, y_val, device,
      batch_size, inverse_fn, idx_start): Generator
  }
  
  class predict_multistep <<Generator>> {
    + __call__(model, values, norm_stats,
      window_size, n_steps, device,
      inverse_fn, config, residual_std,
      y_true, idx_start): Generator
  }
  
  enum PredictionStrategy {
    ONE_STEP
    RECALIBRATION
    RECURSIVE
    DIRECT
  }
}

FastAPIApp --> TrainingPipeline : crée
TrainingPipeline --> DatasetManager : utilise
TrainingPipeline --> ModelManager : utilise
TrainingPipeline ..> train_MLP : si MLP
TrainingPipeline ..> train_CNN : si CNN
TrainingPipeline ..> train_LSTM : si LSTM
TrainingPipeline ..> test_model_validation : phase 2
TrainingPipeline ..> predict_multistep : phase 3
predict_multistep --> PredictionStrategy

@enduml""",

    # Diagramme de Classes - Serveur Data
    "class_serveur_data": """@startuml
skinparam classAttributeIconSize 0
skinparam backgroundColor #FEFEFE
skinparam class {
    BackgroundColor #F8F9FA
    BorderColor #2C3E50
}

title Diagramme de Classes - Serveur Data (FastAPI + JSON)

package "API FastAPI" #E8F6F3 {
  class FastAPIApp <<Controller>> {
    + info_all(req: ChoixDatasetRequest): dict
    + data_solo(payload: ChoixDatasetRequest2): dict
    + add_dataset(packet: AddDatasetPacket): dict
    + data_suppression(payload: deleteDatasetRequest): str
    + model_all(req: ChoixModelerequest): dict
    + model_add(payload: newModelRequest): str
    + model_delete(payload: DeleteModelRequest): str
    + contexte_add_solo(paquet: PaquetComplet2): dict
    + contexte_obtenir_solo(payload: ChoixContexteRequest): dict
    + root(): dict
  }
}

package "Services de Gestion" #EBF5FB {
  class DatasetService <<Service>> {
    - DATA_DIR: Path
    --
    + construire_json_datasets(): Dict[str, Any]
    + construire_un_dataset(name, date_debut, date_fin, pas): Dict
    + add_new_dataset(name: str, data: TimeSeriesData): void
    + remove_dataset(name: str): void
    + extraire_infos_dataset(path_json: Path): Tuple
    - parse_ts(ts_str: str): datetime
  }

  class ModelService <<Service>> {
    - MODEL_DIR: Path
    --
    + construire_modeles(): Dict[str, Any]
    + add_new_model(name: str, data: str): void
    + remove_model(name: str): void
  }

  class ContexteService <<Service>> {
    - CONTEXT_DIR: Path
    --
    + contexte_add_mlp(**kwargs): void
    + contexte_add_cnn(**kwargs): void
    + contexte_add_lstm(**kwargs): void
    + transmettre_contexte(name: str): dict
  }
}

package "DTOs Pydantic" #FDF2E9 {
  class TimeSeriesData <<BaseModel>> {
    + timestamps: List[datetime]
    + values: List[Optional[float]]
  }
  
  class ChoixDatasetRequest2 <<BaseModel>> {
    + name: str
    + dates: List[str]
    + pas_temporel: int
  }
  
  class AddDatasetPacket <<BaseModel>> {
    + payload_name: str
    + payload_dataset_add: TimeSeriesDataStr
  }
  
  class PaquetComplet2 <<BaseModel>> {
    + payload: dict
    + payload_model: dict
    + payload_dataset: dict
    + payload_name_model: dict
  }
}

package "Stockage Fichiers" #FDEDEC {
  folder "datasets/" as DS
  folder "models/" as MD
  folder "contextes/" as CTX
}

FastAPIApp --> DatasetService
FastAPIApp --> ModelService
FastAPIApp --> ContexteService

DatasetService --> DS : lit/écrit
ModelService --> MD : lit/écrit
ContexteService --> CTX : lit/écrit

@enduml""",

    # Diagramme de Classes - DTOs Pydantic
    "class_pydantic": """@startuml
skinparam classAttributeIconSize 0
skinparam backgroundColor #FEFEFE

title Diagramme de Classes - DTOs Pydantic (Serveur IA)

package "Classes Pydantic" #EBF5FB {
  class TimeSeriesData <<BaseModel>> {
    + timestamps: List[datetime]
    + values: List[Optional[float]]
  }
  
  class PaquetComplet <<BaseModel>> {
    + Parametres_temporels: Optional[Parametres_temporels]
    + Parametres_choix_reseau_neurones: Optional[Parametres_choix_reseau_neurones]
    + Parametres_choix_loss_fct: Optional[Parametres_choix_loss_fct]
    + Parametres_optimisateur: Optional[Parametres_optimisateur]
    + Parametres_entrainement: Optional[Parametres_entrainement]
  }
  
  class Parametres_temporels <<BaseModel>> {
    + horizon: Optional[int]
    + portion_decoupage: Optional[float]
  }
  
  class Parametres_choix_reseau_neurones <<BaseModel>> {
    + modele: Optional[Literal["MLP", "LSTM", "GRU", "CNN"]]
  }
  
  class Parametres_choix_loss_fct <<BaseModel>> {
    + fonction_perte: Optional[Literal["MSE", "MAE", "Huber"]]
    + params: Optional[dict]
  }
  
  class Parametres_optimisateur <<BaseModel>> {
    + optimisateur: Optional[Literal["Adam", "SGD", "RMSprop"]]
    + learning_rate: Optional[float]
    + scheduler: Optional[Literal["Plateau", "Cosine", "OneCycle"]]
  }
  
  class Parametres_entrainement <<BaseModel>> {
    + nb_epochs: Optional[int]
    + batch_size: Optional[int]
    + clip_gradient: Optional[float]
  }
  
  class Parametres_archi_reseau_MLP <<BaseModel>> {
    + nb_couches: Optional[int]
    + hidden_size: Optional[int]
    + dropout_rate: Optional[float]
    + fonction_activation: Optional[str]
  }
  
  class Parametres_archi_reseau_CNN <<BaseModel>> {
    + nb_couches: Optional[int]
    + hidden_size: Optional[int]
    + kernel_size: Optional[int]
    + stride: Optional[int]
    + padding: Optional[int]
  }
  
  class Parametres_archi_reseau_LSTM <<BaseModel>> {
    + nb_couches: Optional[int]
    + hidden_size: Optional[int]
    + bidirectional: Optional[bool]
    + batch_first: Optional[bool]
  }
}

PaquetComplet *-- Parametres_temporels
PaquetComplet *-- Parametres_choix_reseau_neurones
PaquetComplet *-- Parametres_choix_loss_fct
PaquetComplet *-- Parametres_optimisateur
PaquetComplet *-- Parametres_entrainement

@enduml""",

    # Diagramme de Séquence - Entraînement
    "sequence_training": """@startuml
skinparam backgroundColor #FEFEFE
skinparam sequenceArrowThickness 2

title Diagramme de Séquence - Pipeline Entrainement Complet

actor Utilisateur as U
participant "UI\\n(Tkinter)" as UI #E8F6F3
participant "Serveur IA\\n(FastAPI)" as IA #EBF5FB
participant "Serveur Data\\n(FastAPI)" as DATA #FDF2E9

== Phase 0: Configuration ==
U -> UI: Configure paramètres
UI -> UI: Formatter_JSON_global()

== Phase 1: Récupération Dataset ==
U -> UI: Clic "Start"
activate UI
UI -> IA: POST /datasets/fetch_dataset
activate IA
IA -> DATA: POST /datasets/data_solo
activate DATA
DATA --> IA: {timestamps, values}
deactivate DATA
IA --> UI: {status: "success", data: {...}}
deactivate IA

== Phase 2: Entraînement (SSE) ==
UI -> IA: POST /train_full
activate IA

IA -> IA: TrainingPipeline()
IA -> IA: preprocess + normalize
IA -> IA: split 80/10/10

IA -->> UI: SSE: {"type": "split_info", ...}

loop Pour chaque epoch
  IA -> IA: Forward + Backward
  IA -->> UI: SSE: {"type": "epoch", "avg_loss": 0.123}
  UI -> UI: update_plot()
end

== Phase 3: Validation ==
IA -->> UI: SSE: {"type": "phase", "phase": "validation"}

loop Pour chaque point validation
  IA -> IA: model(X_val[i])
  IA -->> UI: SSE: {"type": "val_pair", "y": [...], "yhat": [...]}
end

IA -> IA: compute_residual_std()
IA -->> UI: SSE: {"type": "val_end", "residual_std": 0.15}

== Phase 4: Test Prédictif ==
IA -->> UI: SSE: {"type": "phase", "phase": "prediction"}

loop Pour chaque pas
  IA -> IA: predict_multistep()
  IA -> IA: Calcul intervalle confiance
  IA -->> UI: SSE: {"type": "pred_point", ...}
end

IA -->> UI: SSE: {"type": "fin_pipeline", "done": 1}
deactivate IA

UI -> UI: Affiche graphiques
deactivate UI

@enduml""",

    # Diagramme de Séquence - Gestion Datasets
    "sequence_datasets": """@startuml
skinparam backgroundColor #FEFEFE
skinparam sequenceArrowThickness 2

title Diagramme de Séquence - Gestion des Datasets

actor Utilisateur as U
participant "UI\\n(Tkinter)" as UI #E8F6F3
participant "Serveur IA\\n(FastAPI)" as IA #EBF5FB
participant "Serveur Data\\n(FastAPI)" as DATA #FDF2E9

== Chargement Initial ==
U -> UI: Lance application
activate UI
UI -> IA: POST /datasets/info_all
activate IA
IA -> DATA: POST /datasets/info_all
activate DATA
DATA -> DATA: construire_json_datasets()
DATA --> IA: {"EURO": {...}, "BTC": {...}}
deactivate DATA
IA --> UI: JSON Datasets
deactivate IA
UI -> UI: JSON_Datasets = data
deactivate UI

== Ajout Dataset ==
U -> UI: Clic "Charger Dataset"
activate UI
UI -> UI: filedialog.askopenfilename()
UI -> UI: Valide avec Pydantic

alt Validation OK
  UI -> IA: POST /datasets/add_dataset
  activate IA
  IA -> DATA: POST /datasets/add_dataset
  activate DATA
  DATA -> DATA: add_new_dataset()
  DATA --> IA: {"ok": true}
  deactivate DATA
  IA --> UI: {"ok": true}
  deactivate IA
  UI -> UI: rafraichir_liste()
else Erreur
  UI -> UI: messagebox.showwarning()
end
deactivate UI

== Suppression Dataset ==
U -> UI: Clic "Supprimer"
activate UI
UI -> IA: POST /datasets/data_suppression_proxy
activate IA
IA -> DATA: POST /datasets/data_supression
activate DATA
DATA -> DATA: remove_dataset()
DATA --> IA: "supprimé"
deactivate DATA
IA --> UI: "supprimé"
deactivate IA
UI -> UI: rafraichir_liste()
deactivate UI

@enduml""",

    # Diagramme de Séquence - Stratégies Prédiction
    "sequence_prediction": """@startuml
skinparam backgroundColor #FEFEFE
skinparam sequenceArrowThickness 2

title Diagramme de Séquence - Stratégies de Prédiction

participant "TrainingPipeline" as TP #E8F6F3
participant "predict_multistep" as PM #EBF5FB
participant "Model\\n(PyTorch)" as M #FDEDEC

TP -> PM: predict_multistep(model, config)
activate PM

PM -> PM: model.eval()
PM -> PM: Extraire fenêtre initiale

PM -->> TP: yield {"type": "pred_start"}

alt strategy == ONE_STEP
  loop Pour chaque pas
    PM -> M: forward(window)
    M --> PM: pred_norm
    PM -> PM: denormalize
    PM -> PM: uncertainty = residual_std
    PM -> PM: low/high = pred ± z*σ
    PM -->> TP: yield {"type": "pred_point"}
    PM -> PM: Recalibration immédiate
  end

else strategy == RECALIBRATION
  loop Pour chaque pas
    PM -> M: forward(window)
    M --> PM: pred_norm
    PM -> PM: uncertainty = σ * sqrt(steps+1)
    PM -->> TP: yield {"type": "pred_point"}
    alt steps >= recalib_every
      PM -> PM: Recalibration
    else
      PM -> PM: window[-1] = pred
    end
  end

else strategy == RECURSIVE
  loop Pour chaque pas
    PM -> M: forward(window)
    M --> PM: pred_norm
    PM -> PM: uncertainty croissante
    PM -->> TP: yield {"type": "pred_point"}
    PM -> PM: window[-1] = pred (jamais recalib)
  end

else strategy == DIRECT
  loop Chunks de H pas
    PM -> M: forward(window)
    M --> PM: [pred_1, ..., pred_H]
    loop h = 1 à H
      PM -->> TP: yield {"type": "pred_point"}
    end
    PM -> PM: Shift window de H
  end
end

PM -> PM: compute_metrics()
PM -->> TP: yield {"type": "pred_end", "metrics": {...}}

deactivate PM

@enduml""",

    # Diagramme de Composants
    "component": """@startuml
skinparam backgroundColor #FEFEFE
skinparam component {
    BackgroundColor #F8F9FA
    BorderColor #2C3E50
}

title Diagramme de Composants - Architecture MLApp

package "Client" #E8F6F3 {
  [Interface Tkinter/CTk] as UI
  component "Matplotlib" as MPL
  component "Threading" as THR
  
  UI --> MPL : visualisation
  UI --> THR : async
}

package "Serveur IA" #EBF5FB {
  [FastAPI App] as IA_API
  component "TrainingPipeline" as TP
  component "PyTorch Models" as PT
  
  package "Modules" {
    [train_MLP]
    [train_CNN]
    [train_LSTM]
    [test_validation]
    [predict_multistep]
  }
  
  IA_API --> TP
  TP --> PT
  TP --> [train_MLP]
  TP --> [train_CNN]
  TP --> [train_LSTM]
  TP --> [test_validation]
  TP --> [predict_multistep]
}

package "Serveur Data" #FDF2E9 {
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

UI -down-> IA_API : HTTP/SSE\\nport 8000
IA_API -down-> DATA_API : HTTP\\nport 8001

@enduml"""
}

# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Créer le dossier de sortie
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Génération des diagrammes UML pour MLApp")
    print(f"Dossier de sortie: {output_dir}")
    print("=" * 60)
    
    # Toujours générer les fichiers .puml
    generate_puml_files(output_dir)
    
    success = 0
    failed = 0
    
    print(f"\nGénération des fichiers {args.format.upper()}...")
    
    for name, code in DIAGRAMS.items():
        if args.plantuml_jar:
            # Utiliser le JAR local
            if generate_with_jar(name, code, output_dir, args.plantuml_jar, args.format):
                success += 1
            else:
                failed += 1
        else:
            # Utiliser le serveur web
            if generate_with_server(name, code, output_dir, args.format):
                success += 1
            else:
                failed += 1
    
    print("=" * 60)
    print(f"Terminé: {success} succès, {failed} échecs")
    
    if failed > 0 and not args.plantuml_jar:
        print("\n⚠️  Le serveur PlantUML peut bloquer certaines requêtes.")
        print("   Solutions alternatives:")
        print("   1. Installer PlantUML localement:")
        print("      brew install plantuml")
        print(f"      plantuml -tpng {output_dir}/puml/*.puml -o {output_dir}")
        print("   2. Ou télécharger plantuml.jar et utiliser:")
        print(f"      python {sys.argv[0]} --plantuml-jar /path/to/plantuml.jar")
        print("   3. Ou utiliser l'extension VS Code 'PlantUML'")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
