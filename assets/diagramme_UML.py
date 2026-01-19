#!/usr/bin/env python3
"""
Générateur de Diagrammes UML pour MLApp
Génère automatiquement :
- 3 diagrammes de classes (UI, Serveur IA, Serveur Data)
- 1 diagramme de séquence (entraînement complet)
- 1 diagramme de composants (architecture globale)
- 1 diagramme de cas d'utilisation
"""

import os
from pathlib import Path


class UMLDiagramGenerator:
    def __init__(self, output_dir="uml_diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_all(self):
        """Génère tous les diagrammes"""
        print("🎨 Génération des diagrammes UML pour MLApp...")
        
        self.generate_class_diagram_ui()
        self.generate_class_diagram_serveur_ia()
        self.generate_class_diagram_serveur_data()
        self.generate_sequence_diagram()
        self.generate_component_diagram()
        self.generate_use_case_diagram()
        
        print(f"\n✅ Tous les diagrammes ont été générés dans : {self.output_dir}/")
        print("\nPour générer les PNG, installez PlantUML et exécutez :")
        print(f"  plantuml {self.output_dir}/*.puml")
    
    def save_diagram(self, filename, content):
        """Sauvegarde un diagramme PlantUML"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ {filename}")
    
    def generate_class_diagram_ui(self):
        """Diagramme de classes : Interface Utilisateur"""
        diagram = r"""@startuml diagramme_classes_UI
!theme plain
skinparam classAttributeIconSize 0
skinparam backgroundColor #FFFFFF
skinparam roundcorner 10

title Diagramme de Classes - Interface Utilisateur (React + Zustand)

package "Frontend React" {
    
    class UseStore <<Zustand Store>> {
        + config: GlobalConfig
        + modelConfig: ModelConfig
        + isTraining: boolean
        + trainingData: EpochLoss[]
        + testingData: FinalPlotData
        + metrics: MetricsResponse
        --
        + startTraining()
        + stopTraining()
        + addTrainingPoint(epoch, loss)
        + setTestingData(data)
        + setMetrics(metrics)
    }

    class App {
        + render()
    }

    class TrainingControl {
        + handleStart()
        + handleStop()
    }

    class TrainingChart {
        + render(lossSeries)
    }

    class TestingChart {
        + render(testingData)
        + zoomXY()
    }

    class ModelSelector {
        + handleModelChange()
    }

    class NetworkArchitecture {
        + renderLayerConfig()
    }

    class TrainingParams {
        + handleParamChange()
    }
    
    class DatasetList {
        + handleSelect()
    }

    class HorizonConfig {
        + handleHorizonChange()
    }

    ' ==== Services API ====
    class trainingAPI <<Service>> {
        + startTraining(config, modelConfig, callbacks)
        + stopTraining()
    }

    class datasetAPI <<Service>> {
        + fetchDataset(payload)
    }

    ' ==== Relations ====
    App *-- TrainingControl
    App *-- TrainingChart
    App *-- TestingChart
    App *-- ModelSelector
    App *-- NetworkArchitecture
    App *-- TrainingParams
    App *-- DatasetList
    App *-- HorizonConfig

    TrainingControl ..> UseStore : reads/writes
    TrainingChart ..> UseStore : reads
    TestingChart ..> UseStore : reads
    ModelSelector ..> UseStore : writes
    
    TrainingControl ..> trainingAPI : calls
    DatasetList ..> datasetAPI : calls
}

note right of UseStore
  Store global (Zustand)
  Centralise l'état de l'application
  et les données de visualisation
end note

@enduml
"""
        self.save_diagram("01_classes_UI.puml", diagram)
    
    def generate_class_diagram_serveur_ia(self):
        """Diagramme de classes : Serveur IA"""
        diagram = """@startuml diagramme_classes_ServeurIA
!theme plain
skinparam classAttributeIconSize 0
skinparam backgroundColor #FFFFFF
skinparam roundcorner 10

title Diagramme de Classes - Serveur IA (FastAPI + PyTorch)

package "Serveur IA" {
    
    ' ==== API FastAPI ====
    class FastAPIApp {
        - app : FastAPI
        - router : APIRouter
        --
        + post_train_full(payload: PaquetComplet, payload_model: dict) : StreamingResponse
        + get_root() : dict
        + get_models() : list
        + post_predict(data: dict) : dict
    }
    
    ' ==== Modèles Pydantic ====
    class TimeSeriesData <<Pydantic>> {
        + timestamps : List[datetime]
        + values : List[Optional[float]]
        --
        + model_post_init() : void
    }
    
    class PaquetComplet <<Pydantic>> {
        + Parametres_temporels : Optional[Parametres_temporels]
        + Parametres_choix_reseau_neurones : Optional[Parametres_choix_reseau_neurones]
        + Parametres_choix_loss_fct : Optional[Parametres_choix_loss_fct]
        + Parametres_optimisateur : Optional[Parametres_optimisateur]
        + Parametres_entrainement : Optional[Parametres_entrainement]
        + Parametres_visualisation_suivi : Optional[Parametres_visualisation_suivi]
    }
    
    class Parametres_archi_reseau_MLP <<Pydantic>> {
        + nb_couches : Optional[int]
        + hidden_size : Optional[int]
        + dropout_rate : Optional[float]
        + fonction_activation : Optional[str]
    }
    
    class Parametres_archi_reseau_CNN <<Pydantic>> {
        + nb_couches : Optional[int]
        + hidden_size : Optional[int]
        + kernel_size : Optional[int]
        + stride : Optional[int]
        + padding : Optional[int]
        + fonction_activation : Optional[str]
    }
    
    class Parametres_archi_reseau_LSTM <<Pydantic>> {
        + nb_couches : Optional[int]
        + hidden_size : Optional[int]
        + bidirectional : Optional[bool]
        + batch_first : Optional[bool]
    }
    
    ' ==== Modèles PyTorch (Abstract) ====
    abstract class NeuralNet <<PyTorch>> {
        # input_size : int
        # output_size : int
        --
        + {abstract} forward(x: Tensor) : Tensor
        # _init_weights() : void
        + get_num_parameters() : int
    }
    
    ' ==== Modèles Concrets ====
    class MLP <<PyTorch>> {
        - layers : nn.ModuleList
        - dropout : nn.Dropout
        - batch_norm : nn.BatchNorm1d
        - hidden_size : int
        - nb_couches : int
        - activation : str
        --
        + __init__(input_size: int, output_size: int, config: dict)
        + forward(x: Tensor) : Tensor
        - _create_layers() : void
        - _get_activation(name: str) : nn.Module
    }
    
    class CNN <<PyTorch>> {
        - conv_layers : nn.ModuleList
        - fc_layers : nn.ModuleList
        - kernel_size : int
        - stride : int
        - padding : int
        --
        + __init__(input_size: int, output_size: int, config: dict)
        + forward(x: Tensor) : Tensor
        - _calculate_conv_output_size() : int
    }
    
    class LSTM <<PyTorch>> {
        - lstm : nn.LSTM
        - fc : nn.Linear
        - hidden_size : int
        - nb_couches : int
        - bidirectional : bool
        --
        + __init__(input_size: int, output_size: int, config: dict)
        + forward(x: Tensor) : Tensor
        + init_hidden(batch_size: int) : tuple
    }
    
    ' ==== Factory Pattern ====
    class ModelFactory {
        --
        + {static} create(model_type: str, config: dict) : NeuralNet
        - {static} _validate_config(model_type: str, config: dict) : void
    }
    
    ' ==== Strategy Pattern - Loss ====
    class LossFactory {
        --
        + {static} create(loss_name: str, params: dict) : nn.Module
    }
    
    ' ==== Strategy Pattern - Optimizer ====
    class OptimizerFactory {
        --
        + {static} create(optimizer_name: str, model_params, lr: float, weight_decay: float) : Optimizer
    }
    
    ' ==== Entraînement ====
    class Trainer {
        - model : NeuralNet
        - criterion : nn.Module
        - optimizer : Optimizer
        - device : str
        - history : list
        --
        + __init__(model: NeuralNet, criterion, optimizer, device: str)
        + train_epoch(X: Tensor, y: Tensor, batch_size: int) : dict
        + compute_gradient_norm() : float
        + save_checkpoint(path: str) : void
        + load_checkpoint(path: str) : void
    }
    
    ' ==== Préparation des Données ====
    class DataPreprocessor {
        - normalization_params : dict
        --
        + filter_series_by_dates(timestamps, values, dates) : tuple
        + build_supervised_tensors(values, horizon, step) : tuple
        + normalize_data(data: Tensor, method: str) : tuple
        + create_inverse_function(params: dict) : Callable
        + split_train_test(X, y, portion: float) : tuple
    }
    
    ' ==== Testing ====
    class TestingModule {
        --
        + test_model(model, X_test, y_test, inverse_fn, batch_size, device) : Generator
        + compute_metrics(y_true, y_pred) : dict
        - _calculate_mse(y_true, y_pred) : float
        - _calculate_mae(y_true, y_pred) : float
        - _calculate_rmse(y_true, y_pred) : float
        - _calculate_mape(y_true, y_pred) : float
        - _calculate_r2(y_true, y_pred) : float
    }
    
    ' ==== SSE Streaming ====
    class SSEStreamer {
        --
        + {static} format_sse(data: dict) : str
        + {static} stream_training_events(training_gen: Generator) : Generator
    }
    
    ' ==== Proxy vers Serveur Data ====
    class DataServerProxy {
        - base_url : str
        - session : requests.Session
        --
        + get_dataset(dataset_id: str) : TimeSeriesData
        + save_model(model_data: dict) : str
        + get_model(model_id: str) : dict
        + list_datasets() : list
        + list_models() : list
    }
    
    ' ==== Relations ====
    NeuralNet <|-- MLP : hérite
    NeuralNet <|-- CNN : hérite
    NeuralNet <|-- LSTM : hérite
    
    ModelFactory ..> NeuralNet : crée
    ModelFactory ..> MLP : instancie
    ModelFactory ..> CNN : instancie
    ModelFactory ..> LSTM : instancie
    
    FastAPIApp --> PaquetComplet : reçoit
    FastAPIApp --> Trainer : utilise
    FastAPIApp --> DataPreprocessor : utilise
    FastAPIApp --> TestingModule : utilise
    FastAPIApp --> SSEStreamer : utilise
    FastAPIApp --> DataServerProxy : utilise
    
    Trainer --> NeuralNet : entraîne
    Trainer --> LossFactory : utilise
    Trainer --> OptimizerFactory : utilise
    
    PaquetComplet o-- Parametres_archi_reseau_MLP
    PaquetComplet o-- Parametres_archi_reseau_CNN
    PaquetComplet o-- Parametres_archi_reseau_LSTM
}

note right of ModelFactory
  Factory Pattern
  Crée dynamiquement le bon
  type de modèle selon config
end note

note bottom of LossFactory
  Strategy Pattern
  Sélectionne la fonction
  de perte appropriée
end note

note bottom of DataServerProxy
  Proxy Pattern
  Interface unique vers
  le Serveur Data
end note

@enduml
"""
        self.save_diagram("02_classes_ServeurIA.puml", diagram)
    
    def generate_class_diagram_serveur_data(self):
        """Diagramme de classes : Serveur Data"""
        diagram = """@startuml diagramme_classes_ServeurData
!theme plain
skinparam classAttributeIconSize 0
skinparam backgroundColor #FFFFFF
skinparam roundcorner 10

title Diagramme de Classes - Serveur Data (REST API + Stockage)

package "Serveur Data" {
    
    ' ==== API REST ====
    class DataAPI {
        - app : Flask/FastAPI
        - dataset_service : DatasetService
        - model_service : ModelService
        - context_service : ContextService
        --
        + post_datasets(data: dict) : dict
        + get_datasets() : list
        + get_dataset_by_id(id: str) : dict
        + delete_dataset(id: str) : dict
        + post_models(model_data: dict) : dict
        + get_models() : list
        + get_model_by_id(id: str) : dict
        + delete_model(id: str) : dict
    }
    
    ' ==== Modèles de Domaine ====
    class Dataset {
        - id : str
        - name : str
        - timestamps : List[datetime]
        - values : List[float]
        - created_at : datetime
        - size : int
        - date_debut : datetime
        - date_fin : datetime
        - metadata : dict
        --
        + __init__(name: str, timestamps, values)
        + validate() : bool
        + get_date_range() : tuple
        + filter_by_dates(debut: str, fin: str) : Dataset
        + to_dict() : dict
        + from_dict(data: dict) : Dataset
    }
    
    class SavedModel {
        - id : str
        - name : str
        - model_type : str
        - state_dict_path : str
        - created_at : datetime
        - version : str
        - file_size : int
        --
        + __init__(name: str, model_type: str, state_dict: bytes)
        + save_state_dict(data: bytes) : str
        + load_state_dict() : bytes
        + to_dict() : dict
        + from_dict(data: dict) : SavedModel
    }
    
    class Context {
        - id : str
        - model_id : str
        - architecture : dict
        - training_params : dict
        - normalization_params : dict
        - test_metrics : dict
        - training_history : list
        - created_at : datetime
        --
        + __init__(model_id: str, **kwargs)
        + add_training_history(epoch: int, loss: float) : void
        + set_test_metrics(metrics: dict) : void
        + to_dict() : dict
        + from_dict(data: dict) : Context
    }
    
    ' ==== Services Métier ====
    class DatasetService {
        - storage : IStorage
        --
        + create_dataset(data: dict) : Dataset
        + get_all_datasets() : list
        + get_dataset_by_id(id: str) : Dataset
        + delete_dataset(id: str) : bool
        + check_name_exists(name: str) : bool
        - _generate_id() : str
    }
    
    class ModelService {
        - storage : IStorage
        --
        + save_model(model_data: dict) : SavedModel
        + get_all_models() : list
        + get_model_by_id(id: str) : SavedModel
        + delete_model(id: str) : bool
        + get_model_with_context(id: str) : dict
    }
    
    class ContextService {
        - storage : IStorage
        --
        + create_context(context_data: dict) : Context
        + get_context_by_model_id(model_id: str) : Context
        + update_context(id: str, data: dict) : Context
        + delete_context(id: str) : bool
    }
    
    ' ==== Interface de Stockage ====
    interface IStorage {
        + save(key: str, data: Any) : void
        + load(key: str) : Any
        + delete(key: str) : void
        + exists(key: str) : bool
        + list_keys(prefix: str) : list
    }
    
    ' ==== Implémentations du Stockage ====
    class FileSystemStorage {
        - base_path : Path
        - datasets_dir : Path
        - models_dir : Path
        - contexts_dir : Path
        --
        + __init__(base_path: str)
        + save(key: str, data: Any) : void
        + load(key: str) : Any
        + delete(key: str) : void
        + exists(key: str) : bool
        + list_keys(prefix: str) : list
        - _ensure_directories() : void
    }
    
    class DatabaseStorage {
        - connection_string : str
        - engine : Engine
        - session : Session
        --
        + __init__(connection_string: str)
        + save(key: str, data: Any) : void
        + load(key: str) : Any
        + delete(key: str) : void
        + exists(key: str) : bool
        + list_keys(prefix: str) : list
        - _connect() : void
    }
    
    class S3Storage {
        - bucket_name : str
        - client : boto3.Client
        - region : str
        --
        + __init__(bucket: str, region: str)
        + save(key: str, data: Any) : void
        + load(key: str) : Any
        + delete(key: str) : void
        + exists(key: str) : bool
        + list_keys(prefix: str) : list
    }
    
    ' ==== Utilitaires ====
    class ValidationHelper {
        --
        + {static} validate_time_series(timestamps, values) : bool
        + {static} validate_model_data(data: dict) : bool
        + {static} check_unique_name(name: str, existing: list) : bool
    }
    
    class SerializationHelper {
        --
        + {static} serialize_to_json(obj: Any) : str
        + {static} deserialize_from_json(json_str: str) : Any
        + {static} encode_state_dict(state_dict: dict) : str
        + {static} decode_state_dict(encoded: str) : dict
    }
    
    ' ==== Relations ====
    DataAPI --> DatasetService : utilise
    DataAPI --> ModelService : utilise
    DataAPI --> ContextService : utilise
    
    DatasetService --> Dataset : gère
    ModelService --> SavedModel : gère
    ContextService --> Context : gère
    
    DatasetService --> IStorage : utilise
    ModelService --> IStorage : utilise
    ContextService --> IStorage : utilise
    
    IStorage <|.. FileSystemStorage : implémente
    IStorage <|.. DatabaseStorage : implémente
    IStorage <|.. S3Storage : implémente
    
    SavedModel "1" -- "1" Context : associé à
    
    DataAPI ..> ValidationHelper : utilise
    DataAPI ..> SerializationHelper : utilise
}

note right of IStorage
  Strategy Pattern
  Permet de changer facilement
  le backend de stockage
end note

note bottom of DataAPI
  Point d'entrée REST
  pour toutes les opérations
  sur les données et modèles
end note

@enduml
"""
        self.save_diagram("03_classes_ServeurData.puml", diagram)
    
    def generate_sequence_diagram(self):
        """Diagramme de séquence : Entraînement complet"""
        diagram = r"""@startuml diagramme_sequence_entrainement
!theme plain
skinparam backgroundColor #FFFFFF
skinparam sequenceMessageAlign center

title Diagramme de Séquence - Entraînement Complet avec Streaming

actor "Opérateur" as User
participant "UI\n(React)" as UI
participant "Store\n(Zustand)" as Store
participant "TrainingAPI" as TrainAPI
participant "Serveur IA\n(FastAPI)" as IA
participant "Trainer" as Engine
participant "Data\nPreprocessor" as Prep
participant "Model\nFactory" as Factory
participant "SSE\nStreamer" as SSE
participant "Serveur Data\n(REST API)" as Data

== Phase 1 : Configuration ==
User -> UI : Configure paramètres\n(modèle, hyperparamètres, etc.)
UI -> Store : Update config

== Phase 2 : Lancement Entraînement ==
User -> UI : Clique "Lancer"
UI -> Store : startTraining()
UI -> TrainAPI : startTraining(config)
TrainAPI -> IA : POST /train_full
activate IA

== Phase 3 : Récupération Dataset ==
IA -> Data : GET /datasets/{id}
activate Data
alt Succès
    Data -> Data : Charge dataset
    Data --> IA : TimeSeriesData\n(timestamps, values)
else Erreur (Fichier introuvable / Corrompu)
    Data --> IA : 404/500 Error
    IA -> SSE : Stream erreur
    SSE -> TrainAPI : data: {"type":"error", "msg":"..."}
    TrainAPI -> UI : onError()
    UI -> Store : stopTraining()
    deactivate Data
    destroy IA
end
deactivate Data

== Phase 4 : Préparation des Données ==
IA -> Prep : filter_series_by_dates()
activate Prep
Prep --> IA : timestamps_filtrés, values_filtrés
deactivate Prep

IA -> Prep : build_supervised_tensors()
activate Prep
Prep --> IA : X, y (tenseurs)
deactivate Prep

IA -> Prep : normalize_data()
activate Prep
Prep --> IA : X_norm, y_norm, params_norm
deactivate Prep

IA -> Prep : split_train_test()
activate Prep
Prep --> IA : X_train, y_train, X_test, y_test
deactivate Prep

IA -> SSE : Stream événement "split"
SSE -> TrainAPI : data: {"type":"info", "n_train":800, "n_test":200}
TrainAPI -> UI : onEvent(info)
UI -> UI : Affiche info split

== Phase 5 : Instanciation Modèle ==
IA -> Factory : create(model_type, config)
activate Factory
Factory -> Factory : Valide configuration
Factory -> Factory : Instancie MLP/CNN/LSTM
Factory --> IA : model (NeuralNet)
deactivate Factory

IA -> Engine : __init__(model, criterion, optimizer, device)
activate Engine

== Phase 6 : Entraînement avec Streaming ==
loop Pour chaque epoch (1 à nb_epochs)
    Engine -> Engine : Forward pass\ny_pred = model(X_train)
    Engine -> Engine : Calcul loss\nloss = criterion(y_pred, y_train)
    Engine -> Engine : Backward pass\nloss.backward()
    Engine -> Engine : Gradient clipping (si activé)
    Engine -> Engine : optimizer.step()
    Engine -> Engine : Calcul gradient_norm
    
    Engine --> IA : Événement epoch\n{"epoch", "loss", "gradient_norm"}
    IA -> SSE : format_sse(événement)
    SSE -> TrainAPI : data: {"type":"epoch", "loss":0.45}
    TrainAPI -> UI : onEvent(epoch)
    UI -> Store : addTrainingPoint()
    Store -> UI : Update TrainingChart
    
    alt Si bouton Annuler cliqué
        User -> UI : Clique "Arrêter"
        UI -> TrainAPI : stopTraining()
        TrainAPI -> IA : Interruption signal
        IA -> Engine : Stop training
        note right: Entraînement arrêté\nà la fin de l'epoch courante
    end
end

Engine --> IA : Modèle entraîné + historique
deactivate Engine

== Phase 7 : Test Automatique ==
IA -> IA : test_model(model, X_test, y_test, inverse_fn)
activate IA

loop Pour chaque échantillon de test
    IA -> IA : y_pred = model(X_test[i])
    IA -> IA : Dénormalisation\ninverse_fn(y_test[i]), inverse_fn(y_pred)
    IA -> SSE : Stream prédiction
    SSE -> TrainAPI : data: {"type":"pred_point", ...}
    TrainAPI -> UI : onEvent(pred_point)
end

IA -> IA : Calcul métriques finales\n(MSE, MAE, RMSE, MAPE, R²)
IA -> SSE : Stream métriques
SSE -> TrainAPI : data: {"type":"final_plot_data", ...}
TrainAPI -> UI : onComplete()
UI -> Store : setTestingData()
Store -> UI : Update TestingChart
deactivate IA

== Phase 8 : Prédiction future (Optionnel) ==
User -> UI : Définit horizon H
UI -> TrainAPI : predict(horizon=H)
TrainAPI -> IA : POST /predict
activate IA
IA -> Prep : Prépare entrée\n(dernières observations, fenêtres)
activate Prep
Prep --> IA : x_0, inverse_fn
deactivate Prep

loop Pour t = 1..H
    IA -> IA : y_pred_t = model(x_t)
    IA -> IA : Dénormalisation\ninverse_fn(y_pred_t)
    IA -> SSE : Stream prédiction future
    SSE -> TrainAPI : data: {"type":"forecast_step",...}
    IA -> IA : Met à jour x_{t+1}\n(roll/auto-régression)
end

IA --> TrainAPI : Série de prédictions
TrainAPI --> UI : Résultat
deactivate IA

== Phase 9 : Sauvegarde Automatique ==
IA -> IA : Prépare contexte complet\n(archi, params, metrics, history)
IA -> Data : POST /models\n(state_dict + context)
activate Data
Data -> Data : Génère UUID
Data -> Data : Sauvegarde fichier .pth
Data -> Data : Sauvegarde contexte JSON
Data --> IA : {"id": "uuid-1234", "created_at": "..."}
deactivate Data

IA -> SSE : Stream événement "complete"
SSE -> TrainAPI : data: {"type":"fin_pipeline"}
TrainAPI -> UI : onComplete()
deactivate IA

User -> UI : Consulte résultats

@enduml
"""
        self.save_diagram("04_sequence_entrainement.puml", diagram)
    
    def generate_component_diagram(self):
        """Diagramme de composants : Architecture globale simplifiée"""
        diagram = r"""@startuml diagramme_composants_simplifie
!theme plain
skinparam backgroundColor #FFFFFF
skinparam componentStyle rectangle

title Schéma d'Architecture - MLApp (Vue Simplifiée)

cloud "Opérateur" as user

package "Interface Utilisateur" <<Electron App>> {
    [React Frontend] as ui
    [Zustand Store] as store
    [Recharts] as plot
    [API Services] as apis
    
    ui -down-> store : state
    ui -down-> plot : render
    ui -down-> apis : calls
}

package "Serveur IA" <<Backend FastAPI>> {
    [FastAPI] as api
    [Trainer] as train
    [Neural Networks] as models
    [Data Preprocessor] as prep
    [SSE Streamer] as sse
    [Data Proxy] as proxy
    
    api -down-> train : orchestre
    train -down-> models : entraîne
    train -down-> prep : utilise
    api -down-> sse : stream
    api -down-> proxy : communique
}

package "Serveur Data" <<Backend REST>> {
    [REST API] as dataapi
    [Dataset Service] as ds
    [Model Service] as ms
    [Storage] as storage
    
    dataapi -down-> ds : gère datasets
    dataapi -down-> ms : gère modèles
    ds -down-> storage : persiste
    ms -down-> storage : persiste
}

database "Stockage" {
    [Fichiers JSON] as json
    [Modèles .pth] as pth
    [Métadonnées] as meta
    
    storage -down-> json
    storage -down-> pth
    storage -down-> meta
}

' === Relations entre composants ===
user -down-> ui : interactions

apis -right-> api : HTTP/REST\n+ SSE
note on link
  Protocoles :
  - POST /train_full (SSE)
  - GET /models
  - POST /predict
end note

proxy -right-> dataapi : HTTP/REST
note on link
  Protocoles :
  - GET /datasets/{id}
  - POST /models
  - GET /models/{id}
end note

' === Notes importantes ===
note right of ui
  Technologies :
  - React + TypeScript
  - Electron (packaging)
  - Recharts (visu)
end note

note right of api
  Technologies :
  - FastAPI
  - PyTorch
  - Pydantic (validation)
  - SSE (streaming)
end note

note right of dataapi
  Technologies :
  - REST API
  - Stockage fichiers
  - JSON + binaire (.pth)
end note

note as N1
  <b>Règle Architecturale Fondamentale :</b>
  L'Interface Utilisateur ne communique
  JAMAIS directement avec le Serveur Data.
  
  Toutes les communications passent
  obligatoirement par le Serveur IA
  qui fait office de proxy.
end note

@enduml
"""
        self.save_diagram("05_composants_simplifie.puml", diagram)
    
    def generate_use_case_diagram(self):
        """Diagramme de cas d'utilisation"""
        diagram = r"""@startuml diagramme_cas_utilisation
!theme plain
skinparam backgroundColor #FFFFFF
skinparam usecaseBackgroundColor LightYellow
skinparam actorBackgroundColor LightBlue

title Diagramme de Cas d'Utilisation - MLApp

left to right direction

actor "Opérateur Métier" as user
actor "Serveur IA" as ia
actor "Serveur Data" as data

rectangle "MLApp - Application de Prédiction" {

    package "Configuration du Modèle" {
        usecase "Choisir Architecture\n(MLP/CNN/LSTM)" as UC5
        usecase "Configurer\nHyperparamètres" as UC6
        usecase "Définir Paramètres\nTemporels" as UC7
        usecase "Sélectionner\nFonction de Perte" as UC8
        usecase "Configurer\nOptimiseur" as UC9
    }
    
    ' === Configuration ===
    package "Configuration du Modèle" {
        usecase "Choisir Architecture\n(MLP/CNN/LSTM)" as UC5
        usecase "Configurer\nHyperparamètres" as UC6
        usecase "Définir Paramètres\nTemporels" as UC7
        usecase "Sélectionner\nFonction de Perte" as UC8
        usecase "Configurer\nOptimiseur" as UC9
    }
    
    ' === Entraînement ===
    package "Entraînement & Monitoring" {
        usecase "Lancer\nEntraînement" as UC10
        usecase "Visualiser Loss\nen Temps Réel" as UC11
        usecase "Suivre Progression" as UC12
        usecase "Annuler\nEntraînement" as UC13
    }
    
    ' === Évaluation ===
    package "Évaluation & Test" {
        usecase "Tester Modèle\nAutomatiquement" as UC14
        usecase "Consulter Métriques\n(MSE, MAE, R²)" as UC15
        usecase "Visualiser\nPrédictions vs Réel" as UC16
    }
    
    ' === Prédiction ===
    package "Prédiction Future" {
        usecase "Définir Horizon\nde Prédiction" as UC17
        usecase "Générer Prédictions\nFutures" as UC18
        usecase "Exporter Résultats" as UC19
    }
    
    ' === Sauvegarde ===
    package "Gestion des Modèles" {
        usecase "Sauvegarder Modèle\n+ Contexte" as UC20
        usecase "Charger Modèle\nExistant" as UC21
        usecase "Lister Modèles\nDisponibles" as UC22
        usecase "Comparer Modèles" as UC23
    }
}

' === Relations Utilisateur ===
user --> UC1
user --> UC2
user --> UC3
user --> UC4
user --> UC5
user --> UC6
user --> UC7
user --> UC8
user --> UC9
user --> UC10
user --> UC11
user --> UC12
user --> UC13
user --> UC14
user --> UC15
user --> UC16
user --> UC17
user --> UC18
user --> UC19
user --> UC20
user --> UC21
user --> UC22
user --> UC23

' === Dépendances ===
UC10 ..> UC2 : <<include>>
UC10 ..> UC5 : <<include>>
UC10 ..> UC6 : <<include>>
UC14 ..> UC10 : <<extend>>
UC20 ..> UC14 : <<include>>
UC18 ..> UC21 : <<include>>

' === Relations avec Serveurs ===
UC10 --> ia : traite
UC14 --> ia : évalue
UC18 --> ia : prédit

UC1 --> data : stocke
UC2 --> data : récupère
UC20 --> data : sauvegarde
UC21 --> data : charge
UC22 --> data : liste

' === Notes ===
note right of UC10
  Utilise Server-Sent Events (SSE)
  pour le streaming en temps réel
  des métriques d'entraînement
end note

note right of UC20
  Sauvegarde complète :
  - Poids du modèle (.pth)
  - Architecture et hyperparamètres
  - Paramètres de normalisation
  - Métriques de test
  - Historique d'entraînement
end note

note bottom of UC21
  Permet la reproductibilité
  complète des expériences
end note

@enduml
"""
        self.save_diagram("06_cas_utilisation.puml", diagram)


def main():
    """Point d'entrée principal"""
    generator = UMLDiagramGenerator()
    generator.generate_all()
    
    print("\n" + "="*60)
    print("📋 Instructions pour générer les images PNG :")
    print("="*60)
    print("\n1. Installez PlantUML :")
    print("   - macOS: brew install plantuml")
    print("   - Linux: apt-get install plantuml")
    print("   - Windows: téléchargez depuis plantuml.com")
    print("\n2. Générez les PNG :")
    print("   cd uml_diagrams")
    print("   plantuml *.puml")
    print("\n3. Les fichiers PNG seront créés dans le même dossier")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()