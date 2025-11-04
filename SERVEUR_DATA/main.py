# ====================================
# IMPORTs
# ====================================
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict

# ====================================
# MODELS / DATACLASSES
# ====================================

@dataclass
class TimeSeriesData:
    """Représente une série temporelle"""
    timestamps: List[str]
    values: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation JSON"""
        return asdict(self)

# ====================================
# CONFIGURATION
# ====================================

DATASETS_DIR = os.getenv("DATASETS_DIR", "./datasets")

# python -m uvicorn SERVEUR_DATA.main:app --host 0.0.0.0 --port 8001 --reload --reload-dir /Users/gatienseguy/Documents/VSCode/PROJET_GL

app = FastAPI()


# ====================================
# CLASSE GESTION DONNÉES
# ====================================

class DataStorageManager:
    """Gère le stockage et la récupération des fichiers de données"""
    
    def __init__(self, datasets_dir: str = DATASETS_DIR):
        """Initialise le gestionnaire de stockage"""
        self.datasets_dir = datasets_dir
        self._ensure_datasets_dir_exists()
    
    def _ensure_datasets_dir_exists(self):
        """Crée le répertoire des datasets s'il n'existe pas"""
        if not os.path.exists(self.datasets_dir):
            os.makedirs(self.datasets_dir)
    
    def _get_dataset_filepath(self, dataset_name: str) -> str:
        """Retourne le chemin complet d'un fichier dataset"""
        return os.path.join(self.datasets_dir, f"{dataset_name}.json")
    
    def list_available_datasets(self) -> List[str]:
        """
        Récupère la liste des noms des datasets disponibles
        
        Returns:
            List[str]: Liste des noms de datasets (sans extension .json)
            
        Raises:
            Exception: En cas d'erreur d'accès au répertoire
        """
        try:
            if not os.path.exists(self.datasets_dir):
                return []
            
            files = os.listdir(self.datasets_dir)
            datasets = [f.replace('.json', '') for f in files if f.endswith('.json')]
            return sorted(datasets)
        
        except Exception as e:
            raise Exception(f"Erreur lors de la lecture du répertoire datasets: {str(e)}")
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """Vérifie si un dataset existe"""
        filepath = self._get_dataset_filepath(dataset_name)
        return os.path.exists(filepath)
    
    def load_dataset_raw(self, dataset_name: str) -> Dict[str, Any]:
        """
        Charge les données brutes d'un dataset depuis le fichier JSON
        
        Args:
            dataset_name (str): Nom du dataset
        
        Returns:
            Dict: Contenu du fichier JSON
            
        Raises:
            FileNotFoundError: Si le dataset n'existe pas
            ValueError: Si le format JSON est invalide
        """
        filepath = self._get_dataset_filepath(dataset_name)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset '{dataset_name}' non trouvé")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Format JSON invalide pour '{dataset_name}': {str(e)}")
    
    def save_dataset(self, dataset_name: str, data: Dict[str, Any]):
        """
        Sauvegarde un dataset dans le répertoire
        
        Args:
            dataset_name (str): Nom du dataset
            data (Dict): Données à sauvegarder
        """
        filepath = self._get_dataset_filepath(dataset_name)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise Exception(f"Erreur lors de la sauvegarde du dataset: {str(e)}")


# ====================================
# CLASSE TRAITEMENT DONNÉES
# ====================================

class DataProcessor:
    """Traite et filtre les données de séries temporelles"""
    
    @staticmethod
    def validate_time_series_data(data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Valide que les données ont le bon format
        
        Args:
            data (Dict): Données à valider
        
        Returns:
            Tuple[bool, str]: (est_valide, message_erreur)
        """
        if "timestamps" not in data:
            return False, "Champ 'timestamps' manquant"
        
        if "values" not in data:
            return False, "Champ 'values' manquant"
        
        if not isinstance(data["timestamps"], list):
            return False, "'timestamps' doit être une liste"
        
        if not isinstance(data["values"], list):
            return False, "'values' doit être une liste"
        
        if len(data["timestamps"]) != len(data["values"]):
            return False, "Nombre de timestamps et values différents"
        
        if len(data["timestamps"]) == 0:
            return False, "Données vides"
        
        return True, ""
    
    @staticmethod
    def parse_date(date_str: str) -> datetime:
        """
        Parse une date en différents formats
        
        Args:
            date_str (str): Date en string
        
        Returns:
            datetime: Objet datetime
            
        Raises:
            ValueError: Si le format est non reconnu
        """
        formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y"]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Format de date non reconnu: {date_str}")
    
    @staticmethod
    def filter_by_dates(
        timestamps: List[str],
        values: List[float],
        date_start: Optional[str] = None,
        date_end: Optional[str] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Filtre les données par plage de dates
        
        Args:
            timestamps (List[str]): Liste des timestamps
            values (List[float]): Liste des valeurs
            date_start (str, optional): Date de début (YYYY-MM-DD)
            date_end (str, optional): Date de fin (YYYY-MM-DD)
        
        Returns:
            Tuple[List, List]: Données filtrées (timestamps, values)
            
        Raises:
            ValueError: Si les dates sont invalides
        """
        if date_start is None and date_end is None:
            return timestamps, values
        
        filtered_timestamps = []
        filtered_values = []
        
        try:
            start = DataProcessor.parse_date(date_start) if date_start else None
            end = DataProcessor.parse_date(date_end) if date_end else None
            
            for ts, val in zip(timestamps, values):
                ts_date = DataProcessor.parse_date(ts)
                
                if start and ts_date < start:
                    continue
                if end and ts_date > end:
                    continue
                
                filtered_timestamps.append(ts)
                filtered_values.append(val)
        
        except ValueError as e:
            raise ValueError(f"Erreur lors du filtrage par dates: {str(e)}")
        
        return filtered_timestamps, filtered_values
    
    @staticmethod
    def create_time_series_data(
        timestamps: List[str],
        values: List[float]
    ) -> TimeSeriesData:
        """Crée un objet TimeSeriesData"""
        return TimeSeriesData(timestamps=timestamps, values=values)


# ====================================
# CLASSE ORCHESTRATION
# ====================================

class DataService:
    """Service principal pour la gestion des données"""
    
    def __init__(self, datasets_dir: str = DATASETS_DIR):
        """Initialise le service"""
        self.storage = DataStorageManager(datasets_dir)
        self.processor = DataProcessor()
    
    def get_datasets_list(self) -> List[str]:
        """
        Récupère la liste de tous les datasets disponibles
        
        Returns:
            List[str]: Liste des noms de datasets
        """
        return self.storage.list_available_datasets()
    
    def fetch_dataset(
        self,
        dataset_name: str,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None
    ) -> TimeSeriesData:
        """
        Récupère un dataset avec filtrage optionnel par dates
        
        Args:
            dataset_name (str): Nom du dataset
            date_start (str, optional): Date de début
            date_end (str, optional): Date de fin
        
        Returns:
            TimeSeriesData: Les données filtrées
            
        Raises:
            FileNotFoundError: Si le dataset n'existe pas
            ValueError: Si les données sont invalides
        """
        # Charger les données brutes
        raw_data = self.storage.load_dataset_raw(dataset_name)
        
        # Valider le format
        is_valid, error_msg = self.processor.validate_time_series_data(raw_data)
        if not is_valid:
            raise ValueError(f"Format de données invalide: {error_msg}")
        
        # Filtrer par dates
        timestamps, values = self.processor.filter_by_dates(
            raw_data["timestamps"],
            raw_data["values"],
            date_start,
            date_end
        )
        
        if not timestamps:
            raise ValueError(f"Aucune donnée trouvée pour la plage {date_start} à {date_end}")
        
        # Créer et retourner l'objet TimeSeriesData
        return self.processor.create_time_series_data(timestamps, values)
    
    def add_dataset(self, dataset_name: str, data: Dict[str, Any]):
        """
        Ajoute un nouveau dataset
        
        Args:
            dataset_name (str): Nom du dataset
            data (Dict): Données au format {"timestamps": [...], "values": [...]}
            
        Raises:
            ValueError: Si les données ne sont pas valides
        """
        is_valid, error_msg = self.processor.validate_time_series_data(data)
        if not is_valid:
            raise ValueError(f"Format de données invalide: {error_msg}")
        
        self.storage.save_dataset(dataset_name, data)


# ====================================
# ROUTES
# ====================================

# Initialiser le service
data_service = DataService()


@app.get("/datasets/list")
def get_datasets_list():
    """
    Liste tous les datasets disponibles
    
    Response:
    {
        "status": "success",
        "datasets": ["EURO", "GOLD", "BITCOIN", ...]
    }
    """
    try:
        datasets = data_service.get_datasets_list()
        
        return {
            "status": "success",
            "datasets": datasets,
            "count": len(datasets)
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.post("/datasets/fetch")
def fetch_dataset(request_payload: dict):
    """
    Récupère un dataset spécifique avec filtrage optionnel par dates
    
    Request body:
    {
        "name": "EURO",
        "dates": ["2020-01-01", "2023-12-31"]  // optionnel
    }
    
    Response:
    {
        "status": "success",
        "data": {
            "timestamps": ["2020-01-01", ...],
            "values": [1.10, ...]
        }
    }
    """
    try:
        # Validation de la requête
        if "name" not in request_payload:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Champ 'name' manquant"
                }
            )
        
        dataset_name = request_payload["name"]
        
        # Extraction optionnelle des dates
        date_start = None
        date_end = None
        
        if "dates" in request_payload and isinstance(request_payload["dates"], list):
            if len(request_payload["dates"]) >= 1:
                date_start = request_payload["dates"][0]
            if len(request_payload["dates"]) >= 2:
                date_end = request_payload["dates"][1]
        
        # Récupérer le dataset
        time_series = data_service.fetch_dataset(dataset_name, date_start, date_end)
        
        return {
            "status": "success",
            "data": time_series.to_dict()
        }
    
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": str(e)
            }
        )
    
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": str(e)
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.post("/datasets/add")
def add_dataset(request_payload: dict):
    """
    Ajoute un nouveau dataset au serveur
    
    Request body:
    {
        "name": "NOUVEAU",
        "data": {
            "timestamps": ["2020-01-01", ...],
            "values": [1.10, ...]
        }
    }
    
    Response:
    {
        "status": "success",
        "message": "Dataset 'NOUVEAU' ajouté avec succès"
    }
    """
    try:
        if "name" not in request_payload:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Champ 'name' manquant"
                }
            )
        
        if "data" not in request_payload:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Champ 'data' manquant"
                }
            )
        
        dataset_name = request_payload["name"]
        data = request_payload["data"]
        
        # Ajouter le dataset
        data_service.add_dataset(dataset_name, data)
        
        return {
            "status": "success",
            "message": f"Dataset '{dataset_name}' ajouté avec succès"
        }
    
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": str(e)
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )



@app.get("/")
def root():
    """Vérification de l'état du serveur"""
    response = {"message": "Serveur DATA actif !"}
    return response