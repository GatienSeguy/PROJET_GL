from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import torch
import os
import json

# ====================================
# CONFIGURATION
# ====================================
MODELS_DIR = "saved_models"
DATA_DIR = "Datas"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI()

# ====================================
# CLASSES PYDANTIC
# ====================================
class TimeSeriesData(BaseModel):
    timestamps: List[datetime] = Field(..., description="Liste UTC triée croissante (ISO 8601)")
    values: List[Optional[float]] = Field(..., description="Valeurs numériques (Null si manquante)")
    
    def model_post_init(self, __context) -> None:
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps et values doivent avoir la même longueur")

class RequestDataset(BaseModel):
    dates: Optional[List[str]] = Field(None, description="[date_debut, date_fin] format AAAA-MM-JJ")
    name: Optional[str] = Field(None, description="Nom du fichier JSON (ex: 'series_temp')")

# ====================================
# ROUTES MODÈLES
# ====================================
@app.post("/models/upload/")
async def upload_model(file: UploadFile = File(...)):
    """Reçoit et sauvegarde un modèle .pth avec timestamp unique"""
    if not file.filename.endswith('.pth'):
        raise HTTPException(400, "Seuls les fichiers .pth sont acceptés")
    
    # Identifiant unique : timestamp + nom original
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = file.filename.replace('.pth', '')
    unique_filename = f"{safe_name}_{timestamp}.pth"
    
    save_path = os.path.join(MODELS_DIR, unique_filename)
    
    with open(save_path, "wb") as f:
        f.write(await file.read())
    
    return {
        "message": "Modèle sauvegardé",
        "filename": unique_filename,
        "path": save_path
    }

@app.get("/models/list/")
def list_models():
    """Liste tous les modèles disponibles"""
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    return {"models": models, "count": len(models)}

# ====================================
# ROUTES DONNÉES
# ====================================
@app.post("/data/get/")
def get_filtered_data(request: RequestDataset):
    """Récupère les données filtrées par dates depuis un fichier JSON"""
    
    # Nom du fichier par défaut
    filename = request.name if request.name else "timeseries_data"
    if not filename.endswith('.json'):
        filename += '.json'
    
    filepath = os.path.join(DATA_DIR, filename)
    
    # Vérifier existence
    if not os.path.exists(filepath):
        raise HTTPException(404, f"Fichier {filename} introuvable dans {DATA_DIR}")
    
    # Charger les données
    with open(filepath, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    
    series = TimeSeriesData(**data_dict)
    
    # Filtrage par dates si demandé
    if request.dates and len(request.dates) == 2:
        date_debut = datetime.fromisoformat(request.dates[0])
        date_fin = datetime.fromisoformat(request.dates[1])
        
        filtered_ts = []
        filtered_vals = []
        
        for ts, val in zip(series.timestamps, series.values):
            if date_debut <= ts <= date_fin:
                filtered_ts.append(ts)
                filtered_vals.append(val)
        
        return {
            "timestamps": filtered_ts,
            "values": filtered_vals
        }
    
    # Retour complet si pas de filtre
    return {
        "timestamps": series.timestamps,
        "values": series.values
    }




# App web
@app.get("/")
def root():
    return {
        "message": "Serveur DATA actif",
        "endpoints": {
            "upload_model": "POST /models/upload/",
            "list_models": "GET /models/list/",
            "get_data": "POST /data/get/"
        }
    }