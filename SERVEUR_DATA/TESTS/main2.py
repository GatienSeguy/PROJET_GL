from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import os
import json

app = FastAPI()

# Base de données en mémoire pour stocker les datasets
DB = {}

# Modèle Pydantic pour recevoir les datasets
class Dataset(BaseModel):
    dates: Union[List[str], None] = None
    timestamps: Union[List[str], None] = None
    values: List[Union[float, None]]  # Accepte les null, qu'on filtrera ensuite

SAVE_DIR = "Datas2_test"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/datasets/")
async def create_dataset(dataset: Dataset):
    # Choisir la bonne clé
    dates = dataset.dates if dataset.dates is not None else dataset.timestamps
    if dates is None:
        return {"error": "Le dataset doit contenir 'dates' ou 'timestamps'"}
    
    # Filtrer les null dans values et garder seulement les dates correspondantes
    filtered_dates = []
    filtered_values = []
    for d, v in zip(dates, dataset.values):
        if v is not None:
            filtered_dates.append(d)
            filtered_values.append(v)
    
    if not filtered_values:
        return {"error": "Le dataset ne contient aucune valeur valide."}
    
    # Générer un ID unique
    dataset_id = len(DB) + 1
    
    # Stocker en mémoire
    DB[dataset_id] = {"dates": filtered_dates, "values": filtered_values}
    
    # Sauvegarder dans un fichier JSON local
    save_path = os.path.join(SAVE_DIR, f"dataset_{dataset_id}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(DB[dataset_id], f, ensure_ascii=False, indent=2)
    
    return {"id": dataset_id, "message": "Dataset enregistré avec succès."}

@app.get("/datasets/")
def list_datasets():
    # Retourner tous les IDs stockés
    return {"datasets": list(DB.keys())}
