from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import xarray as xr
from datetime import datetime
import json

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"


# ----------------------------
# Models
# ----------------------------
class ChoixDatasetRequest(BaseModel):
    message: str

# ----------------------------
# Utils
# ----------------------------
def extraire_infos_dataset(path_json: Path):
    """
    Extraction des infos pour tes datasets JSON :
    - date début : premier timestamp
    - date fin   : dernier timestamp
    - pas        : différence entre t[1] et t[0] (au format 'Xd Xh Xm Xs')
    """

    with open(path_json, "r") as f:
        data = json.load(f)

    timestamps = data.get("timestamps", [])

    if not timestamps or len(timestamps) < 2:
        raise ValueError(f"Dataset {path_json.name} invalide : timestamps insuffisants")

    # Conversion en datetime
    ts = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in timestamps]

    date_debut = timestamps[0]
    date_fin = timestamps[-1]

    # Pas = différence entre les deux premiers timestamps
    delta = ts[1] - ts[0]

    # Formatage du pas temporel
    jours = delta.days
    secondes = delta.seconds
    heures = secondes // 3600
    minutes = (secondes % 3600) // 60
    secs = secondes % 60

    # Création d'un format humain
    parts = []
    if jours != 0:
        parts.append(f"{jours}j")
    if heures != 0:
        parts.append(f"{heures}h")
    if minutes != 0:
        parts.append(f"{minutes}m")
    if secs != 0:
        parts.append(f"{secs}s")

    pas = " ".join(parts) if parts else "0s"

    return date_debut, date_fin, pas

def construire_json_datasets():
    if not DATA_DIR.exists():
        raise RuntimeError(f"Le dossier {DATA_DIR} n’existe pas")

    result = {}

    # Parcours direct des fichiers JSON dans Datas2_test
    for file in DATA_DIR.iterdir():
        if not file.is_file():
            continue
        if not file.suffix.lower() == ".json":
            continue

        dataset_id = file.stem  # "EURO", "CACAO"

        debut, fin, pas = extraire_infos_dataset(file)

        result[dataset_id] = {
            "nom": dataset_id,
            "dates": [debut, fin],
            "pas_temporel": pas
        }
    return result

def construire_un_dataset(name: str, date_debut: str, date_fin: str, pas: str):
    if not DATA_DIR.exists():
        raise RuntimeError(f"Le dossier {DATA_DIR} n’existe pas")

    result = {}

    # Parcours direct des fichiers JSON dans Datas2_test
    for file in DATA_DIR.iterdir():
        if not file.is_file():
            continue
        if not file.suffix.lower() == ".json":
            continue
        if name == file.stem:
            dataset_id = file.stem  # "EURO", "CACAO"
            # Extraction des données
            with open(file, "r") as f:
                data = json.load(f)
            # Construction du JSON comportant les datas liés aux timestamps entre la date_debut et date_fin avec le pas
            
            data_filtered = {}
            timestamps = data.get("timestamps", [])
            values = data.get("values", {})
          
            filtered_timestamps = []
            tnew = date_debut
            for t in timestamps:
                
                if date_debut <= tnew <= date_fin:
                    filtered_timestamps.append(tnew)
                    data_filtered["timestamps"] = filtered_timestamps
                    tnew = tnew + pas
          
            # construction du json des valeurs filtrées
            filtered_values = {}
            for key, vals in values.items():
                filtered_vals = []
                for i, t in enumerate(timestamps):
                    if date_debut <= t <= date_fin:
                        filtered_vals.append(vals[i])
                filtered_values[key] = filtered_vals
            new_id = hash(frozenset(filtered_values.items()))
            result[new_id] = {
                "nom": new_id,
                "dates": [date_debut, date_fin],
                "pas_temporel": pas,
                "data": {
                    "timestamps": filtered_timestamps,
                    "values": filtered_values
                }
            }
            

    return {"error": "Dataset not found"}


# ----------------------------
# Endpoint
# ----------------------------
@app.post("/datasets/info_all")
async def info_all(req: ChoixDatasetRequest):
    if req.message != "choix dataset":
        raise HTTPException(status_code=400, detail="Message inconnu")
    try:
        json_final = construire_json_datasets()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return json_final


@app.post("/dataset/data_solo")
async def info_all(req: ChoixDatasetRequest):
    if req.message != "choix dataset":
        raise HTTPException(status_code=400, detail="Message inconnu")
    try:
        nom = req.name
        date_debut = req.date_debut
        date_fin = req.date_fin
        pas = req.pas

        json_final = construire_un_dataset(nom, date_debut, date_fin, pas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return json_final

