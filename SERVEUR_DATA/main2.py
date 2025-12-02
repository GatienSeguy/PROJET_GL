from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import uvicorn

# ------------------ ---------
# App & chemins
# ----------------------------

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"

# ----------------------------
# Modèles de requêtes
# ----------------------------

class ChoixDatasetRequest(BaseModel):
    message: str


class ChoixDatasetRequest2(BaseModel):
    name: str
    dates: List[str]        # [date_debut, date_fin]
    pas_temporel: int       # ENTIER : 1 => tous les points, 2 => 1 sur 2, etc.

class newDatasetRequest(BaseModel):
    name: str
    data: Dict[str, Any]

class deleteDatasetRequest(BaseModel):
    name: str
class ChoixModelerequest(BaseModel):
    message: str
# ----------------------------
# Utils
# ----------------------------

def parse_ts(ts_str: str) -> datetime:
    """
    Essaie plusieurs formats de timestamps :
    - 'YYYY-MM-DD HH:MM:SS'
    - 'YYYY-MM-DDTHH:MM:SS'
    - 'YYYY-MM-DD'
    """
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            return datetime.strptime(ts_str, f)
        except ValueError:
            continue
    raise ValueError(f"Format de timestamp inconnu : {ts_str}")


def extraire_infos_dataset(path_json: Path):
    """
    Pour un dataset JSON de la forme :
    {
        "timestamps": [...],
        "values": [...]
    }

    -> renvoie :
      - date début  : premier timestamp (string brute)
      - date fin    : dernier timestamp (string brute)
      - pas         : delta entre les deux premiers timestamps (affiché en format humain)
    """
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("test1")
    timestamps = data.get("timestamps", [])
    print("test2")
    if not timestamps or len(timestamps) < 2:
        raise ValueError(f"Dataset {path_json.name} invalide : timestamps insuffisants")
    print("test3")
    ts_dt = [parse_ts(timestamps[0]),parse_ts(timestamps[1])]
    print("test4")
    date_debut = timestamps[0]
    date_fin = timestamps[-1]

    delta = ts_dt[1] - ts_dt[0]

    jours = delta.days
    secondes = delta.seconds
    heures = secondes // 3600
    minutes = (secondes % 3600) // 60
    secs = secondes % 60

    parts = []
    if jours:
        parts.append(f"{jours}j")
    if heures:
        parts.append(f"{heures}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs:
        parts.append(f"{secs}s")

    pas = " ".join(parts) if parts else "0s"

    return date_debut, date_fin, pas


def construire_json_datasets() -> Dict[str, Any]:
    """
    Renvoie les métadonnées des datasets présents dans DATA_DIR.
    {
      "EURO": {
          "nom": "EURO",
          "dates": [date_debut, date_fin],
          "pas_temporel": "1h"
      },
      ...
    }
    """
    if not DATA_DIR.exists():
        raise RuntimeError(f"Le dossier {DATA_DIR} n’existe pas")

    result: Dict[str, Any] = {}

    for file in DATA_DIR.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() != ".json":
            continue

        dataset_id = file.stem

        debut, fin, pas = extraire_infos_dataset(file)

        result[dataset_id] = {
            "nom": dataset_id,
            "dates": [debut, fin],
            "pas_temporel": pas
        }

    return result


def construire_un_dataset(name: str, date_debut: str, date_fin: str, pas: int) -> Dict[str, Any]:
    """
    Découpe le dataset `name` selon :

        L = [date_debut : date_fin : pas]

    c’est-à-dire :
      - on prend tous les points avec
            date_debut <= timestamp <= date_fin
      - puis on garde 1 point tous les `pas` indices.

    `pas` est un ENTIER STRICTEMENT POSITIF.
    """

    if not DATA_DIR.exists():
        raise RuntimeError(f"Le dossier {DATA_DIR} n’existe pas")

    # sécurisation du pas
    try:
        step = int(pas)
    except Exception:
        raise ValueError("pas_temporel doit être un entier")
    if step <= 0:
        raise ValueError("pas_temporel doit être strictement positif")

    # parse des bornes
    d0 = parse_ts(date_debut)
    d1 = parse_ts(date_fin)
    if d1 < d0:
        raise ValueError("date_fin doit être >= date_debut")

    # on cherche le bon fichier
    for file in DATA_DIR.iterdir():
        if not file.is_file() or file.suffix.lower() != ".json":
            continue
        if file.stem != name:
            continue

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        timestamps = data.get("timestamps", [])
        values = data.get("values", None)

        if not timestamps:
            return {"error": f"Dataset '{name}' sans timestamps"}
        if values is None:
            return {"error": f"Dataset '{name}' sans 'values'"}

        ts_dt = [parse_ts(t) for t in timestamps]

        # indices dans l'intervalle [d0, d1]
        idx_in_range = [
            i for i, t in enumerate(ts_dt)
            if d0 <= t <= d1
        ]

        if not idx_in_range:
            return {"error": f"Aucune donnée dans l'intervalle demandé pour '{name}'"}

        # L = [start : end : step]
        start_idx = idx_in_range[0]
        end_idx = idx_in_range[-1]

        indices_final = list(range(start_idx, end_idx + 1, step))

        # timestamps filtrés
        filtered_ts = [timestamps[i] for i in indices_final]

        # values filtrés
        data_out: Dict[str, Any] = {"timestamps": filtered_ts}

        if isinstance(values, list):
            filtered_values = [values[i] for i in indices_final]
            data_out["values"] = filtered_values

        elif isinstance(values, dict):
            # multi-séries
            for k, v in values.items():
                if not isinstance(v, list):
                    raise ValueError(
                        f"Pour les values dict, chaque entrée doit être une liste (clé={k})"
                    )
                data_out[k] = [v[i] for i in indices_final]
        else:
            return {"error": f"Format de 'values' inattendu pour '{name}' (type={type(values)})"}

        print(f"DEBUG dataset={name}, nb_total={len(timestamps)}, nb_filtré={len(filtered_ts)}")

        return {
            name: {
                "nom": name,
                "dates": [date_debut, date_fin],
                "pas_temporel": step,
                "data": data_out
            }
        }

    # si on a rien trouvé
    return {"error": f"Dataset '{name}' not found"}

def add_new_dataset(name: str, data: Dict[str, Any]) -> None:
    """
    Ajoute un nouveau dataset dans DATA_DIR avec le nom `name` et les données `data`.
    """
    if not DATA_DIR.exists():
        raise RuntimeError(f"Le dossier {DATA_DIR} n’existe pas")

    path_new_dataset = DATA_DIR / f"{name}.json"

    if path_new_dataset.exists():
        raise ValueError(f"Dataset '{name}' existe déjà et ne peut pas être ajouté")
    
    with open(path_new_dataset, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Dataset '{name}' ajouté avec succès dans {path_new_dataset}")

def remove_dataset(name: str) -> None:
    """
    Supprime le dataset `name` de DATA_DIR.
    """
    if not DATA_DIR.exists():
        raise RuntimeError(f"Le dossier {DATA_DIR} n’existe pas")

    path_dataset = DATA_DIR / f"{name}.json"

    if not path_dataset.exists():
        raise ValueError(f"Dataset '{name}' n'existe pas et ne peut pas être supprimé")

    path_dataset.unlink()

    print(f"Dataset '{name}' supprimé avec succès de {path_dataset}")


      
# ----------------------------
# Endpoints
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


@app.post("/datasets/data_solo")
async def data_solo(payload: ChoixDatasetRequest2):
    print("DATA SERVER received fetch_dataset for:", payload.name)

    try:
        json_final = construire_un_dataset(
            name=payload.name,
            date_debut=payload.dates[0],
            date_fin=payload.dates[1],
            pas=payload.pas_temporel
        )
    except ValueError as e:
        # erreurs de format (dates, pas, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    print("\nOn est bien\n")

    if "error" in json_final:
        print("t'as fait nawak")
        raise HTTPException(status_code=404, detail=json_final["error"])

    return json_final

## Pourquoi il y a une erreur 404 ici ?  modifie copilot stp
##
@app.post("/datasets/data_add")
async def data_addd(payload: newDatasetRequest):
    ## c'est bon ici ?  
    print("DATA SERVER received fetch_dataset for:", payload.name)

    try:
        add_new_dataset(
            name=payload.name,
            data=payload.data
        )
    except ValueError as e:
        # erreurs de format (dates, pas, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    print("\nOn est bien\n")

    

    return "dataset ajouté avec succès"
@app.post("/datasets/data_supression")
async def data_suppression(payload: deleteDatasetRequest):
    print("DATA SERVER received delete_dataset for:", payload.name)

    try:
        remove_dataset(
            name=payload.name
        )
    except ValueError as e:
        # erreurs de format (dates, pas, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    print("\nOn est bien\n")

    return "dataset supprimé avec succès"


if __name__ == "__main__":
    uvicorn.run("main2:app", host="0.0.0.0", port=8001, reload=True)

