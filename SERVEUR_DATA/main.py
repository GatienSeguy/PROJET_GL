from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import xarray as xr
from datetime import datetime, timedelta
from itertools import zip_longest
from typing import Optional, Tuple, Literal, List

import re

import json

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"


# ----------------------------
# Models
# ----------------------------
class ChoixDatasetRequest(BaseModel):
    message: str
    name: str = None
    # dates: Optional[List[str]] = None
    date_debut: str = None
    date_fin: str = None
    pas_temporel: str = None

class ChoixDatasetRequest2(BaseModel):
    # message: str
    name: str = None
    dates: Optional[List[str]] = None
    # date_debut: str = None
    # date_fin: str = None
    pas_temporel: str = None


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

def construire_un_dataset(name: str, date_debut: str, date_fin: str, pas: int):
    import re
    from datetime import datetime, timedelta

    def parse_pas(pas_str: str) -> timedelta:
        """Parse '1d', '12h', '30m', '15s' ou '1d 12h' en timedelta."""
        if not pas:
            return timedelta(days=1)
        regex = r"(\d+)\s*([dhms])"
        kwargs = {}
        for amount, unit in re.findall(regex, pas):
            n = int(amount)
            if unit == "d":
                kwargs["days"] = kwargs.get("days", 0) + n
            elif unit == "h":
                kwargs["hours"] = kwargs.get("hours", 0) + n
            elif unit == "m":
                kwargs["minutes"] = kwargs.get("minutes", 0) + n
            elif unit == "s":
                kwargs["seconds"] = kwargs.get("seconds", 0) + n
        if not kwargs:
            return timedelta(days=1)
        return timedelta(**kwargs)

    def try_parse_ts(ts_str: str):
        """Essaye plusieurs formats de timestamp, retourne datetime."""
        fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]
        for f in fmts:
            try:
                return datetime.strptime(ts_str, f)
            except Exception:
                continue
        # si aucun format ne passe, lever une erreur
        raise ValueError(f"Timestamp format inconnu: {ts_str}")

    def value_at_nearest(ts_list, val_list, target_dt):
        """Retourne la valeur dans val_list associée au timestamp le plus proche de target_dt.
           ts_list et val_list doivent être de longueurs compatibles. Retourne None si aucune valeur."""
        if not ts_list or not val_list:
            return None
        # calcule les différences absolues et prend l'indice du minimum
        diffs = [abs((t - target_dt).total_seconds()) for t in ts_list]
        idx = int(min(range(len(diffs)), key=lambda i: diffs[i]))
        try:
            return val_list[idx]
        except Exception:
            return None

    # --- vérifications préliminaires ---
    if not DATA_DIR.exists():
        raise RuntimeError(f"Le dossier {DATA_DIR} n’existe pas")

    # parser bornes de dates (date_debut/date_fin attendus en YYYY-MM-DD)
    try:
        d0 = datetime.strptime(date_debut, "%Y-%m-%d")
        d1 = datetime.strptime(date_fin, "%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Format date_debut/date_fin invalide (attendu YYYY-MM-DD): {e}")

    pas_td = parse_pas(pas)

    # protection : pas non nul
    if pas_td.total_seconds() <= 0:
        raise ValueError("Le pas temporel doit être strictement positif.")

    # parcours des fichiers
    for file in DATA_DIR.iterdir():
        if not file.is_file() or file.suffix.lower() != ".json":
            continue
        if name != file.stem:
            continue

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        timestamps = data.get("timestamps", [])
        values = data.get("values", {})

        # parse timestamps du fichier en datetimes
        try:
            ts_dt = [try_parse_ts(t) for t in timestamps]
        except Exception as e:
            print(f"ERROR: impossible de parser les timestamps pour {file.name}: {e}")
            return {"error": "Format timestamps invalide"}

        if not ts_dt:
            return {"error": f"Dataset '{name}' sans timestamps"}

        # générer la grille temporelle au pas demandé (on commence à d0 00:00:00)
        generated = []
        cur = datetime.combine(d0.date(), datetime.min.time())
        # si les timestamps du fichier ont heures, mieux commencer à d0 à 00:00:00, c'est cohérent
        while cur <= datetime.combine(d1.date(), datetime.max.time()):
            generated.append(cur)
            cur += pas_td

        # tronquer generated pour rester dans [first_ts, last_ts] si besoin ? On garde selon d0..d1 demandés
        # conversion en string
        generated_str = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in generated]

        # préparer filtered_values selon le format de 'values'
        filtered_values = {}

        if isinstance(values, dict):
            for key, vals in values.items():
                # sécuriser vals -> list
                if not isinstance(vals, (list, tuple)):
                    try:
                        vals = list(vals)
                    except Exception:
                        print(f"WARN: values '{key}' non itérable (type={type(vals)}); on met une série vide")
                        filtered_values[key] = [None] * len(generated)
                        continue

                # si vals shorter/longer than timestamps, zip coupera automatiquement
                # construire val_series à partir des generated en prenant valeur la plus proche
                series = []
                for target_dt in generated:
                    v = value_at_nearest(ts_dt, vals, target_dt)
                    series.append(v)
                filtered_values[key] = series

        elif isinstance(values, list):
            # cas où values est une liste simple (une seule série)
            series = []
            for target_dt in generated:
                v = value_at_nearest(ts_dt, values, target_dt)
                series.append(v)
            filtered_values["values"] = series

        else:
            print(f"WARN: 'values' format inattendu pour {file.name} (type={type(values)}). On retourne vide.")
            filtered_values = {}

        # debug (optionnel)
        print(f"DEBUG dataset={file.name}, generated timestamps={len(generated_str)}, series keys={list(filtered_values.keys())}")
        for k, v in filtered_values.items():
            print(f"  - {k}: {len(v)} valeurs (ex: {v[:3]})")

        # utiliser le nom du fichier comme clé d'ID (stable et lisible)
        dataset_key = file.stem

        return {
            dataset_key: {
                "nom": name,
                "dates": [date_debut, date_fin],
                "pas_temporel": pas,
                "data": {
                    "timestamps": generated_str,
                    "values": filtered_values["values"]
                }
            }
        }

    return {"error": f"Dataset '{name}' not found"}


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

@app.post("/datasets/data_solo")
async def info_all(payload: ChoixDatasetRequest2):
    print("DATA SERVER received fetch_dataset for:", payload.name) 
    json_final = construire_un_dataset(
        name=payload.name,
        date_debut=payload.dates[0],
        date_fin=payload.dates[1],
        pas=payload.pas_temporel
    )
    print(f"\n")
    print("On est bien")
    print(f"\n")
    
    if "error" in json_final:
        print("t'as fait nawak")
        raise HTTPException(status_code=404, detail=json_final["error"])
    return json_final


