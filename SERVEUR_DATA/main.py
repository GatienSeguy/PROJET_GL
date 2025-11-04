# main.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone
import os, json, hashlib, math

from .classes import DatasetIn, DatasetOut

# python -m uvicorn SERVEUR_DATA.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /Users/gatienseguy/Documents/VSCode/PROJET_GL

# --------------------------
# Répertoires de stockage
# --------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STORE_DIR = os.path.join(BASE_DIR, "store")
MODELS_DIR = os.path.join(STORE_DIR, "models")
DATASETS_DIR = os.path.join(STORE_DIR, "datasets")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# --------------------------
# Petits utilitaires
# --------------------------
def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    # trim + collapse spaces + lower
    return " ".join(s.split()).strip().lower()

def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def json_dump(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def json_load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_dt(x: str) -> datetime:
    """
    Parse robuste d'une date ISO (gère 'Z', offset, sous-secondes).
    Retourne un datetime naïf en UTC pour comparaison simple.
    """
    s = str(x).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception as e:
        raise ValueError(f"Format de date invalide: {x}") from e
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

def _filter_record_by_dates(record: Dict[str, Any], start: str, end: str) -> Dict[str, Any]:
    """
    Filtre record['data'] pour ne garder que start <= t <= end.
    record: {"id","name","time_kind","data":[{"t":..., "v":...},...],...}
    start/end: ISO strings
    Retour: record cloné avec 'data' borné + 'slice' meta.
    """
    if not record or "data" not in record:
        raise ValueError("Record invalide (clé 'data' manquante).")

    t0 = _to_dt(start)
    t1 = _to_dt(end)
    if t1 < t0:
        t0, t1 = t1, t0

    out_data = []
    for p in record.get("data", []):
        t = p.get("t")
        if t is None:
            continue
        try:
            if isinstance(t, (int, float)):
                dt = datetime.fromtimestamp(float(t))
            else:
                dt = _to_dt(str(t))
        except Exception:
            continue
        if t0 <= dt <= t1:
            try:
                v = float(p.get("v"))
            except Exception:
                continue
            out_data.append({"t": t, "v": v})

    new_rec = dict(record)
    new_rec["data"] = out_data
    new_rec["slice"] = {"start": start, "end": end, "n_points": len(out_data)}
    return new_rec

# --------------------------
# App
# --------------------------
app = FastAPI(title="Unified Storage Server", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# --------------------------
# Health
# --------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

# --------------------------
# Helpers internes (listing/lookup)
# --------------------------
def _iter_models_meta() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name in os.listdir(MODELS_DIR):
        if name.endswith(".json"):
            p = os.path.join(MODELS_DIR, name)
            try:
                out.append(json_load(p))
            except Exception:
                continue
    return out

def _find_model_meta_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    for meta in _iter_models_meta():
        if meta.get("id") == model_id:
            return meta
    return None

def _models_by_logical_name(logical_name: str) -> List[Dict[str, Any]]:
    target = _norm(logical_name)
    items = [m for m in _iter_models_meta() if _norm(m.get("logical_name")) == target]
    items.sort(key=lambda m: m.get("created_utc", ""), reverse=True)
    return items

def _iter_dataset_records() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name in os.listdir(DATASETS_DIR):
        if name.endswith(".json"):
            p = os.path.join(DATASETS_DIR, name)
            try:
                out.append(json_load(p))
            except Exception:
                continue
    return out

def _datasets_by_name(ds_name: str) -> List[Dict[str, Any]]:
    target = _norm(ds_name)
    items = [d for d in _iter_dataset_records() if _norm(d.get("name")) == target]
    items.sort(key=lambda d: d.get("created_utc", ""), reverse=True)
    return items

def _model_name_exists(name: str) -> bool:
    return len(_models_by_logical_name(name)) > 0

def _dataset_name_exists(name: str) -> bool:
    return len(_datasets_by_name(name)) > 0

# ==========================
# MODELS: upload, list, download, by-name
# ==========================
@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),  # nom logique OBLIGATOIRE et UNIQUE
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Le nom du modèle (name) est obligatoire et ne doit pas être vide.")
    if _model_name_exists(name):
        raise HTTPException(status_code=409, detail=f"Nom de modèle déjà utilisé: '{name}'")

    allowed_ext = {".pth", ".pt", ".onnx", ".bin", ".safetensors", ".joblib", ".pkl"}
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Extension non autorisée: {ext}")

    uid = uuid4().hex
    safe_name = f"{uid}__{os.path.basename(file.filename)}"
    out_path = os.path.join(MODELS_DIR, safe_name)

    # Écriture en chunks
    with open(out_path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    meta = {
        "id": uid,
        "logical_name": name,
        "logical_name_norm": _norm(name),
        "original_filename": file.filename,
        "stored_filename": safe_name,
        "size_bytes": os.path.getsize(out_path),
        "sha256": sha256_of_file(out_path),
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }
    json_dump(out_path + ".json", meta)
    return {"message": "Modèle sauvegardé", "id": uid, "meta": meta}

@app.get("/models")
def list_models():
    items = _iter_models_meta()
    items.sort(key=lambda m: m.get("created_utc", ""), reverse=True)
    return {"models": items}

@app.get("/models/{model_id}/download")
def download_model(model_id: str):
    meta = _find_model_meta_by_id(model_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Modèle introuvable.")
    bin_path = os.path.join(MODELS_DIR, meta["stored_filename"])
    if not os.path.exists(bin_path):
        raise HTTPException(status_code=404, detail="Fichier binaire introuvable.")
    return FileResponse(bin_path, filename=meta["original_filename"])

# --- Recherche par nom (modèles)
@app.get("/models/by-name/{logical_name}")
def list_models_by_name(logical_name: str):
    items = _models_by_logical_name(logical_name)
    return {"models": items}

@app.get("/models/by-name/{logical_name}/latest")
def get_latest_model_meta(logical_name: str):
    items = _models_by_logical_name(logical_name)
    if not items:
        raise HTTPException(status_code=404, detail="Aucun modèle pour ce nom logique.")
    return items[0]

@app.get("/models/by-name/{logical_name}/download")
def download_latest_model(logical_name: str):
    items = _models_by_logical_name(logical_name)
    if not items:
        raise HTTPException(status_code=404, detail="Aucun modèle pour ce nom logique.")
    meta = items[0]
    bin_path = os.path.join(MODELS_DIR, meta["stored_filename"])
    if not os.path.exists(bin_path):
        raise HTTPException(status_code=404, detail="Fichier binaire introuvable.")
    return FileResponse(bin_path, filename=meta["original_filename"])

# ==========================
# DATASETS: post JSON, list, get, by-name, slice
# ==========================
@app.post("/datasets", response_model=DatasetOut)
def add_dataset(payload: DatasetIn):
    # unicité du nom
    if not payload.name or not payload.name.strip():
        raise HTTPException(status_code=400, detail="Le nom du dataset (name) est obligatoire.")
    if _dataset_name_exists(payload.name):
        raise HTTPException(status_code=409, detail=f"Nom de dataset déjà utilisé: '{payload.name}'")

    time_kind = "dates" if payload.dates else ("timestamps" if payload.timestamps else None)

    times = payload.dates if payload.dates is not None else payload.timestamps
    if times is not None and len(times) != len(payload.values):
        raise HTTPException(status_code=400, detail="Longueurs incohérentes entre time et values.")

    pairs = []
    if times is None:
        for i, v in enumerate(payload.values):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            pairs.append((i, float(v)))
    else:
        for t, v in zip(times, payload.values):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            pairs.append((t, float(v)))

    if not pairs:
        raise HTTPException(status_code=400, detail="Aucun point valide après filtrage (toutes valeurs nulles ?).")

    ds_id = uuid4().hex
    record = {
        "id": ds_id,
        "name": payload.name,
        "name_norm": _norm(payload.name),
        "time_kind": time_kind,
        "data": [{"t": t, "v": v} for (t, v) in pairs],
        "meta": payload.meta or {},
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }

    out_json = os.path.join(DATASETS_DIR, f"{ds_id}.json")
    json_dump(out_json, record)

    preview = {
        "head": record["data"][:5],
        "tail": record["data"][-5:],
        "min": min(d["v"] for d in record["data"]),
        "max": max(d["v"] for d in record["data"]),
    }
    return DatasetOut(
        id=ds_id,
        name=payload.name,
        n_points=len(record["data"]),
        time_kind=time_kind,
        preview=preview,
        meta=record["meta"],
    )

@app.get("/datasets")
def list_datasets():
    items = []
    for rec in _iter_dataset_records():
        items.append({
            "id": rec.get("id"),
            "name": rec.get("name"),
            "n_points": len(rec.get("data", [])),
            "time_kind": rec.get("time_kind"),
            "created_utc": rec.get("created_utc"),
        })
    items.sort(key=lambda r: r.get("created_utc", ""), reverse=True)
    return {"datasets": items}

@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    path = os.path.join(DATASETS_DIR, f"{dataset_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset introuvable.")
    return json_load(path)

@app.get("/datasets/{dataset_id}/download")
def download_dataset_file(dataset_id: str):
    path = os.path.join(DATASETS_DIR, f"{dataset_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset introuvable.")
    return FileResponse(path, filename=f"{dataset_id}.json", media_type="application/json")

# --- Recherche par nom (datasets)
@app.get("/datasets/by-name/{ds_name}")
def list_datasets_by_name(ds_name: str):
    items = _datasets_by_name(ds_name)
    out = [{
        "id": it.get("id"),
        "name": it.get("name"),
        "n_points": len(it.get("data", [])),
        "time_kind": it.get("time_kind"),
        "created_utc": it.get("created_utc"),
    } for it in items]
    return {"datasets": out}

@app.get("/datasets/by-name/{ds_name}/latest")
def get_latest_dataset(ds_name: str):
    items = _datasets_by_name(ds_name)
    if not items:
        raise HTTPException(status_code=404, detail="Aucun dataset pour ce nom.")
    return items[0]

@app.get("/datasets/by-name/{ds_name}/download")
def download_latest_dataset_file(ds_name: str):
    items = _datasets_by_name(ds_name)
    if not items:
        raise HTTPException(status_code=404, detail="Aucun dataset pour ce nom.")
    latest = items[0]
    ds_id = latest.get("id")
    if not ds_id:
        raise HTTPException(status_code=500, detail="Dataset corrompu (id manquant).")
    path = os.path.join(DATASETS_DIR, f"{ds_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Fichier dataset introuvable.")
    return FileResponse(path, filename=f"{ds_id}.json", media_type="application/json")

# --- SLICE par nom + dates (GET)
@app.get("/datasets/by-name/{ds_name}/slice")
def get_dataset_slice(ds_name: str, start: str, end: str):
    """
    Renvoie le DERNIER dataset pour ce nom, découpé entre [start, end].
    start/end: ISO strings ('YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SSZ', etc.)
    """
    items = _datasets_by_name(ds_name)
    if not items:
        raise HTTPException(status_code=404, detail="Aucun dataset pour ce nom.")

    latest = items[0]
    ds_id = latest.get("id")
    if not ds_id:
        raise HTTPException(status_code=500, detail="Dataset corrompu (id manquant).")

    path = os.path.join(DATASETS_DIR, f"{ds_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Fichier dataset introuvable.")

    record = json_load(path)
    try:
        sliced = _filter_record_by_dates(record, start, end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return sliced

# --- SLICE par payload (POST)
class SliceIn(BaseModel):
    name: str = Field(..., description="Nom du dataset")
    dates: List[str] = Field(..., min_items=2, max_items=2, description="[start, end]")

@app.post("/datasets/slice")
def post_dataset_slice(payload: SliceIn):
    items = _datasets_by_name(payload.name)
    if not items:
        raise HTTPException(status_code=404, detail="Aucun dataset pour ce nom.")

    latest = items[0]
    ds_id = latest.get("id")
    if not ds_id:
        raise HTTPException(status_code=500, detail="Dataset corrompu (id manquant).")

    path = os.path.join(DATASETS_DIR, f"{ds_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Fichier dataset introuvable.")

    record = json_load(path)
    start, end = payload.dates[0], payload.dates[1]
    try:
        sliced = _filter_record_by_dates(record, start, end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return sliced

# ==========================
# INVENTORY (listes combinées)
# ==========================
@app.get("/inventory")
def inventory():
    models = []
    for meta in _iter_models_meta():
        models.append({
            "id": meta.get("id"),
            "logical_name": meta.get("logical_name"),
            "original_filename": meta.get("original_filename"),
            "size_bytes": meta.get("size_bytes"),
            "sha256": meta.get("sha256"),
            "created_utc": meta.get("created_utc"),
        })
    models.sort(key=lambda m: m.get("created_utc", ""), reverse=True)

    datasets = []
    for rec in _iter_dataset_records():
        datasets.append({
            "id": rec.get("id"),
            "name": rec.get("name"),
            "n_points": len(rec.get("data", [])),
            "time_kind": rec.get("time_kind"),
            "created_utc": rec.get("created_utc"),
            "meta": rec.get("meta", {}),
        })
    datasets.sort(key=lambda r: r.get("created_utc", ""), reverse=True)

    return {
        "nb_models": len(models),
        "nb_datasets": len(datasets),
        "models": models,
        "datasets": datasets,
    }

@app.get("/inventory/by-name")
def inventory_by_name(model_name: Optional[str] = None, dataset_name: Optional[str] = None):
    if model_name:
        models = [{
            "id": m.get("id"),
            "logical_name": m.get("logical_name"),
            "original_filename": m.get("original_filename"),
            "size_bytes": m.get("size_bytes"),
            "sha256": m.get("sha256"),
            "created_utc": m.get("created_utc"),
        } for m in _models_by_logical_name(model_name)]
    else:
        models = []

    if dataset_name:
        ds_items = _datasets_by_name(dataset_name)
        datasets = [{
            "id": d.get("id"),
            "name": d.get("name"),
            "n_points": len(d.get("data", [])),
            "time_kind": d.get("time_kind"),
            "created_utc": d.get("created_utc"),
            "meta": d.get("meta", {}),
        } for d in ds_items]
    else:
        datasets = []

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "models": models,
        "datasets": datasets,
        "nb_models": len(models),
        "nb_datasets": len(datasets),
    }

# ==========================
# Vérification disponibilité noms
# ==========================
@app.get("/names/check")
def check_names(model_name: Optional[str] = None, dataset_name: Optional[str] = None):
    out: Dict[str, Any] = {}
    if model_name is not None:
        out["model_name"] = model_name
        out["model_available"] = not _model_name_exists(model_name)
    if dataset_name is not None:
        out["dataset_name"] = dataset_name
        out["dataset_available"] = not _dataset_name_exists(dataset_name)
    if not out:
        return {"detail": "Fournir model_name et/ou dataset_name en query string."}
    return out