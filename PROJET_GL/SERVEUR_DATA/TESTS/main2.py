from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date
import uuid, sqlite3, json

app = FastAPI(title="Serveur Dataset TimeSeries")

# -------------------------
# Modèle pour un point de données (date + valeur)
# -------------------------
class DataPoint(BaseModel):
    date: date
    value: float

# Modèle pour la requête contenant la série temporelle
class DataSetRequest(BaseModel):
    data: list[DataPoint] = Field(..., min_items=1)  # Au moins 1 point
    start_date: date | None = None
    end_date: date | None = None

# -------------------------
# Initialisation de la base SQLite
# -------------------------
def init_db():
    conn = sqlite3.connect("datasets.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        data TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    init_db()

# -------------------------
# Endpoint pour recevoir un dataset
# -------------------------
@app.post("/datasets/")
async def create_dataset(dataset: DataSetRequest):
    # Validation minimale : au moins un point de données
    if not dataset.data:
        raise HTTPException(status_code=400, detail="Le dataset ne contient aucune donnée.")

    # Calculer start_date et end_date si non fournis
    start = dataset.start_date or min(dp.date for dp in dataset.data)
    end = dataset.end_date or max(dp.date for dp in dataset.data)
    if start > end:
        raise HTTPException(status_code=400, detail="start_date ne peut pas être après end_date.")

    # Générer un ID unique
    dataset_id = str(uuid.uuid4())

    # Préparer les données JSON brutes à stocker
    raw_data = [{"date": dp.date.isoformat(), "value": dp.value} for dp in dataset.data]

    # Enregistrer dans SQLite
    conn = sqlite3.connect("datasets.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO datasets (id, start_date, end_date, data) VALUES (?, ?, ?, ?)",
        (dataset_id, start.isoformat(), end.isoformat(), json.dumps(raw_data))
    )
    conn.commit()
    conn.close()

    return {"message": "Dataset enregistré avec succès", "dataset_id": dataset_id, "start_date": start, "end_date": end}

# -------------------------
# Endpoint pour consulter un dataset par ID
# -------------------------
@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    conn = sqlite3.connect("datasets.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, start_date, end_date, data FROM datasets WHERE id=?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Dataset introuvable")

    return {
        "id": row[0],
        "start_date": row[1],
        "end_date": row[2],
        "data": json.loads(row[3])
    }

# -------------------------
# Si on lance ce script directement
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
