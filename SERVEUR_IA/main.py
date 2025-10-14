from fastapi import FastAPI
from typing import Optional, Tuple, Literal, List
from pydantic import BaseModel, Field, conint, confloat
from datetime import date,datetime

# uvicorn main:app --reload   
app = FastAPI()

last_config = None
# ====================================
# MODÈLES PYDANTIC - Classes
# ====================================

class TimeSeriesData(BaseModel):
    """
    Une unique série temporelle : timestamps et valeurs alignés (même longueur).
    """
    timestamps: List[datetime] = Field(
        ..., description="Liste UTC triée croissante (ISO 8601)"
    )
    values: List[Optional[float]] = Field(
        ..., description="Valeurs numériques (Null si manquante), même longueur que timestamps"
    )

     # Mini garde-fou : même taille
    def model_post_init(self, __context) -> None:
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps et values doivent avoir la même longueur")




class ConfigTempo(BaseModel):
    """
    Configuration des paramètres temporels et découpage des données
    (transmise entre serveurs)
    coint(gt = 0) c'est un entier qui vérifie int >0, coint(lt=0) : int < 0
    """

    # --- 1) Horizon de prédiction : H >0 ---
    horizon: conint(gt=0) = Field(
        ...,
        description="Nombre de pas temporels à prédire (ex: 24 pour 24h)"
    )

    # --- 2) Bornes temporelles : vect(date_debut, date_fin)---
    dates: Optional[Tuple[date, date]] = Field(
        None,
        description="Période temporelle utilisée, format (AAAA-MM-JJ, AAAA-MM-JJ)"
    )

    # --- 3) Pas temporel (résolution) : pas_tempo >0---
    pas_temporel: conint(gt=0) = Field(
        1,
        description="Extraction de données tous les p pas temporels"
    )

    # --- 4) Découpage train/test : 0 < pourcentage_train <1---
    split_train: confloat(gt=0, lt=1) = Field(
        0.9,
        description="Proportion des données utilisée pour l'entraînement (ex: 0.9 = 90%)"
    )

    # --- 5) Fréquence temporelle : min, jour , heure, semaine, mois (option) ---
    freq: Optional[Literal["T","H","D","W","M"]] = Field(
        None,
        description="Pas temporel du dataset (T=min, H=heure, D=jour, etc.)"
    )



# Wrapper



# ====================================
# ROUTES
# ====================================



@app.post("/train")
def recevoir_TempoConfig(config: ConfigTempo):
    """
    Reçoit uniquement la configuration temporelle
    """
    global last_config
    last_config = config
    print("\n" + "="*70)
    print(" CONFIGURATION REÇUE")
    print("="*70)
    print("\n TEMPO :")
    print(f"   - Horizon : {config.horizon}")
    print(f"   - Dates   : {config.dates}")
    print(f"   - Pas     : {config.pas_temporel}")
    print(f"   - Split train : {config.split_train}")
    print(f"   - Freq    : {config.freq}")


    return {
        "status": "OK",
        "resume": {
            "horizon": config.horizon,
        }
    }

@app.get("/")
def accueil():
    """
    Retourne les dernières données reçues via /train
    """
    if last_config is None:
        return {"message": "Aucune configuration reçue pour l’instant."}
    
    return {
        "message": "Dernière configuration reçue :",
        "horizon": last_config.horizon,
        "dates": last_config.dates,
        "pas_temporel": last_config.pas_temporel,
        "split_train": last_config.split_train,
        "freq": last_config.freq
    }
