from fastapi import FastAPI
from typing import Optional, Tuple, Literal, List
from pydantic import BaseModel, Field, conint, confloat
from datetime import date,datetime

# uvicorn main:app --reload   
app = FastAPI()

last_config_tempo = None
last_config_TimeSeries = None
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



# ====================================
# ROUTES
# ====================================



@app.post("/tempoconfig")
def recevoir_TempoConfig(config: ConfigTempo):
    """
    Reçoit uniquement la configuration temporelle
    """
    global last_config_tempo
    last_config_tempo = config
    print("\n" + "="*70)
    print(f" CONFIGURATION REÇUE \n")
    print("="*70)
    print("\n TEMPO :")
    print(f"   - Horizon : {config.horizon} \n")
    print(f"   - Dates   : {config.dates} \n")
    print(f"   - Pas     : {config.pas_temporel} \n")
    print(f"   - Split train : {config.split_train} \n")
    print(f"   - Freq    : {config.freq} \n")
    print(f"################ FIN ############  \n")


    return {
        "status": "OK",
        "resume": {
            "horizon": config.horizon,
        }
    }

@app.post("/timeseries")
def recevoir_SeriesData(series: TimeSeriesData):
    """
    Reçoit les données de séries temporelles
    """
    global last_config_series
    last_config_series = series

    n = len(series.values)
    print("\n" + "="*70)
    print(" DONNÉES SÉRIE TEMPORELLE REÇUES")
    print("="*70)
    print(f"   - Nombre de points : {n}")
    if n > 0:
        print(f"   - Premier timestamp : {series.timestamps[0]} \n")
        print(f"   - Dernier timestamp : {series.timestamps[-1]} \n")
        print(f"   - Première valeur   : {series.values[0]}\n")
        print(f"   - Dernière valeur   : {series.values[-1]} \n")
    print("="*70 + "\n")

    print(f"################ FIN ############  \n")
    return {"status": "OK", "nb_points": n}


@app.get("/")
def accueil():
    """
    Retourne les dernières données reçues via /tempoconfig et /timeseries
    """
    response = {"message": "Serveur IA actif !"}

    if last_config_tempo:
        response["tempo"] = {
            "horizon": last_config_tempo.horizon,
            "dates": last_config_tempo.dates,
            "pas_temporel": last_config_tempo.pas_temporel,
            "split_train": last_config_tempo.split_train,
            "freq": last_config_tempo.freq
        }
    else:
        response["tempo"] = "Aucune configuration temporelle reçue."

    if last_config_series:
        response["series"] = {
            "nb_points": len(last_config_series.values),
            "first": {
                "timestamp": last_config_series.timestamps[0],
                "value": last_config_series.values[0]
            },
            "last": {
                "timestamp": last_config_series.timestamps[-1],
                "value": last_config_series.values[-1]
            }
        }
    else:
        response["series"] = "Aucune série reçue."

    return response