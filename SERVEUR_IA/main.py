from fastapi import FastAPI
from typing import Optional, Tuple, Literal, List
from pydantic import BaseModel, Field, conint, confloat
from datetime import date,datetime

from classes import (
    TimeSeriesData,
    Parametres_temporels,
    Parametres_archi_reseau,
    Parametres_choix_reseau_neurones,
    Parametres_choix_loss_fct,
    Parametres_optimisateur,
    Parametres_entrainement,
    Parametres_visualisation_suivi,
    PaquetComplet,
)

## EN LOCAL :
# uvicorn main:app --reload   


## EN SERVEUR SUR WIFI :
# 1- Faire sur terminal de mon mac :
#  ipconfig getifaddr en0  

# 2- Faire pour lancer le serveur
# cd /Users/gatienseguy/Documents/VSCode/PROJET_GL/SERVEUR_IA
#  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

#3 - à la txrx : mettre : URL = ""http:// IP DE L'ORDI HOST DU SERVEUR :8000" "
app = FastAPI()

last_config_tempo = None
last_config_TimeSeries = None






# ====================================
# ROUTES
# ====================================
@app.post("/train_full")
def recevoir_paquet(paquet: PaquetComplet):
    """
    Reçoit le paquet complet envoyé depuis l'interface Tkinter.
    """

    print("\n" + "="*80)
    print("PAQUET COMPLET REÇU")
    print("="*80)

    if paquet.Parametres_temporels:
        print(" Paramètres temporels :", paquet.Parametres_temporels.model_dump())
    if paquet.Parametres_choix_reseau_neurones:
        print("Choix du modèle :", paquet.Parametres_choix_reseau_neurones.modele)
    if paquet.Parametres_archi_reseau:
        print("Architecture :", paquet.Parametres_archi_reseau.model_dump())
    if paquet.Parametres_choix_loss_fct:
        print(" Fonction de perte :", paquet.Parametres_choix_loss_fct.model_dump())
    if paquet.Parametres_optimisateur:
        print(" Optimisateur :", paquet.Parametres_optimisateur.model_dump())
    if paquet.Parametres_entrainement:
        print(" Entraînement :", paquet.Parametres_entrainement.model_dump())
    if paquet.Parametres_visualisation_suivi:
        print(" Visualisation :", paquet.Parametres_visualisation_suivi.model_dump())

    print("="*80 + "\n")

    return {"status": "OK", "message": "Paquet complet recu et valide "}


@app.post("/tempoconfig")
def recevoir_TempoConfig(config: Parametres_temporels):
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
    print(f"   - Split train : {config.portion_decoupage} \n")
    print(f"################ FIN ############  \n")
    return {"status": "OK", "message": "Paquet complet recu et valide "}




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
    return {"status": "OK", "message": "Paquet complet recu et valide "}




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
            "split_train": last_config_tempo.portion_decoupage,
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