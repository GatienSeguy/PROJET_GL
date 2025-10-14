from fastapi import FastAPI
from typing import Optional, Tuple, Literal, List
from pydantic import BaseModel, Field, conint, confloat
from datetime import date,datetime

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




class Parametres_temporels(BaseModel):
    horizon: Optional[int] = Field(None, description="Nombre de pas temporels à prédire")
    dates: Optional[List[str]] = Field(None, description="Période de début/fin (AAAA-MM-JJ)")
    pas_temporel: Optional[int] = Field(None, description="Pas temporel entre deux points")
    portion_decoupage: Optional[confloat(gt=0, lt=1)] = Field(None, description="Proportion de découpage train/test")




class Parametres_archi_reseau(BaseModel):
    """
    Paramétrage du réseau de neurones
    (structure interne du modèle)
    """

    # --- 1) Nombre de couches (profondeur) ---
    nb_couches: conint(ge=1, le=100) = Field(
        ...,
        description="Profondeur du réseau (ex: 2 à 6)"
    )

    # --- 2) Taille cachée / latente ---
    hidden_size: conint(gt=0) = Field(
        ...,
        description="Dimension des représentations internes (ex: 128)"
    )

    # --- 3) Taux de dropout ---
    dropout_rate: confloat(ge=0.0, le=1.0) = Field(
        0.0,
        description="Fraction de neurones désactivés pendant l'entraînement (ex: 0.1)"
    )

    # --- 4) Fonction d’activation ---
    fonction_activation : Literal["ReLU", "GELU", "tanh", "sigmoid", "leaky_relu"] = Field(
        "ReLU",
        description="Type de fonction d'activation interne (ReLU / GELU / tanh / ...)"
    )


##### Nouvelles calsses
class Parametres_choix_reseau_neurones(BaseModel):
    modele: Optional[Literal["RNN", "LSTM", "GRU", "CNN"]] = Field(None, description="Type de modèle choisi")


class Parametres_choix_loss_fct(BaseModel):
    fonction_perte: Optional[Literal["MSE", "MAE", "Huber"]] = Field(None, description="Fonction de perte")
    params: Optional[dict] = Field(None, description="Paramètres spécifiques de la fonction de perte")

class Parametres_optimisateur(BaseModel):
    optimisateur: Optional[Literal["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta"]] = Field(None)
    learning_rate: Optional[float] = Field(None)
    decroissance: Optional[float] = Field(None)
    scheduler: Optional[Literal["Plateau", "Cosine", "OneCycle", "None"]] = Field(None)
    patience: Optional[int] = Field(None)

class Parametres_entrainement(BaseModel):
    nb_epochs: Optional[int] = Field(None)
    batch_size: Optional[int] = Field(None)
    clip_gradient: Optional[float] = Field(None)
    
class Parametres_visualisation_suivi(BaseModel):
    metriques: Optional[List[str]] = Field(None, description="Liste des métriques suivies pendant l’entraînement")



class PaquetComplet(BaseModel):
    Parametres_temporels: Optional[Parametres_temporels]
    Parametres_choix_reseau_neurones: Optional[Parametres_choix_reseau_neurones]
    Parametres_archi_reseau: Optional[Parametres_archi_reseau]
    Parametres_choix_loss_fct: Optional[Parametres_choix_loss_fct]
    Parametres_optimisateur: Optional[Parametres_optimisateur]
    Parametres_entrainement: Optional[Parametres_entrainement]
    Parametres_visualisation_suivi: Optional[Parametres_visualisation_suivi]

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