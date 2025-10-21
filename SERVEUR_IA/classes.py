from typing import Optional, Tuple, Literal, List
from pydantic import BaseModel, Field, conint, confloat
from datetime import datetime
# ====================================
# MODÈLES PYDANTIC - Classes
# ====================================

class TimeSeriesData(BaseModel):
    """
    Une unique série temporelle : timestamps et valeurs alignés (même longueur).
    """
    timestamps: List[datetime] = Field(..., description="Liste UTC triée croissante (ISO 8601)")
    values: List[Optional[float]] = Field(..., description="Valeurs numériques (Null si manquante), même longueur que timestamps")
     
     # Mini garde-fou : même taille
    def model_post_init(self, __context) -> None:
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps et values doivent avoir la même longueur")




class Parametres_temporels(BaseModel):
    horizon: Optional[int] = Field(None, description="Nombre de pas temporels à prédire")
    dates: Optional[List[str]] = Field(None, description="Période de début/fin (AAAA-MM-JJ)")
    pas_temporel: Optional[int] = Field(None, description="Pas temporel entre deux points")
    portion_decoupage: Optional[confloat(gt=0, lt=1)] = Field(None, description="Proportion de découpage train/test")





##### Nouvelles calsses
class Parametres_choix_reseau_neurones(BaseModel):
    modele: Optional[Literal["MLP", "LSTM", "GRU", "CNN"]] = Field(None, description="Type de modèle choisi")


class Parametres_choix_loss_fct(BaseModel):
    fonction_perte: Optional[Literal["MSE", "MAE", "Huber"]] = Field(None, description="Fonction de perte")
    params: Optional[dict] = Field(None, description="Paramètres spécifiques de la fonction de perte")

class Parametres_optimisateur(BaseModel):
    optimisateur: Optional[Literal["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta"]] = Field(None)
    learning_rate: Optional[float] = Field(None)
    decroissance: Optional[float] = Field(None)
    scheduler: Optional[Literal["Plateau", "Cosine", "OneCycle","None"]] = Field(None)
    patience: Optional[int] = Field(None)

class Parametres_entrainement(BaseModel):
    nb_epochs: Optional[int] = Field(None)
    batch_size: Optional[int] = Field(None)
    clip_gradient: Optional[float] = Field(None)
    
class Parametres_visualisation_suivi(BaseModel):
    metriques: Optional[List[str]] = Field(None, description="Liste des métriques suivies pendant l’entraînement")






class Parametres_archi_reseau_MLP(BaseModel):
    """
    Paramétrage du réseau de neurones de type MLP
    (structure interne du modèle)
    """

    # --- 1) Nombre de couches (profondeur) ---
    nb_couches: Optional[conint(ge=1, le=100)] = Field(
        None,
        description="Profondeur du réseau (ex: 2 à 6)"
    )

    # --- 2) Taille cachée / latente ---
    hidden_size: Optional[conint(gt=0)] = Field(
        None,
        description="Dimension des représentations internes (ex: 128)"
    )

    # --- 3) Taux de dropout ---
    dropout_rate: Optional[confloat(ge=0.0, le=1.0)] = Field(
        None,
        description="Fraction de neurones désactivés pendant l'entraînement (ex: 0.1)"
    )

    # --- 4) Fonction d’activation ---
    fonction_activation: Optional[Literal["ReLU", "GELU", "tanh", "sigmoid", "leaky_relu"]] = Field(
        None,
        description="Type de fonction d'activation interne (ReLU / GELU / tanh / ...)"
    )



class Parametres_archi_reseau_CNN(BaseModel):
    """
    Paramétrage du réseau de neurones de type CNN
    (structure interne du modèle)
    """

    # --- 1) Nombre de couches (profondeur) ---
    nb_couches: Optional[conint(ge=1, le=100)] = Field(
        None,
        description="Profondeur du réseau (ex: 2 à 6)"
    )

    # --- 2) Taille cachée / latente ---
    hidden_size: Optional[conint(gt=0)] = Field(
        None,
        description="Dimension des représentations internes (ex: 128)"
    )


    # --- 4) Fonction d’activation ---
    fonction_activation: Optional[Literal["ReLU", "GELU", "tanh", "sigmoid", "leaky_relu"]] = Field(
        None,
        description="Type de fonction d'activation interne (ReLU / GELU / tanh / ...)"
    )

    kernel_size: Optional[conint(gt=0)] = Field(
        None,
        description="taille du noyau convolutif"
    )

    stride: Optional[conint(gt=0)] = Field(
        None,
        description="pas d'application du noyau"
    )

    padding: Optional[conint(gt=-1)] = Field(
        None,
        description="Nombre de 0 dans noyau"
    )



class Parametres_archi_reseau_LSTM(BaseModel):
    """
    Paramétrage du réseau de neurones de type LSTM
    (structure interne du modèle)
    """

    # --- 1) Nombre de couches (profondeur) ---
    nb_couches: Optional[conint(ge=1, le=100)] = Field(
        None,
        description="Profondeur du réseau (ex: 2 à 6)"
    )

    # --- 2) Taille cachée / latente ---
    hidden_size: Optional[conint(gt=0)] = Field(
        None,
        description="Dimension des représentations internes (ex: 128)"
    )


    bidirectional: Optional[bool] = Field(
        None,
        description="Choix de la bidirection : True, ou unidirection : Flase"
    )

    batch_first: Optional[bool] = Field(
        None,
        description="Batch first ?"
    )





class PaquetComplet(BaseModel):
    Parametres_temporels: Optional[Parametres_temporels]
    Parametres_choix_reseau_neurones: Optional[Parametres_choix_reseau_neurones]
    Parametres_choix_loss_fct: Optional[Parametres_choix_loss_fct]
    Parametres_optimisateur: Optional[Parametres_optimisateur]
    Parametres_entrainement: Optional[Parametres_entrainement]
    Parametres_visualisation_suivi: Optional[Parametres_visualisation_suivi]
    # Parametres_archi_reseau_MLP: Optional[Parametres_archi_reseau_MLP]
    # Parametres_archi_reseau_CNN: Optional[Parametres_archi_reseau_CNN]
    # Parametres_archi_reseau_LSTM: Optional[Parametres_archi_reseau_LSTM]
    
    


