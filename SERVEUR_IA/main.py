from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

app = FastAPI()


# ====================================
# MODÈLES PYDANTIC - CONFIGURATION
# ====================================

class HyperparametersArchitecture(BaseModel):
    """Configuration de l'architecture du modèle"""
    model_type: Literal["lstm", "gru", "transformer", "rnn"] = Field(
        description="Type de modèle de réseau de neurones"
    )
    batch_size: int = Field(
        ge=1, le=512,
        description="Taille du batch pour l'entraînement"
    )
    num_layers: int = Field(
        ge=1, le=10,
        description="Nombre de couches dans le réseau"
    )


class ConfigurationOptimisation(BaseModel):
    """Configuration de l'optimisation"""
    optimizer: Literal["adam", "sgd", "rmsprop", "adamw"] = Field(
        default="adam",
        description="Type d'optimiseur"
    )
    loss_function: Literal["mse", "mae", "huber", "cross_entropy"] = Field(
        default="mse",
        description="Fonction de perte"
    )
    learning_rate: float = Field(
        gt=0, lt=1,
        description="Taux d'apprentissage (0 < lr < 1)"
    )
    epochs: int = Field(
        ge=1, le=1000,
        description="Nombre d'époques d'entraînement"
    )
    # Paramètres optionnels additionnels
    weight_decay: Optional[float] = Field(
        default=0.0, ge=0,
        description="Régularisation L2 (weight decay)"
    )
    momentum: Optional[float] = Field(
        default=0.9, ge=0, lt=1,
        description="Momentum pour SGD"
    )


class ConfigEntrainement(BaseModel):
    """Configuration complète pour l'entraînement"""
    action: Literal["start", "stop"] = Field(
        description="Commande : start ou stop"
    )
    architecture: HyperparametersArchitecture = Field(
        description="Configuration de l'architecture du modèle"
    )
    optimisation: ConfigurationOptimisation = Field(
        description="Configuration de l'optimisation"
    )
    horizon: int = Field(
        gt=0, le=365,
        description="Horizon de prédiction (1-365)"
    )
    data: List[float] = Field(
        min_length=10,
        description="Données d'entraînement (min 10 points)"
    )
    
    # Paramètres optionnels
    send_weights_every: Optional[int] = Field(
        default=10, ge=1,
        description="Envoyer poids toutes les K itérations"
    )
    test_size: Optional[float] = Field(
        default=0.1, gt=0, lt=1,
        description="Proportion pour le test (défaut 10%)"
    )


# ====================================
# ROUTES
# ====================================

@app.get("/")
def accueil():
    return {
        "message": "Serveur IA actif !",
        "version": "0",
        "endpoints": ["/", "/train", "/docs"]
    }


@app.post("/train")
def recevoir_config(config: ConfigEntrainement):
    """
    Reçoit la configuration complète pour l'entraînement
    """
    print("\n" + "="*70)
    print(" CONFIGURATION REÇUE")
    print("="*70)
    print(f"🎬 Action : {config.action}")
    print(f"\n ARCHITECTURE :")
    print(f"   - Type de modèle : {config.architecture.model_type}")
    print(f"   - Batch size : {config.architecture.batch_size}")
    print(f"   - Nombre de couches : {config.architecture.num_layers}")
    print(f"\n OPTIMISATION :")
    print(f"   - Optimiseur : {config.optimisation.optimizer}")
    print(f"   - Fonction de perte : {config.optimisation.loss_function}")
    print(f"   - Learning rate : {config.optimisation.learning_rate}")
    print(f"   - Epochs : {config.optimisation.epochs}")
    print(f"   - Weight decay : {config.optimisation.weight_decay}")
    print(f"   - Momentum : {config.optimisation.momentum}")
    print(f"\n DONNÉES :")
    print(f"   - Horizon : {config.horizon}")
    print(f"   - Nombre de points : {len(config.data)}")
    print(f"   - Premier point : {config.data[0]}")
    print(f"   - Dernier point : {config.data[-1]}")
    print(f"\n PARAMÈTRES :")
    print(f"   - Envoyer poids toutes les {config.send_weights_every} itérations")
    print(f"   - Taille du test : {config.test_size * 100}%")
    print("="*70 + "\n")
    
    if config.action == "start":
        return {
            "status": "Configuration reçue - Prêt à démarrer l'entraînement",
            "config_resume": {
                "architecture": {
                    "model_type": config.architecture.model_type,
                    "batch_size": config.architecture.batch_size,
                    "num_layers": config.architecture.num_layers
                },
                "optimisation": {
                    "optimizer": config.optimisation.optimizer,
                    "loss": config.optimisation.loss_function,
                    "learning_rate": config.optimisation.learning_rate,
                    "epochs": config.optimisation.epochs
                },
                "data": {
                    "horizon": config.horizon,
                    "nb_points": len(config.data)
                }
            }
        }
    else:
        return {
            "status": "Commande STOP reçue",
            "message": "L'entraînement sera arrêté"
        }