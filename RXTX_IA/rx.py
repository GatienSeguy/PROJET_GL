import requests
import json

# ====================================
# CONFIGURATION DU CLIENT
# ====================================

URL_SERVEUR = "http://localhost:8000/train"


# ====================================
# FONCTION D'ENVOI
# ====================================

def envoyer_configuration(config_dict):
    """
    Envoie une configuration au serveur IA
    et affiche la réponse
    """
    print("\n" + "="*70)
    print(" ENVOI DE LA CONFIGURATION AU SERVEUR IA")
    print("="*70)
    print("\n Configuration envoyée :")
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    
    try:
        # Envoi de la requête POST
        response = requests.post(URL_SERVEUR, json=config_dict)
        
        print(f"\n Code de réponse : {response.status_code}")
        
        if response.status_code == 200:
            print(" SUCCÈS - Configuration acceptée !\n")
            print(" Réponse du serveur :")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(" ERREUR - Configuration rejetée !\n")
            print(" Détails de l'erreur :")
            erreurs = response.json()
            if 'detail' in erreurs:
                for erreur in erreurs['detail']:
                    print(f"   • Champ : {' -> '.join(map(str, erreur['loc']))}")
                    print(f"     Problème : {erreur['msg']}")
                    print(f"     Type : {erreur['type']}\n")
    
    except requests.exceptions.ConnectionError:
        print(" ERREUR : Impossible de se connecter au serveur")
        print("   Assure-toi que le serveur tourne sur http://localhost:8000")
    except Exception as e:
        print(f" ERREUR inattendue : {e}")
    
    print("="*70 + "\n")


# ====================================
# EXEMPLES DE CONFIGURATIONS
# ====================================

# Configuration 1 : COMPLÈTE et VALIDE 
config_complete = {
    "action": "start",
    "architecture": {
        "model_type": "lstm",
        "batch_size": 32,
        "num_layers": 3
    },
    "optimisation": {
        "optimizer": "adam",
        "loss_function": "mse",
        "learning_rate": 0.001,
        "epochs": 100,
        "weight_decay": 0.0001,
        "momentum": 0.9
    },
    "horizon": 24,
    "data": [1.5, 2.3, 3.1, 4.8, 5.2, 6.7, 7.1, 8.9, 9.3, 10.5, 11.2, 12.8, 
             13.4, 14.7, 15.2, 16.8, 17.3, 18.9, 19.1, 20.5],
    "send_weights_every": 5,
    "test_size": 0.1
}

# Configuration 2 : MINIMALE (avec valeurs par défaut) 
config_minimale = {
    "action": "start",
    "architecture": {
        "model_type": "gru",
        "batch_size": 16,
        "num_layers": 2
    },
    "optimisation": {
        "learning_rate": 0.01,
        "epochs": 50
        # optimizer, loss_function, weight_decay, momentum utiliseront les valeurs par défaut
    },
    "horizon": 48,
    "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

# Configuration 3 : STOP 
config_stop = {
    "action": "stop",
    "architecture": {
        "model_type": "lstm",
        "batch_size": 32,
        "num_layers": 3
    },
    "optimisation": {
        "learning_rate": 0.001,
        "epochs": 100
    },
    "horizon": 24,
    "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

# Configuration 4 : INVALIDE - Batch size trop grand 
config_invalide_batch = {
    "action": "start",
    "architecture": {
        "model_type": "lstm",
        "batch_size": 1000,  #  > 512
        "num_layers": 3
    },
    "optimisation": {
        "learning_rate": 0.001,
        "epochs": 100
    },
    "horizon": 24,
    "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

# Configuration 5 : INVALIDE - Learning rate trop grand 
config_invalide_lr = {
    "action": "start",
    "architecture": {
        "model_type": "transformer",
        "batch_size": 64,
        "num_layers": 4
    },
    "optimisation": {
        "learning_rate": 1.5,  # X >= 1
        "epochs": 100
    },
    "horizon": 24,
    "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}


# ====================================
# MENU INTERACTIF
# ====================================

def menu():
    """Menu pour choisir quelle configuration envoyer"""
    print("\n" + "="*70)
    print(" CLIENT D'ENVOI VERS SERVEUR IA")
    print("="*70)
    print("\nChoisissez une configuration à envoyer :")
    print("  1. Configuration complète (valide)")
    print("  2. Configuration minimale (valide)")
    print("  3. Commande STOP")
    print("  4. Configuration invalide - Batch size trop grand")
    print("  5. Configuration invalide - Learning rate trop grand")
    print("  6. Envoyer TOUTES les configurations (test complet)")
    print("  0. Quitter")
    
    choix = input("\n Votre choix : ")
    
    if choix == "1":
        envoyer_configuration(config_complete)
    elif choix == "2":
        envoyer_configuration(config_minimale)
    elif choix == "3":
        envoyer_configuration(config_stop)
    elif choix == "4":
        envoyer_configuration(config_invalide_batch)
    elif choix == "5":
        envoyer_configuration(config_invalide_lr)
    elif choix == "6":
        print("\n Envoi de TOUTES les configurations...")
        envoyer_configuration(config_complete)
        input("\nAppuyez sur Entrée pour continuer...")
        envoyer_configuration(config_minimale)
        input("\nAppuyez sur Entrée pour continuer...")
        envoyer_configuration(config_stop)
        input("\nAppuyez sur Entrée pour continuer...")
        envoyer_configuration(config_invalide_batch)
        input("\nAppuyez sur Entrée pour continuer...")
        envoyer_configuration(config_invalide_lr)
    elif choix == "0":
        print("\n Au revoir !")
        return False
    else:
        print("\n Choix invalide")
    
    return True


# ====================================
# POINT D'ENTRÉE
# ====================================

if __name__ == "__main__":
    continuer = True
    while continuer:
        continuer = menu()