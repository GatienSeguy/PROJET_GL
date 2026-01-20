import json### à  tester avec main_test.py  (serveur DATA) en local 
import requests
import base64
from pathlib import Path 
from datetime import datetime

# URL du serveur
URL = "http://127.0.0.0:8001"

BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR.parent
DATASET_DIR = CODE_DIR / "SERVEUR_DATA" / "datasets"
MODELS_DIR = CODE_DIR / "SERVEUR_DATA" / "models"
CONTEXTES_DIR = CODE_DIR / "SERVEUR_DATA" / "contextes"


print(f"DATASET_DIR = {DATASET_DIR}")
def send_dataset(file_path: Path, dataset_name: str, raise_on_error: bool = True) -> bool:
    """
    Envoie un dataset au serveur DATA.
    
    Args:
        file_path: Path vers le fichier JSON contenant le dataset.
        dataset_name: Nom sous lequel le dataset sera enregistré.
        raise_on_error: Si True, lève une exception en cas d'erreur.
                        Si False, retourne False pour un doublon ou erreur serveur.

    Returns:
        True si l'upload a réussi, False si doublon ou échec (et raise_on_error=False)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    # Lire le contenu du fichier JSON
    with file_path.open("r", encoding="utf-8") as f:
        time_series_data = json.load(f)

    # Construire le payload attendu par /datasets/data_add
    payload = {
        "name": dataset_name,
        "data": time_series_data
    }

    # Envoyer le JSON
    response = requests.post(f"{URL}/datasets/data_add", json=payload)
    print("Status code:", response.status_code)
    try:
        resp_json = response.json()
        print("Response:", resp_json)
    except ValueError:
        print("Response text:", response.text)
        resp_json = {}

    # Vérification du statut
    if response.status_code == 200:
        # ✅ Vérification locale
        dataset_nom = dataset_name + ".json"
        path_new_dataset = DATASET_DIR / dataset_nom
        if not path_new_dataset.exists():
            raise FileNotFoundError(
                f"Le dataset '{dataset_nom}' n'existe pas dans {DATASET_DIR}"
            )
        print(f"✅ Dataset '{dataset_name}' envoyé et enregistré avec succès.")
        return True
    else:
        if raise_on_error:
            raise RuntimeError(f"❌ Échec de l'upload du dataset '{dataset_name}': {resp_json.get('detail')}")
        else:
            print(f"✅ Doublon détecté pour le dataset '{dataset_name}' (comme prévu)")
            return False
def send_dataset_double(file_path: Path, dataset_name: str):
    print(f"\n--- Envoi du dataset '{dataset_name}' (premier envoi) ---")
    send_dataset(file_path, dataset_name)

    print(f"\n--- Envoi du dataset '{dataset_name}' (deuxième envoi, doit échouer) ---")
    success = send_dataset(file_path, dataset_name, raise_on_error=False)
    if not success:
        print(f"✅ Le serveur a correctement refusé le doublon du dataset '{dataset_name}'.")

    print("\nTest terminé, le code continue normalement.")

def test_info_all_bad_message():
    """
    Envoie un message invalide à /datasets/info_all
    ➜ doit renvoyer 400
    ➜ le test ne doit PAS crasher
    """
    payload = {
        "message": "nawak_total"
    }

    response = requests.post(f"{URL}/datasets/info_all", json=payload)

    print("Status code:", response.status_code)
    print("Response:", response.text)

    if response.status_code != 400:
        raise RuntimeError(
            f"❌ ERREUR : attendu 400, reçu {response.status_code}"
        )

    print("✅ Mauvais message correctement rejeté (400)")
    print("➡️ Le code continue normalement\n")


def dataset_all():
    """
    Vérifie que les datasets renvoyés par l'API correspondent exactement
    aux fichiers présents dans DATASET_DIR.
    """
    url = f"{URL}/datasets/info_all"

    payload = {
        "message": "choix dataset"
    }

    response = requests.post(url, json=payload)

    print("Status code:", response.status_code)

    try:
        api_datasets = response.json()
        print("Response:", api_datasets)
    except ValueError:
        print("Response text:", response.text)
        raise RuntimeError("Réponse non JSON")

    if response.status_code != 200:
        raise RuntimeError("Échec récupération des datasets")

    # 📁 Datasets présents sur le disque
    if not DATASET_DIR.exists():
        raise RuntimeError(f"Le dossier {DATASET_DIR} n’existe pas")

    disk_datasets = sorted(
        f.stem for f in DATASET_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".json"
    )

    # 🌐 Datasets renvoyés par l'API
    api_dataset_names = sorted(api_datasets.keys())

    print("Datasets disque :", disk_datasets)
    print("Datasets API    :", api_dataset_names)

    # 🔍 Comparaison stricte
    if disk_datasets != api_dataset_names:
        raise RuntimeError(
            "❌ Incohérence entre DATASET_DIR et l'API\n"
            f"Disque : {disk_datasets}\n"
            f"API    : {api_dataset_names}"
        )

    print("✅ Les datasets API correspondent exactement aux fichiers du dossier.")



def test_data_solo(dataset_name: str, date_debut: str, date_fin: str, pas: int):
    payload = {
        "name": dataset_name,
        "dates": [date_debut, date_fin],
        "pas_temporel": pas
    }

    response = requests.post(f"{URL}/datasets/data_solo", json=payload)
    print("Status code:", response.status_code)

    if response.status_code != 200:
        print("❌ Erreur serveur :", response.text)
        return

    result = response.json()

    # structure exacte retournée
    if dataset_name not in result:
        print("❌ Clé dataset absente dans la réponse")
        return

    data = result[dataset_name]["data"]
    timestamps = [
        datetime.fromisoformat(ts)
        for ts in data["timestamps"]
    ]

    d0 = datetime.fromisoformat(date_debut)
    d1 = datetime.fromisoformat(date_fin)

    # 1️⃣ date_debut : borne basse
    if timestamps[0] >= d0:
        print("✅ date_debut respectée (borne basse)")
    else:
        print("❌ date_debut non respectée")

    # 2️⃣ date_fin : borne haute
    if timestamps[-1] <= d1:
        print("✅ date_fin respectée (borne haute)")
    else:
        print("❌ date_fin non respectée")

    # 3️⃣ pas logique (indice)
    if pas < 1:
        print("❌ pas invalide")
        return

    # On vérifie juste que l'espacement est constant
    deltas = [
        (timestamps[i+1] - timestamps[i]).days
        for i in range(len(timestamps) - 1)
    ]

    if len(set(deltas)) >= 1:
        print(f"✅ pas appliqué correctement (indice = {pas})")

    print("✅ TEST GLOBAL OK pour", dataset_name)

def test_erreur_data_solo():
    """
    Teste la robustesse de la route /datasets/data_solo et de construire_un_dataset
    en forçant différents cas d'erreurs. Génère une erreur si le serveur ne renvoie pas
    le code d'erreur attendu.
    """

    tests = [
        {
            "desc": "dataset inexistant",
            "payload": {
                "name": "dataset_inexistant",
                "dates": ["2025-01-01", "2025-01-10"],
                "pas_temporel": 1
            }
        },
        {
            "desc": "date_debut après date_fin",
            "payload": {
                "name": "sofiane_est_un_dieu",
                "dates": ["2025-12-01", "2025-01-01"],
                "pas_temporel": 1
            }
        },
        {
            "desc": "pas ≤ 0",
            "payload": {
                "name": "sofiane_est_un_dieu",
                "dates": ["2025-01-01", "2025-12-01"],
                "pas_temporel": 0
            }
        },
        {
            "desc": "intervalle sans données",
            "payload": {
                "name": "sofiane_est_un_dieu",
                "dates": ["1900-01-01", "1900-01-10"],
                "pas_temporel": 1
            }
        },
        {
            "desc": "dataset incomplet (timestamps ou values manquants)",
            "payload": {
                "name": "dataset_incomplet",
                "dates": ["2025-01-01", "2025-12-01"],
                "pas_temporel": 1
            }
        }
    ]

    for test in tests:
        print(f"\n--- Test : {test['desc']} ---")
        payload = test["payload"]
        response = requests.post(f"{URL}/datasets/data_solo", json=payload)
        status = response.status_code

        try:
            resp_json = response.json()
        except ValueError:
            resp_json = {"raw_text": response.text}

        print("Status code:", status)
        print("Response:", resp_json)

        # On s'attend à un code d'erreur HTTP (400 ou 404)
        if status == 200:
            raise RuntimeError(f"❌ Échec du test '{test['desc']}': le serveur n'a pas généré d'erreur alors qu'il aurait dû.")
        else:
            print(f"✅ Erreur correctement détectée pour '{test['desc']}'")

    print("\n✅ Tous les tests d'erreur ont été exécutés avec succès, les cas invalides ont bien généré des erreurs.")


def delete_dataset(dataset_name: str, raise_on_error: bool = True) -> bool:
    """
    Supprime un dataset sur le serveur DATA.
    
    Args:
        dataset_name: Nom du dataset à supprimer
        raise_on_error: Si True, lève une exception en cas d'erreur
                        Si False, retourne False si le dataset n'existe pas ou erreur serveur

    Returns:
        True si suppression réussie, False si échec et raise_on_error=False
    """
    payload = {"name": dataset_name}
    response = requests.post(f"{URL}/datasets/data_supression", json=payload)
    print("Status code:", response.status_code)

    try:
        resp_json = response.json()
        print("Response:", resp_json)
    except ValueError:
        print("Response text:", response.text)
        resp_json = {}

    dataset_nom = dataset_name + ".json"
    path_dataset = DATASET_DIR / dataset_nom

    if response.status_code == 200:
        # Vérification locale
        if path_dataset.exists():
            raise RuntimeError(f"❌ Le dataset '{dataset_nom}' existe encore dans {DATASET_DIR}")
        print(f"✅ Dataset '{dataset_name}' supprimé avec succès.")
        return True
    else:
        if raise_on_error:
            raise RuntimeError(
                f"❌ Échec de la suppression du dataset '{dataset_name}': {resp_json.get('detail')}"
            )
        else:
            print(f"✅ Dataset '{dataset_name}' introuvable ou déjà supprimé (comme prévu)")
            return False


def delete_dataset_double(dataset_name: str):
    """
    Teste la suppression double d'un dataset :
    - Premier appel : doit réussir
    - Deuxième appel : doit échouer (dataset déjà supprimé) et afficher un message clair
    """
    print(f"\n--- Premier appel suppression du dataset '{dataset_name}' ---")
    delete_dataset(dataset_name)

    print(f"\n--- Deuxième appel suppression du dataset '{dataset_name}' (doublon attendu) ---")
    success = delete_dataset(dataset_name, raise_on_error=False)
    if not success:
        print(f"✅ Le serveur a correctement refusé la suppression du dataset déjà supprimé")

    # Vérification locale
    dataset_nom = dataset_name + ".json"
    path_dataset = DATASET_DIR / dataset_nom
    if not path_dataset.exists():
        print(f"✅ Dataset '{dataset_name}' bien absent localement après suppression")

    print("\nTest terminé, le code continue normalement.\n")

    print(f"✅ Dataset '{dataset_name}' supprimé avec succès.")
def send_model(file_path: Path, model_name: str, raise_on_error: bool = True) -> bool:
    """
    Envoie un modèle au serveur. 
    Si raise_on_error=False, ne lève pas d'exception en cas d'erreur, renvoie False.
    Retourne True si upload réussi.
    """
    url = f"{URL}/models/model_add"

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    with file_path.open("rb") as file:
        encoded_data = base64.b64encode(file.read()).decode("utf-8")

    payload = {"name": model_name, "data": encoded_data}

    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    try:
        resp_json = response.json()
        print("Response:", resp_json)
    except ValueError:
        print("Response text:", response.text)
        resp_json = {}

    if response.status_code == 200:
        # ⚠️ TEST LOCAL
        path_new_model = MODELS_DIR / f"{model_name}.pth"
        if not path_new_model.exists():
            raise FileNotFoundError(
                f"Le modèle '{model_name}.pth' n'existe pas dans {MODELS_DIR}"
            )
        print(f"✅ Modèle '{model_name}.pth' envoyé et enregistré avec succès.")
        return True
    else:
        if raise_on_error:
            raise RuntimeError(f"❌ Échec de l'upload du modèle '{model_name}': {resp_json.get('detail')}")
        else:
            # Ne pas lever, juste indiquer échec
            print(f"✅ Doublon détecté pour '{model_name}' (comme prévu)")
            return False


def send_model_double(file_path: Path, model_name: str):
    print(f"\n--- Envoi du modèle '{model_name}' (premier envoi) ---")
    send_model(file_path, model_name)

    print(f"\n--- Envoi du modèle '{model_name}' (deuxième envoi, doit échouer) ---")
    success = send_model(file_path, model_name, raise_on_error=False)
    if not success:
        print("✅ Le serveur a correctement refusé le doublon.")

    print("\nTest terminé, le code continue normalement.")




def model_all():
    """
    Vérifie que les modèles renvoyés par l'API correspondent exactement
    aux fichiers présents dans MODELS_DIR.
    """
    url = f"{URL}/models/model_all"

    payload = {
        "message": "choix_models"
    }

    response = requests.post(url, json=payload)

    print("Status code:", response.status_code)

    try:
        api_models = response.json()
        print("Response:", api_models)
    except ValueError:
        print("Response text:", response.text)
        raise RuntimeError("Réponse non JSON")

    if response.status_code != 200:
        raise RuntimeError("Échec récupération des modèles")

    # 📁 Modèles présents sur le disque
    if not MODELS_DIR.exists():
        raise RuntimeError(f"Le dossier {MODELS_DIR} n’existe pas")

    disk_models = sorted(
        f.stem for f in MODELS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".pth"
    )

    # 🌐 Modèles renvoyés par l'API
    api_model_names = sorted(api_models.keys())

    print("Modèles disque :", disk_models)
    print("Modèles API    :", api_model_names)

    # 🔍 Comparaison stricte
    if disk_models != api_model_names:
        raise RuntimeError(
            "❌ Incohérence entre MODELS_DIR et l'API\n"
            f"Disque : {disk_models}\n"
            f"API    : {api_model_names}"
        )

    print("✅ Les modèles API correspondent exactement aux fichiers du dossier.")

def delete_model(model_name: str):
    url = f"{URL}/models/model_delete"

    payload = {
        "name": model_name
    }

    response = requests.post(url, json=payload)

    print("Status code:", response.status_code)

    try:
        print("Response:", response.json())
    except ValueError:
        print("Response text:", response.text)

    if response.status_code != 200:
        raise RuntimeError("Échec de la suppression du modèle")

    # ⚠️ TEST LOCAL (assumé, même philosophie)
    path_model = MODELS_DIR / f"{model_name}.pth"

    if path_model.exists():
        raise RuntimeError(
            f"❌ Le modèle '{model_name}.pth' existe encore dans {MODELS_DIR}"
        )

    print(f"✅ Modèle '{model_name}.pth' supprimé avec succès.")


def send_contexte(paquet: dict):
    url = "http://127.0.0.1:8001/contexte/add_solo"

    response = requests.post(url, json=paquet)
    print("Status code:", response.status_code)

    try:
        print("Réponse JSON:", response.json())
    except ValueError:
        print("Réponse brute:", response.text)

    if response.status_code != 200:
        raise RuntimeError("❌ Échec de l'envoi du contexte")

    # Vérification que le contexte est bien enregistré
    context_name = paquet["payload_name_model"]["name"]
    context_file = CONTEXTES_DIR / f"contexte_{context_name}_cnn.json"
    print("Vérification du fichier contexte :", context_file)


    if not context_file.exists():
        raise FileNotFoundError(f"❌ Le contexte '{context_name}' n'a pas été trouvé dans {CONTEXTES_DIR}")

    print(f"✅ Contexte '{context_name}' bien enregistré dans {CONTEXTES_DIR}")
    return context_file

def get_contexte(context_name: str):
    """
    Récupère un contexte via /contexte/obtenir_solo et vérifie
    qu'il correspond au fichier JSON enregistré.
    """
    url = f"{URL}/contexte/obtenir_solo"
    payload = {"name": context_name}

    response = requests.post(url, json=payload)

    print("Status code:", response.status_code)
    try:
        json_response = response.json()
        print("Response:", json_response)
    except ValueError:
        print("Response text:", response.text)
        raise RuntimeError("Réponse non JSON")

    if response.status_code != 200:
        raise RuntimeError("❌ Échec de la récupération du contexte")

    # lecture locale
        context_file = CONTEXTES_DIR / f"contexte_{context_name}_cnn.json"

    if not context_file.exists():
        raise FileNotFoundError(
            f"Le fichier contexte local '{context_file}' n'existe pas"
        )

    with context_file.open("r", encoding="utf-8") as f:
        local_json = json.load(f)

    if local_json != json_response:
        raise RuntimeError("❌ Incohérence entre le contexte local et l'API")

    print(f"✅ Contexte '{context_name}' récupéré et cohérent avec le fichier local.")

    return json_response


# Exemple d'appel

#test pas 

send_dataset_double(BASE_DIR / "Dataset_test_global.json", "sofiane_est_un_dieu")
a = input("Appuyez sur Entrée pour continuer... (test de la route /datasets/info_all)")
dataset_all()
test_info_all_bad_message()
a = input("Appuyez sur Entrée pour continuer... (test du crop des données)")


for i in range(1,6):
   

    # Exemple d'appel   
    test_data_solo(
        dataset_name="sofiane_est_un_dieu",
        date_debut="2025-05-" +"0" +str(i),
        date_fin="2025-10-" +"0" + str(i),
        pas=i
    )

    print("\n\n")

a = input("Appuyez sur Entrée pour continuer... (test des erreurs lors du crop des données)")
test_erreur_data_solo()

a = input("Appuyez sur Entrée pour continuer... (test de la suppression double)")
delete_dataset_double("sofiane_est_un_dieu")

a = input("Appuyez sur Entrée pour continuer... (test de l'envoi double de modèle)")

# Exemple d'appel pour l'envoi de modèle

send_model_double(BASE_DIR / "test.pth", "gatien_goat")
delete_model("gatien_goat")


json_file = BASE_DIR / "maximebg.json"

with open(json_file, "r", encoding="utf-8") as f:
    PAQUET = json.load(f)


#send_contexte(PAQUET)
#get_contexte("model_maximebg")

#SERVEUR IA