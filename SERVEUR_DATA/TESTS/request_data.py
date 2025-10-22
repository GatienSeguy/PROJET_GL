import requests
import json

# ====================================
# CONFIGURATION
# ====================================
SERVER_URL = "http://:8000"

def get_data(name: str = None, dates: list = None):
    """
    Récupère des données depuis le serveur
    
    Args:
        name: Nom du fichier JSON (ex: 'series_temp')
        dates: Liste [date_debut, date_fin] au format 'AAAA-MM-JJ'
    """
    url = f"{SERVER_URL}/data/get/"
    
    payload = {}
    if name:
        payload["name"] = name
    if dates:
        payload["dates"] = dates
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            print("Données récupérées avec succès !")
            print(f"Nombre de points : {data['count']}")
            print(f"Filtré : {'Oui' if data['filtered'] else 'Non'}")
            
            if data['count'] > 0:
                print(f"\nPremier point : {data['timestamps'][0]} → {data['values'][0]}")
                print(f" Dernier point : {data['timestamps'][-1]} → {data['values'][-1]}")
            
            return data
        else:
            print(f"Erreur {response.status_code}: {response.json()}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"Impossible de se connecter au serveur {SERVER_URL}")
        print(" Vérifiez que le serveur est lancé avec : uvicorn main:app --reload")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print("TEST 1 : Récupération complète")
    print("=" * 50)
    get_data(name="timeseries_data")
    
    print("\n" + "=" * 50)
    print("TEST 2 : Récupération filtrée par dates")
    print("=" * 50)
    get_data(
        name="timeseries_data",
        dates=["2025-01-01", "2025-01-05"]
    )