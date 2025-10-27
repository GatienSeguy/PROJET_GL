import requests
import json

# ====================================
# CONFIGURATION
# ====================================
SERVER_URL = "http://192.168.27.66:8000"

def get_data(name: str = None, dates: list = None):
    """
    Récupère des données depuis le serveur
    
    Args:
        name: Nom du fichier JSON (ex: 'EURO', 'BITCOIN', 'CACAO')
        dates: Liste [date_debut, date_fin] au format 'AAAA-MM-JJ'
    """
    # ⚠️ CORRECTION : Utiliser le bon endpoint
    url = f"{SERVER_URL}/data/get/"  # ← C'était ça le problème !
    
    payload = {}
    if name:
        payload["name"] = name
    if dates:
        payload["dates"] = dates
    
    print(f"🔍 Requête vers : {url}")
    print(f"📦 Payload : {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        print(f"✅ Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ Données récupérées avec succès !")
            print(f"📊 Nombre de points : {len(data.get('timestamps', []))}")
            
            timestamps = data.get('timestamps', [])
            values = data.get('values', [])
            
            if timestamps and values:
                print(f"\n📅 Premier point : {timestamps[0]} → {values[0]}")
                print(f"📅 Dernier point : {timestamps[-1]} → {values[-1]}")
                
                # Statistiques
                print(f"\n📈 Statistiques:")
                print(f"   Min: {min(values)}")
                print(f"   Max: {max(values)}")
                print(f"   Moyenne: {sum(values)/len(values):.4f}")
            
            return data
        else:
            print(f"\n❌ Erreur {response.status_code}")
            try:
                error_data = response.json()
                print(f"💬 Détails : {json.dumps(error_data, indent=2)}")
            except:
                print(f"📄 Réponse : {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Impossible de se connecter au serveur {SERVER_URL}")
        print(f"💡 Vérifications :")
        print(f"   1. Serveur lancé ? → uvicorn main:app --reload")
        print(f"   2. Bonne IP ? → Vérifiez avec ifconfig/ipconfig")
        return None
    
    except requests.exceptions.Timeout:
        print(f"\n⏱️ Timeout : le serveur met trop de temps à répondre")
        return None
    
    except Exception as e:
        print(f"\n❌ Erreur : {type(e).__name__}: {e}")
        return None


def test_connection():
    """Test de connexion au serveur"""
    print("🔌 Test de connexion...")
    try:
        response = requests.get(f"{SERVER_URL}/", timeout=5)
        print(f"✅ Serveur accessible ! Status: {response.status_code}")
        
        # Afficher les endpoints disponibles
        try:
            data = response.json()
            if "endpoints" in data:
                print("\n📋 Endpoints disponibles:")
                for name, path in data["endpoints"].items():
                    print(f"   • {name}: {path}")
        except:
            pass
        
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Serveur inaccessible")
        return False
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False


def list_available_files():
    """Liste les fichiers disponibles sur le serveur"""
    print("\n📂 Tentative de liste des fichiers...")
    try:
        # Votre serveur n'a pas d'endpoint pour lister, mais on peut suggérer
        print("💡 Fichiers probables selon votre arborescence:")
        print("   - BITCOIN")
        print("   - EURO")
        print("   - CACAO")
        print("   - Boites_output")
        print("   - Boites_per_day")
    except Exception as e:
        print(f"❌ Erreur : {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TESTS DE RÉCUPÉRATION DE DONNÉES")
    print("=" * 60)
    
    # Test 0 : Connexion
    print("\n" + "=" * 60)
    print("TEST 0 : Connexion au serveur")
    print("=" * 60)
    if not test_connection():
        print("\n⚠️  Serveur non accessible. Vérifiez qu'il est lancé.")
        list_available_files()
        exit(1)
    
    # Test 1 : Récupération EURO complète
    print("\n" + "=" * 60)
    print("TEST 1 : Récupération complète EURO")
    print("=" * 60)
    data1 = get_data(name="EURO")
    
    # Test 2 : Récupération filtrée
    print("\n" + "=" * 60)
    print("TEST 2 : Récupération EURO filtrée par dates")
    print("=" * 60)
    data2 = get_data(
        name="EURO",
        dates=["2025-01-01", "2025-01-05"]
    )
    
    # Test 3 : BITCOIN
    print("\n" + "=" * 60)
    print("TEST 3 : Récupération BITCOIN")
    print("=" * 60)
    data3 = get_data(name="BITCOIN")
    
    # Test 4 : CACAO
    print("\n" + "=" * 60)
    print("TEST 4 : Récupération CACAO")
    print("=" * 60)
    data4 = get_data(name="CACAO")
    
    print("\n" + "=" * 60)
    print("✅ TESTS TERMINÉS")
    print("=" * 60)