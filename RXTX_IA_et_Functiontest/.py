import csv
import json
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field

class TimeSeriesData(BaseModel):
    """
    Une unique série temporelle : timestamps et valeurs alignés (même longueur).
    """
    timestamps: List[datetime] = Field(..., description="Liste UTC triée croissante (ISO 8601)")
    values: List[Optional[float]] = Field(..., description="Valeurs numériques (Null si manquante), même longueur que timestamps")
     
    def model_post_init(self, __context) -> None:
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps et values doivent avoir la même longueur")

def csv_to_timeseries_json(csv_path: Path, output_path: Path, start_date: str = "2024-01-01"):
    """
    Convertit un CSV en JSON au format TimeSeriesData.
    
    Args:
        csv_path: Chemin vers le fichier CSV
        output_path: Chemin vers le fichier JSON de sortie
        start_date: Date de début (format YYYY-MM-DD)
    """
    timestamps = []
    values = []
    
    # Générer les timestamps (début chaque lundi, semaine par semaine)
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Lire le CSV et extraire les valeurs de la 2ème colonne
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')  # Utiliser point-virgule comme séparateur
        
        # Ignorer l'en-tête si présent
        header = next(reader, None)
        print(f"En-tête détecté : {header}")
        
        for row in reader:
            if len(row) >= 2:  # S'assurer qu'il y a au moins 2 colonnes
                # Ajouter le timestamp
                timestamps.append(current_date)
                
                # Traiter la valeur de la 2ème colonne
                try:
                    # Essayer de convertir en float (supprimer les espaces pour format français)
                    value_str = row[1].strip().replace(' ', '')  # Enlever les espaces (32 793 -> 32793)
                    value = float(value_str)/100000 if value_str else None
                except (ValueError, AttributeError, IndexError):
                    value = None
                
                values.append(value)
                
                # Passer à la semaine suivante
                current_date += timedelta(weeks=1)
    
    # Créer l'objet TimeSeriesData pour validation
    if not timestamps:
        print("⚠️  Aucune donnée trouvée dans le fichier CSV")
        return
    
    ts_data = TimeSeriesData(timestamps=timestamps, values=values)
    
    # Convertir en dict et formater les timestamps en ISO 8601
    output_dict = {
        "timestamps": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in ts_data.timestamps],
        "values": ts_data.values
    }
    
    # Écrire le JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Conversion réussie : {len(timestamps)} entrées")
    print(f"✓ Fichier créé : {output_path}")
    print(f"✓ Période : {timestamps[0].strftime('%Y-%m-%d')} → {timestamps[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Valeurs non-null : {sum(1 for v in values if v is not None)}/{len(values)}")

# Utilisation
if __name__ == "__main__":
    # Chemins des fichiers
    csv_path = Path("/Users/gatienseguy/Documents/VSCode/PROJET_GL/RXTX_IA_et_Functiontest/Boites.csv")
    output_path = Path("/Users/gatienseguy/Documents/VSCode/PROJET_GL/RXTX_IA_et_Functiontest/Boites_output.json")
    
    # Vérifier que le fichier CSV existe
    if not csv_path.exists():
        print(f"❌ Erreur : Le fichier {csv_path} n'existe pas")
    else:
        print(f"📂 Lecture du fichier : {csv_path}")
        csv_to_timeseries_json(
            csv_path=csv_path,
            output_path=output_path,
            start_date="2024-01-01"
        )