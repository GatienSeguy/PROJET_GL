import os
import json
from datetime import datetime

FOLDER = "C:/Users/maxim/Downloads/maregraphie/"  # dossier contenant les 95_*.txt
OUTFILE = "merged_source3.json"

# Si True : override l'année trouvée dans la ligne par l'année extraite du nom de fichier (ex: 95_2025.txt -> 2025)
# Si False : on conserve l'année telle qu'elle est dans la ligne.
OVERRIDE_YEAR_WITH_FILENAME = False

output = {"timestamps": [], "values": []}

def try_parse_datetime(s):
    """Essaie plusieurs formats courants et renvoie un datetime ou None."""
    fmts = ["%d/%m/%Y %H:%M:%S", "%d/%m/%y %H:%M:%S", "%Y-%m-%d %H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None

# Parcours des fichiers triés par nom
for filename in sorted(os.listdir(FOLDER)):
    if not (filename.startswith("95_") and filename.endswith(".txt")):
        continue
    filepath = os.path.join(FOLDER, filename)
    # extraction d'une année possible depuis le nom (ex: 95_2025.txt -> 2025)
    year_from_filename = None
    try:
        base = os.path.splitext(filename)[0]  # "95_2025"
        year_part = base.split("_")[1]
        year_from_filename = int(year_part)
    except Exception:
        year_from_filename = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or ";" not in line:
                continue
            parts = line.strip().split(";")
            if len(parts) != 3:
                continue
            date_str, value_str, source_str = parts
            if source_str.strip() != "3":
                continue

            dt = try_parse_datetime(date_str.strip())
            if dt is None:
                # si on n'a pas pu parser directement, essayons d'extraire jour/mois/heure et forcer l'année du fichier
                # ex: "01/01 00:00:00" ou formats inattendus
                segs = date_str.strip().split()
                if len(segs) >= 2 and "/" in segs[0] and year_from_filename is not None:
                    # tenter dd/mm (sans année) + hour
                    day_month = segs[0]
                    timepart = segs[1]
                    try:
                        dt = datetime.strptime(f"{day_month}/{year_from_filename} {timepart}", "%d/%m/%Y %H:%M:%S")
                    except Exception:
                        dt = None

            if dt is None:
                # si toujours None, on saute la ligne (ou tu peux logger)
                # print(f"Warning: date non parsable dans {filename}: '{date_str}'")
                continue

            if OVERRIDE_YEAR_WITH_FILENAME and year_from_filename is not None:
                # remplace l'année par celle du nom de fichier (utile si les dates internes sont erronées)
                dt = dt.replace(year=year_from_filename)

            iso = dt.strftime("%Y-%m-%d %H:%M:%S")
            try:
                val = float(value_str.replace(",", "."))
            except Exception:
                continue

            output["timestamps"].append(iso)
            output["values"].append(val)

# Optionnel : trier par timestamp croissant (utile si on veut un temps monotone)
# On reconstruit la liste triée en paires
pairs = list(zip(output["timestamps"], output["values"]))
pairs.sort(key=lambda p: p[0])  # tri lexicographique sur ISO -> correct pour les dates
output["timestamps"], output["values"] = zip(*pairs) if pairs else ([], [])

# Convert tuples to lists (json serializable)
output["timestamps"] = list(output["timestamps"])
output["values"] = list(output["values"])

with open(OUTFILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✔ Fichier JSON généré : {OUTFILE} (lignes source==3, timestamps en YYYY-MM-DD HH:MM:SS)")
