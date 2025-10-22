import pandas as pd
import json
from pathlib import Path

# 🟢 Chemin du fichier source
src_path = Path("/Users/gatienseguy/Documents/VSCode/PROJET_GL/RXTX_IA_et_Functiontest/CACAO_6mois.csv")
out_path = src_path.with_suffix(".json")

# 🟢 Détection du séparateur
with open(src_path, "r", encoding="utf-8") as f:
    sample = f.read(2048)
sep = ";" if sample.count(";") > sample.count(",") else ","

# 🟢 Lecture du CSV (2 colonnes : timestamp, valeur)
df = pd.read_csv(src_path, sep=sep, header=None, engine="python")

if df.shape[1] < 2:
    raise ValueError("Le fichier doit avoir au moins deux colonnes (timestamp, valeur).")

df = df.iloc[:, :2]
df.columns = ["timestamp_raw", "value_raw"]

# 🕒 Conversion des timestamps (jour/mois/année si besoin)
ts = pd.to_datetime(df["timestamp_raw"], errors="coerce", dayfirst=True)

# 🔢 Conversion des valeurs
vals = pd.to_numeric(df["value_raw"], errors="coerce")/1000

# 🧹 Nettoyage
clean = pd.DataFrame({"timestamp": ts, "value": vals}).dropna(subset=["timestamp"])
clean = clean.sort_values("timestamp").reset_index(drop=True)

# 📅 Format final : "YYYY-MM-DD HH:MM:SS"
timestamps_fmt = clean["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
values_list = clean["value"].where(pd.notna(clean["value"]), None).tolist()

# 🧩 Structure conforme à TimeSeriesData
payload = {
    "timestamps": timestamps_fmt,
    "values": values_list
}

# 💾 Sauvegarde JSON
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"✅ Conversion réussie ({len(clean)} lignes) → {out_path}")