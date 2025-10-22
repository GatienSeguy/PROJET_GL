# Convert /mnt/data/CACAO_6mois.csv to TimeSeriesData-style JSON
import pandas as pd
import json
from pathlib import Path
from datetime import timezone

src_path = Path("/mnt/data/CACAO_6mois.csv")
out_path = Path("/mnt/data/CACAO_6mois.json")

# Try both separators: ',' and ';'
dfs = []
for sep in [",", ";"]:
    try:
        df_try = pd.read_csv(src_path, sep=sep, header=None, engine="python")
        df_try["__sep__"] = sep
        dfs.append(df_try)
    except Exception:
        pass

if not dfs:
    raise RuntimeError("Impossible de lire le CSV avec ',' ni ';'.")

# Choose the attempt with at least 2 columns and the fewest NaNs in col1
def score_df(d):
    ok_cols = d.shape[1] >= 2
    if not ok_cols:
        return (-1, None)
    # Count non-nan in second column
    non_nan = pd.to_numeric(d.iloc[:,1], errors="coerce").notna().sum()
    return (non_nan, d["__sep__"].iloc[0])

best = max(dfs, key=score_df)
sep_used = best["__sep__"].iloc[0]
best = best.drop(columns="__sep__", errors="ignore")

# Keep only first two columns and rename
df = best.iloc[:, :2].copy()
df.columns = ["timestamp_raw", "value_raw"]

# Parse timestamps -> UTC ISO 8601
ts = pd.to_datetime(df["timestamp_raw"], errors="coerce", utc=False)

# If tz-naive, localize as UTC; if tz-aware, convert to UTC
if getattr(ts.dt, "tz", None) is None:
    ts = ts.dt.tz_localize("UTC")
else:
    ts = ts.dt.tz_convert("UTC")

# Parse numeric values (allow NaN = null)
vals = pd.to_numeric(df["value_raw"], errors="coerce")

# Build clean DataFrame
clean = pd.DataFrame({"timestamp": ts, "value": vals})
clean = clean.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# Create JSON structure
timestamps_iso = clean["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
values_list = clean["value"].where(pd.notna(clean["value"]), None).tolist()

payload = {
    "timestamps": timestamps_iso,
    "values": values_list
}

# Save to JSON
with out_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

# Show a small preview to the user
preview = clean.head(10).copy()
preview["timestamp"] = preview["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
import caas_jupyter_tools as cj
cj.display_dataframe_to_user("Aperçu des 10 premières lignes (UTC)", preview)

sep_used, len(clean), out_path.as_posix()