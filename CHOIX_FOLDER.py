from tkinter import Tk, Button, Label
from tkinter import filedialog
import json
from pydantic import BaseModel, ValidationError
from typing import List, Optional

# =========================
# Modèle Pydantic
# =========================
class TimeSeriesData(BaseModel):
    timestamps: List[str]
    values: List[Optional[float]]   

# =========================
# Initialisation Tkinter
# =========================
root = Tk()
root.title("Sélection JSON")
root.geometry("500x250")

payload = None  # contiendra l'objet TimeSeriesData

# =========================
# Sélection d'un fichier JSON
# =========================
def choose_file():
    global payload

    file_path = filedialog.askopenfilename(
        title="Sélectionner un fichier JSON",
        filetypes=[("Fichiers JSON", "*.json")]
    )

    if not file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validation Pydantic
        payload = TimeSeriesData(**data)

        lbl.config(
            text=f"Fichier chargé avec succès :\n{file_path}",
            fg="green"
        )

        print("Payload validé :")
        print(payload)

    except json.JSONDecodeError:
        lbl.config(text="❌ JSON invalide", fg="red")

    except ValidationError as e:
        lbl.config(text="❌ Structure non conforme au modèle", fg="red")
        print(e)

# =========================
# UI
# =========================
btn_file = Button(
    root,
    text="Sélectionner un fichier JSON",
    command=choose_file
)
btn_file.pack(pady=20)

lbl = Label(
    root,
    text="Aucun fichier sélectionné",
    font=("Arial", 11),
    wraplength=450
)
lbl.pack(pady=10)

# =========================
# Lancement
# =========================
root.mainloop()