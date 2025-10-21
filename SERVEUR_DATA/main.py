from typing import Union
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
import torch.nn as nn
import os

# Crée le dossier de sauvegarde avant tout
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI()

# Exemple de modèle simple
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Exemple Pydantic pour les items
class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


# Endpoint pour sauvegarder un modèle PyTorch
@app.post("/models/save/")
def save_model():
    model = Net()  # création du modèle ici
    model_path = os.path.join(SAVE_DIR, "mon_modele.pth")
    torch.save(model.state_dict(), model_path)
    return {"message": f"Modèle sauvegardé sous {model_path}"}

# Endpoint pour mettre à jour un item
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

# Endpoint pour recevoir un modèle depuis un client
@app.post("/models/upload/")
async def upload_model(file: UploadFile = File(...)):
    save_path = os.path.join(SAVE_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"message": f"Fichier {file.filename} sauvegardé avec succès."}
