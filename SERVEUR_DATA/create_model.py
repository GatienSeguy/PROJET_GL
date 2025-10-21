import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Crée le modèle
model = Net()

# Sauvegarde-le dans un fichier
torch.save(model.state_dict(), "mon_modele.pth")
print("Modèle créé et sauvegardé dans mon_modele.pth ✅")
