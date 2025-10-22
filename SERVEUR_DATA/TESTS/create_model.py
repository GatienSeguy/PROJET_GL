import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Crée et sauvegarde le modèle
model = SimpleNet()
torch.save(model.state_dict(), "mon_modele.pth")
print("✅ Modèle créé et sauvegardé dans mon_modele.pth")