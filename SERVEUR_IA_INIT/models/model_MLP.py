import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MODELE IRMA (gabarit).
    But
    ----
    Décrire en 2-3 lignes le rôle du modèle : inputs, outputs, contraintes.

    Paramètres __init__
    -------------------
    in_dim : int
        Dimension d'entrée (ex. 2n).
    hidden_size : int
        Largeur des couches internes.
    out_dim : int
        Dimension de sortie (ex. logits ou réels).
    num_blocks : int
        Nombre d'unités répétées (si applicable).
    activation : str
        "relu", "gelu" ou "tanh" (liste fermée de choix IRMA).

     use_batchnorm : bool
        Si True, utilise BatchNorm1d dans les blocs internes.

    Formes attendues
    ----------------
    Entrée  x : (B, in_dim)
    Sortie out : (B, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int, #
        out_dim: int,
        num_blocks: int = 2,#
        activation: str = "relu",
        use_batchnorm: bool = False 
    ) -> None:
        super().__init__()
        assert in_dim > 0 and hidden_size > 0 and out_dim > 0, "Dims > 0"
        assert num_blocks >= 1, "Au moins 1 bloc"
        assert activation in {"relu", "gelu", "tanh"}, "Activation non supportée"

        # --- Sélection d'activation standardisée IRMA ---
        self.act_name = activation
        self.act = {"relu": nn.ReLU(inplace=True),
                    "gelu": nn.GELU(),
                    "tanh": nn.Tanh()}[activation]

        # --- Couche d'entrée ---
        self.fc_in = nn.Linear(in_dim, hidden_size)

        # --- Blocs internes (ex. MLP résumés ici) ---
        blocks = []
        for _ in range(num_blocks - 1):
            blocks.append(nn.Linear(hidden_size, hidden_size))
            if use_batchnorm:
                blocks.append(nn.BatchNorm1d(hidden_size))
            blocks.append(self._new_act())
        self.backbone = nn.Sequential(*blocks)  # peut être vide si num_blocks=1

        # --- Projection finale ---
        self.fc_out = nn.Linear(hidden_size, out_dim)

        # --- Initialisations (facultatif) ---
        self.reset_parameters()

    def _new_act(self) -> nn.Module:
        # Recrée l'activation choisie pour éviter le partage d'état
        if self.act_name == "relu": return nn.ReLU(inplace=True)
        if self.act_name == "gelu": return nn.GELU()
        return nn.Tanh()

    def reset_parameters(self) -> None:
        # Initialisation simple et robuste
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_dim)
        retourne : (B, out_dim)
        """
        z = self.fc_in(x)
        z = self.act(z)
        z = self.backbone(z) if len(self.backbone) > 0 else z
        out = self.fc_out(z)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())