import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    MODELE IRMA (version Conv1D pour séries temporelles).

    But
    ----
    Réseau convolutionnel 1D simple :
    prend en entrée une séquence (B, in_dim, T)
    et produit une sortie (B, out_dim, T) ou réduite selon stride.

    Paramètres __init__
    -------------------
    in_dim : int
        Nombre de canaux d'entrée (ex. 1 pour un signal univarié).
    hidden_dim : int
        Largeur (nombre de filtres) des couches internes.
    out_dim : int
        Nombre de canaux de sortie.
    num_blocks : int
        Nombre de blocs convolutionnels.
    activation : str
        "relu", "gelu", "tanh" ou "sigmoid".
    padding : int
        Taille du padding (0, 1, etc.).
    stride : int
        Pas de déplacement du filtre.
    kernel_size : int
        Taille du filtre (fenêtre temporelle).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_blocks: int = 2,
        activation: str = "relu",
        padding: int = 1,
        stride: int = 1,
        kernel_size: int = 3
    ) -> None:
        super().__init__()
        assert in_dim > 0 and hidden_dim > 0 and out_dim > 0, "Dims > 0"
        assert num_blocks >= 1, "Au moins 1 bloc"
        assert activation in {"relu", "gelu", "tanh", "sigmoid"}, "Activation non supportée"

        # --- Sélection d'activation ---
        self.act_name = activation
        self.act = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }[activation]

        # --- Couche d'entrée ---
        self.conv_in = nn.Conv1d(
            in_channels=in_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # --- Blocs internes ---
        blocks = []
        for _ in range(num_blocks - 1):
            blocks += [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                self._new_act()
            ]
        self.backbone = nn.Sequential(*blocks)

        # --- Couche de sortie ---
        self.conv_out = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # --- Initialisation ---
        self.reset_parameters()

    def _new_act(self) -> nn.Module:
        """Renvoie une nouvelle instance de la fonction d’activation."""
        if self.act_name == "relu":
            return nn.ReLU(inplace=True)
        if self.act_name == "gelu":
            return nn.GELU()
        if self.act_name == "sigmoid":
            return nn.Sigmoid()
        return nn.Tanh()

    def reset_parameters(self) -> None:
        """Initialisation des poids (Kaiming adaptée pour conv1d)."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_dim, T)
        retourne : (B, out_dim, T')
        """
        z = self.conv_in(x)
        z = self.act(z)
        if len(self.backbone) > 0:
            z = self.backbone(z)
        out = self.conv_out(z)
        return out

    def num_parameters(self) -> int:
        """Retourne le nombre total de paramètres du modèle."""
        return sum(p.numel() for p in self.parameters())
