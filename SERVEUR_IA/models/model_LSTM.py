import torch
import torch.nn as nn

class LSTM(nn.Module): 
    """
    LSTM modulaire pour séries temporelles, inspiré de la structure Conv1D.

    Structure :
    - Couche d'entrée LSTM
    - Blocs internes empilés LSTM (num_layers-1)
    - Couche de sortie linéaire appliquée sur chaque timestep
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,#
        out_dim: int,
        nb_couches: int = 2,#
        bidirectional: bool = False, #New
        batch_first: bool = True #New
    ):
        super().__init__()
        assert in_dim > 0 and hidden_dim > 0 and out_dim > 0, "Dims > 0"
        assert nb_couches >= 1, "Au moins 1 bloc"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.nb_couches = nb_couches
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        # --- Couche d'entrée LSTM ---
        self.lstm_in = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

        # --- Blocs internes LSTM empilés ---
        blocks = []
        for _ in range(nb_couches - 1):
            blocks.append(
                nn.LSTM(
                    input_size=hidden_dim * (2 if bidirectional else 1),
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=batch_first,
                    bidirectional=bidirectional
                )
            )
        self.backbone = nn.ModuleList(blocks)  # on applique chaque bloc séquentiellement

        # --- Couche de sortie linéaire ---
        factor = 2 if bidirectional else 1
        self.fc_out = nn.Linear(hidden_dim * factor, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialisation simple."""
        for name, param in self.lstm_in.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for lstm in self.backbone:
            for name, param in lstm.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """
        x : (B, T, in_dim) si batch_first=True
        """
        out, _ = self.lstm_in(x)
        for lstm in self.backbone:
            out, _ = lstm(out)
        out = self.fc_out(out)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
