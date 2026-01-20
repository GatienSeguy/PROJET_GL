# ============================================================================
# training_multistep.py - Entraînement Multi-Horizon (Stratégie DIRECT)
# ============================================================================
"""
Stratégie DIRECT pour la prédiction multi-step.

Références:
- Taieb, S. B., et al. (2012). "A review and comparison of strategies for 
  multi-step ahead time series forecasting". Expert Systems with Applications.
- Chevillon, G. (2007). "Direct multi-step estimation and forecasting". 
  Journal of Economic Surveys.

Au lieu de prédire y_{t+1} puis réinjecter pour y_{t+2}, on entraîne
le modèle à prédire DIRECTEMENT [y_{t+1}, y_{t+2}, ..., y_{t+H}].

Avantages:
- Pas d'accumulation d'erreurs
- Chaque horizon est optimisé indépendamment
- Plus stable sur les longues prédictions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Iterator, Dict, Any
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# CONSTRUCTION DES DONNÉES MULTI-HORIZON
# ============================================================================

def build_multistep_supervised_tensors(
    values: List[float],
    window_size: int = 15,
    horizon: int = 10,
    step: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construit des tenseurs (X, Y) pour la prédiction multi-horizon DIRECTE.
    
    Args:
        values: Liste des valeurs de la série
        window_size: Taille de la fenêtre d'entrée
        horizon: Nombre de pas à prédire (sortie)
        step: Pas d'échantillonnage
    
    Returns:
        X: (N, window_size) - fenêtres d'entrée
        Y: (N, horizon) - horizons de sortie correspondants
    
    Exemple avec window_size=3, horizon=2:
        values = [1, 2, 3, 4, 5, 6, 7]
        X[0] = [1, 2, 3], Y[0] = [4, 5]
        X[1] = [2, 3, 4], Y[1] = [5, 6]
        X[2] = [3, 4, 5], Y[2] = [6, 7]
    """
    n = len(values)
    if n < window_size + horizon:
        return torch.empty(0, window_size), torch.empty(0, horizon)
    
    X_list = []
    Y_list = []
    
    max_start = n - window_size - horizon + 1
    
    for i in range(0, max_start, step):
        # Fenêtre d'entrée
        x_window = values[i:i + window_size]
        
        # Horizons de sortie
        y_horizon = values[i + window_size:i + window_size + horizon]
        
        # Vérifier les None
        if any(v is None for v in x_window) or any(v is None for v in y_horizon):
            continue
        
        X_list.append(x_window)
        Y_list.append(y_horizon)
    
    if not X_list:
        return torch.empty(0, window_size), torch.empty(0, horizon)
    
    X = torch.tensor(np.array(X_list, dtype=np.float32))
    Y = torch.tensor(np.array(Y_list, dtype=np.float32))
    
    return X, Y


# ============================================================================
# MODÈLE MULTI-HORIZON
# ============================================================================

class MultiHorizonLSTM(nn.Module):
    """
    LSTM avec sortie multi-horizon directe.
    
    Architecture:
    - LSTM encoder
    - Linear decoder vers H outputs
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizon: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        
        # Encoder LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder multi-horizon
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) ou (batch, seq_len, 1)
        
        Returns:
            (batch, horizon) - prédictions pour les H prochains pas
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, T) -> (B, T, 1)
        
        # Encoder
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Utiliser le dernier hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Decoder multi-horizon
        out = self.fc(last_hidden)  # (batch, horizon)
        
        return out


class MultiHorizonMLP(nn.Module):
    """
    MLP avec sortie multi-horizon directe.
    """
    
    def __init__(
        self,
        window_size: int = 15,
        hidden_size: int = 128,
        num_layers: int = 3,
        horizon: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.window_size = window_size
        self.horizon = horizon
        
        layers = []
        in_dim = window_size
        
        for i in range(num_layers):
            out_dim = hidden_size if i < num_layers - 1 else horizon
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            in_dim = hidden_size
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, window_size) ou (batch, window_size, 1)
        
        Returns:
            (batch, horizon)
        """
        if x.ndim == 3:
            x = x.squeeze(-1)
        return self.net(x)


# ============================================================================
# ENTRAÎNEMENT MULTI-HORIZON
# ============================================================================

def train_multistep(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    model_type: str = "lstm",
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 100,
    device: str = "cpu",
    patience: int = 20,
) -> Iterator[Dict[str, Any]]:
    """
    Entraîne un modèle multi-horizon avec early stopping.
    
    Args:
        X: (N, window_size) entrées
        Y: (N, horizon) sorties
        model_type: "lstm" ou "mlp"
        ... paramètres d'architecture et d'entraînement
    
    Yields:
        Événements de progression (epochs, avg_loss, ...)
    
    Returns:
        Modèle entraîné (via StopIteration.value)
    """
    window_size = X.shape[1]
    horizon = Y.shape[1]
    
    # Créer le modèle
    if model_type.lower() == "lstm":
        model = MultiHorizonLSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
            dropout=dropout
        )
    else:
        model = MultiHorizonMLP(
            window_size=window_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
            dropout=dropout
        )
    
    model = model.to(device)
    
    # Loss et optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2
    )
    
    # DataLoader
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Early stopping
    best_loss = float('inf')
    best_model_state = None
    no_improve_count = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        
        avg_loss = total_loss / n_samples
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Yield progression
        yield {
            "epochs": epoch,
            "avg_loss": avg_loss,
            "best_loss": best_loss,
            "lr": optimizer.param_groups[0]['lr']
        }
        
        # Early stopping check
        if no_improve_count >= patience:
            print(f"[TRAIN] Early stopping à l'epoch {epoch}")
            break
    
    # Restaurer le meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    yield {"done": True, "final_loss": best_loss}
    return model


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Test de training_multistep.py")
    
    # Données de test (sinusoïde)
    t = np.linspace(0, 10 * np.pi, 500)
    values = np.sin(t).tolist()
    
    window_size = 20
    horizon = 10
    
    X, Y = build_multistep_supervised_tensors(values, window_size, horizon)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Normalisation
    mean = X.mean()
    std = X.std()
    X_norm = (X - mean) / std
    Y_norm = (Y - mean) / std
    
    # Entraînement
    gen = train_multistep(
        X_norm, Y_norm,
        model_type="mlp",
        hidden_size=64,
        num_layers=3,
        epochs=50,
        device="cpu"
    )
    
    model = None
    for msg in gen:
        if msg.get("done"):
            print(f"✓ Terminé: loss finale = {msg['final_loss']:.6f}")
        elif "epochs" in msg:
            if msg["epochs"] % 10 == 0:
                print(f"  Epoch {msg['epochs']}: loss = {msg['avg_loss']:.6f}")
    
    try:
        model = gen.send(None)
    except StopIteration as e:
        model = e.value
    
    print(f"✓ Modèle entraîné: {type(model).__name__}")