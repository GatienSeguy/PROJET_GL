"""
SYSTÈME COMPLET DE PRÉDICTION LSTM - VERSION PROPRE
====================================================

Architecture:
- LSTM qui prédit N steps futurs d'un coup (pas autorégressif 1 par 1)
- Training classique
- Test en prédiction VRAIE (pas de triche avec les vraies valeurs)
- Prédiction future

"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Iterator
import matplotlib.pyplot as plt


# ============================================================================
# 1. MODÈLE LSTM SIMPLE ET PROPRE
# ============================================================================

class SimpleLSTM(nn.Module):
    """
    LSTM pour prédiction de séries temporelles
    
    Input:  (batch, seq_len, input_dim)
    Output: (batch, pred_steps)  <- Prédit TOUS les steps d'un coup !
    """
    
    def __init__(
        self, 
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_steps: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_steps = pred_steps
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Projection vers pred_steps sorties
        self.fc = nn.Linear(hidden_dim, pred_steps)
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        output: (batch, pred_steps)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        
        # Prendre la dernière sortie temporelle
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Prédire les pred_steps valeurs futures
        predictions = self.fc(last_output)  # (batch, pred_steps)
        
        return predictions


# ============================================================================
# 2. PRÉPARATION DES DONNÉES
# ============================================================================

def prepare_data(
    values: List[float],
    window_size: int = 30,
    pred_steps: int = 6,
    train_ratio: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Prépare les données pour l'entraînement et le test
    
    Structure:
    - Fenêtre de window_size points → prédit les pred_steps points suivants
    
    Returns:
        X_train, y_train, X_test, y_test, norm_stats
    """
    
    values = np.array(values, dtype=np.float32)
    
    # Normalisation ROBUSTE avec clipping
    mean = np.median(values)  # Médiane au lieu de moyenne
    std = np.std(values)
    
    # Clipper les valeurs extrêmes AVANT normalisation
    lower = np.percentile(values, 1)
    upper = np.percentile(values, 99)
    values = np.clip(values, lower, upper)
    
    # Normalisation
    values_norm = (values - mean) / (std + 1e-8)  # Epsilon pour éviter div par 0
    
    # Clipper après normalisation aussi
    values_norm = np.clip(values_norm, -5, 5)
    
    norm_stats = {'mean': float(mean), 'std': float(std + 1e-8)}
    
    # Création des séquences
    X, y = [], []
    
    for i in range(len(values_norm) - window_size - pred_steps + 1):
        # Fenêtre d'entrée
        X.append(values_norm[i:i + window_size])
        
        # Cibles (les pred_steps valeurs suivantes)
        y.append(values_norm[i + window_size : i + window_size + pred_steps])
    
    X = np.array(X).reshape(-1, window_size, 1)  # (N, window_size, 1)
    y = np.array(y)  # (N, pred_steps)
    
    # Split train/test
    n_train = int(len(X) * train_ratio)
    
    X_train = torch.FloatTensor(X[:n_train])
    y_train = torch.FloatTensor(y[:n_train])
    X_test = torch.FloatTensor(X[n_train:])
    y_test = torch.FloatTensor(y[n_train:])
    
    print(f"\n[DATA] Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  Norm range: [{values_norm.min():.2f}, {values_norm.max():.2f}]")
    
    return X_train, y_train, X_test, y_test, norm_stats


# ============================================================================
# 3. ENTRAÎNEMENT
# ============================================================================

def train_model(
    model: SimpleLSTM,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.0001,  # LR réduit mais pas trop
    device: str = 'cpu'
) -> Iterator[Dict]:
    """Entraîne le modèle et yield les métriques"""
    
    model = model.to(device)
    model.train()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            
            # Vérifier si NaN
            if torch.isnan(loss):
                print(f"\n⚠️  NaN détecté à l'epoch {epoch} ! Réduction LR...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                continue
            
            loss.backward()
            
            # GRADIENT CLIPPING pour éviter les NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if n_batches == 0:
            continue
            
        avg_loss = total_loss / n_batches
        
        yield {
            'epoch': epoch,
            'loss': avg_loss
        }
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")


# ============================================================================
# 4. TEST - PRÉDICTION FUTURE RÉELLE
# ============================================================================

def test_model_future_prediction(
    model: SimpleLSTM,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    norm_stats: Dict,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Test avec VRAIE prédiction future
    
    Principe:
    - On prend chaque fenêtre de test
    - On prédit les pred_steps suivants
    - On compare avec les vraies valeurs
    - PAS d'autorégressif compliqué, juste des prédictions directes
    """
    
    model = model.to(device)
    model.eval()
    
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            # Prendre une fenêtre
            x = X_test[i:i+1].to(device)  # (1, window_size, 1)
            
            # Prédire les pred_steps futurs
            pred_norm = model(x).cpu().numpy()[0]  # (pred_steps,)
            
            # Dénormaliser
            pred = pred_norm * std + mean
            
            # Vraies valeurs
            target_norm = y_test[i].numpy()  # (pred_steps,)
            target = target_norm * std + mean
            
            all_predictions.extend(pred)
            all_targets.extend(target)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Métriques
    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(mse)
    
    var_y = np.var(all_targets)
    r2 = 1.0 - (mse / var_y) if var_y > 0 else None
    
    metrics = {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2) if r2 is not None else None
    }
    
    print(f"\n[TEST] Métriques:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²:   {r2:.6f}" if r2 is not None else "  R²:   N/A")
    
    return all_predictions, all_targets, metrics


# ============================================================================
# 5. PRÉDICTION FUTURE (AU-DELÀ DES DONNÉES)
# ============================================================================

def predict_future(
    model: SimpleLSTM,
    last_window: np.ndarray,
    norm_stats: Dict,
    n_future: int = 30,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Prédit n_future points dans le futur
    
    Mode autorégressif:
    - Utilise les prédictions pour construire les prochaines fenêtres
    """
    
    model = model.to(device)
    model.eval()
    
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    # Normaliser la fenêtre initiale
    window = (last_window - mean) / std
    window = window.reshape(-1)  # Flatten
    
    predictions = []
    
    with torch.no_grad():
        while len(predictions) < n_future:
            # Préparer l'entrée
            x = torch.FloatTensor(window).reshape(1, -1, 1).to(device)
            
            # Prédire
            pred_norm = model(x).cpu().numpy()[0]  # (pred_steps,)
            
            # Ajouter aux prédictions
            for p in pred_norm:
                if len(predictions) < n_future:
                    predictions.append(p)
            
            # Glisser la fenêtre (utilise les prédictions)
            window = np.concatenate([window[len(pred_norm):], pred_norm])
    
    # Dénormaliser
    predictions = np.array(predictions) * std + mean
    
    return predictions


# ============================================================================
# 6. VISUALISATION
# ============================================================================

def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    future_pred: np.ndarray = None,
    title: str = "Prédictions LSTM"
):
    """Affiche les résultats"""
    
    plt.figure(figsize=(16, 6))
    
    # Test
    plt.plot(y_true, 'o-', label='Vraies valeurs (test)', alpha=0.7, markersize=4)
    plt.plot(y_pred, 's-', label='Prédictions (test)', alpha=0.7, markersize=4)
    
    # Future
    if future_pred is not None:
        future_x = range(len(y_true), len(y_true) + len(future_pred))
        plt.plot(future_x, future_pred, 'r^-', label='Prédictions futures', markersize=6)
        plt.axvline(len(y_true) - 1, color='orange', linestyle='--', linewidth=2, label='Début futur')
    
    plt.xlabel('Index')
    plt.ylabel('Valeur')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 7. PIPELINE COMPLET
# ============================================================================

def run_complete_pipeline(
    data_path: str = None,
    values: List[float] = None,
    window_size: int = 30,
    pred_steps: int = 6,
    hidden_dim: int = 128,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = 'cpu'
):
    """Pipeline complet d'entraînement et test"""
    
    print("\n" + "="*70)
    print("SYSTÈME DE PRÉDICTION LSTM - VERSION PROPRE")
    print("="*70)
    
    # Charger les données
    if data_path:
        with open(data_path) as f:
            data = json.load(f)
            values = data['values']
    
    # Préparer les données
    X_train, y_train, X_test, y_test, norm_stats = prepare_data(
        values, window_size, pred_steps
    )
    
    # Créer le modèle
    model = SimpleLSTM(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pred_steps=pred_steps
    )
    
    print(f"\n[MODEL] Architecture:")
    print(f"  Input: (batch, {window_size}, 1)")
    print(f"  LSTM: {num_layers} layers, {hidden_dim} hidden")
    print(f"  Output: (batch, {pred_steps})")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entraîner
    print(f"\n[TRAIN] Starting...")
    for update in train_model(model, X_train, y_train, epochs, batch_size, lr, device):
        pass
    
    # Tester
    print(f"\n[TEST] Évaluation sur données de test...")
    y_pred, y_true, metrics = test_model_future_prediction(
        model, X_test, y_test, norm_stats, device
    )
    
    # Prédire le futur
    print(f"\n[FUTURE] Prédiction de 30 points futurs...")
    last_window = (y_train[-1].numpy() * norm_stats['std'] + norm_stats['mean']).reshape(-1, 1)
    future_pred = predict_future(model, last_window, norm_stats, 30, device)
    
    # Visualiser
    print(f"\n[PLOT] Génération du graphique...")
    plot_results(y_true, y_pred, future_pred, "Prédictions LSTM - Version Propre")
    
    print("\n" + "="*70)
    print("✓ TERMINÉ")
    print("="*70)
    
    return model, metrics, y_pred, y_true, future_pred


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    model, metrics, y_pred, y_true, future_pred = run_complete_pipeline(
        data_path=PROJECT_ROOT / 'SERVEUR_DATA/datasets/CACAO.json',
        window_size=30,      # Contexte de 30 points
        pred_steps=1,        # Prédit 6 points à la fois
        hidden_dim=128,      # Réduit de 256 à 128
        num_layers=2,        # Réduit de 3 à 2
        epochs=10000,          # Plus d'époques
        batch_size=128,
        lr=0.001,
        device='cpu'         # ou 'cpu'
    )

    print(f"\n" + "="*70)
    print("RÉSULTATS FINAUX")
    print("="*70)
    print(f"MSE:  {metrics['MSE']:.6f}")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"R²:   {metrics['R2']:.6f}")
    print("="*70)