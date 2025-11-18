import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from config import ModelConfig, DataConfig


class TimeSeriesDataset(Dataset):
    """Dataset pour séries temporelles"""
    
    def __init__(
        self, 
        data: Dict,
        window_size: int,
        pred_steps: int,
        exog_data: Optional[List[Dict]] = None,
        is_training: bool = True
    ):
        """
        Args:
            data: dict avec 'timestamps' et 'values'
            window_size: taille de la fenêtre d'entrée
            pred_steps: nombre de pas à prédire
            exog_data: liste de dicts de variables exogènes
            is_training: si True, calcule les stats de normalisation
        """
        self.window_size = window_size
        self.pred_steps = pred_steps
        self.is_training = is_training
        
        # Chargement et nettoyage des données
        self.timestamps = np.array(data['timestamps'])
        self.values = self._clean_data(np.array(data['values'], dtype=float))
        
        print(f"  Taille après nettoyage: {len(self.values)}")
        print(f"  Min: {self.values.min():.4f}, Max: {self.values.max():.4f}")
        
        # Variables exogènes
        self.exog = None
        if exog_data is not None and len(exog_data) > 0:
            self.exog = self._load_exog_data(exog_data)
            print(f"  Variables exogènes: {self.exog.shape[1]}")
        
        # Normalisation
        if is_training:
            self._compute_normalization_stats()
        
        self._normalize_data()
    
    def _clean_data(self, values: np.ndarray) -> np.ndarray:
        """Nettoie les données (gère les NaN)"""
        # Comptage des NaN
        n_nan = np.isnan(values).sum()
        if n_nan > 0:
            print(f"  ⚠ {n_nan} valeurs manquantes détectées")
        
        values_series = pd.Series(values)
        
        # Interpolation linéaire
        values_series = values_series.interpolate(method='linear', limit_direction='both')
        
        # Remplissage des valeurs restantes
        if values_series.isna().any():
            values_series = values_series.fillna(values_series.mean())
        
        return values_series.values.reshape(-1, 1)
    
    def _load_exog_data(self, exog_data: List[Dict]) -> np.ndarray:
        """Charge et nettoie les variables exogènes"""
        exog_arrays = []
        for i, exog in enumerate(exog_data):
            exog_values = self._clean_data(np.array(exog['values'], dtype=float))
            exog_arrays.append(exog_values)
        return np.concatenate(exog_arrays, axis=1)
    
    def _compute_normalization_stats(self):
        """Calcule les statistiques de normalisation"""
        self.mean = float(self.values.mean())
        self.std = float(self.values.std())
        
        # Protection contre std = 0
        if self.std < 1e-8:
            print("  ⚠ Écart-type très faible, utilisation de std = 1.0")
            self.std = 1.0
        
        print(f"  Normalisation - Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        
        if self.exog is not None:
            self.exog_mean = self.exog.mean(axis=0, keepdims=True)
            self.exog_std = self.exog.std(axis=0, keepdims=True)
            self.exog_std[self.exog_std < 1e-8] = 1.0
        else:
            self.exog_mean = None
            self.exog_std = None
    
    def _normalize_data(self):
        """Normalise les données"""
        self.values_norm = (self.values - self.mean) / self.std
        
        if self.exog is not None:
            self.exog_norm = (self.exog - self.exog_mean) / self.exog_std
    
    def set_normalization_stats(self, mean: float, std: float, 
                               exog_mean: Optional[np.ndarray] = None,
                               exog_std: Optional[np.ndarray] = None):
        """Définit les statistiques de normalisation (pour test/val)"""
        self.mean = mean
        self.std = std
        self.exog_mean = exog_mean
        self.exog_std = exog_std
        self._normalize_data()
    
    def get_normalization_stats(self) -> Dict:
        """Retourne les statistiques de normalisation"""
        return {
            'mean': self.mean,
            'std': self.std,
            'exog_mean': self.exog_mean.tolist() if self.exog_mean is not None else None,
            'exog_std': self.exog_std.tolist() if self.exog_std is not None else None
        }
    
    def denormalize(self, values: np.ndarray) -> np.ndarray:
        """Dénormalise les valeurs"""
        return values * self.std + self.mean
    
    def __len__(self) -> int:
        return len(self.values) - self.window_size - self.pred_steps + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fenêtre d'entrée
        x_target = self.values_norm[idx:idx + self.window_size]
        
        if self.exog is not None:
            x_exog = self.exog_norm[idx:idx + self.window_size]
            x = np.concatenate([x_target, x_exog], axis=1)
        else:
            x = x_target
        
        # Valeurs à prédire
        y = self.values_norm[idx + self.window_size:idx + self.window_size + self.pred_steps]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


class DataManager:
    """Gestionnaire de données pour l'entraînement"""
    
    def __init__(self, data_config: DataConfig, model_config: ModelConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.datasets = {}
        self.loaders = {}
        self.norm_stats = None
    
    def load_data(self) -> Tuple[Dict, Optional[List[Dict]]]:
        """Charge les données depuis les fichiers"""
        print("="*60)
        print("Chargement des données")
        print("="*60)
        
        # Vérification et affichage du chemin
        print(f"Fichier cible: {self.data_config.target_file.absolute()}")
        
        # Données cibles
        try:
            with open(self.data_config.target_file, 'r') as f:
                target_data = json.load(f)
            print(f"✓ Données cibles: {len(target_data['values'])} points")
        except FileNotFoundError:
            print(f"✗ Fichier non trouvé: {self.data_config.target_file}")
            print(f"  Chemin absolu: {self.data_config.target_file.absolute()}")
            raise
        except json.JSONDecodeError as e:
            print(f"✗ Erreur de parsing JSON: {e}")
            raise
        
        # Variables exogènes
        exog_data = None
        if self.data_config.has_exog and self.data_config.exog_files:
            exog_data = []
            for i, exog_file in enumerate(self.data_config.exog_files):
                try:
                    with open(exog_file, 'r') as f:
                        exog_data.append(json.load(f))
                except FileNotFoundError:
                    print(f"⚠ Variable exogène {i} non trouvée: {exog_file}")
            
            if exog_data:
                print(f"✓ Variables exogènes: {len(exog_data)}")
            else:
                exog_data = None
        
        return target_data, exog_data
    
    def prepare_datasets(self, target_data: Dict, exog_data: Optional[List[Dict]] = None) -> int:
        """Prépare les datasets train/val/test"""
        print("\nCréation des datasets...")
        
        # Création du dataset complet
        full_dataset = TimeSeriesDataset(
            target_data,
            self.model_config.window_size,
            self.model_config.pred_steps,
            exog_data,
            is_training=True
        )
        
        # Sauvegarde des stats de normalisation
        self.norm_stats = full_dataset.get_normalization_stats()
        
        # Split des données
        total_size = len(full_dataset)
        train_size = int(total_size * self.model_config.train_ratio)
        val_size = int(total_size * self.model_config.val_ratio)
        test_size = total_size - train_size - val_size
        
        if train_size == 0 or val_size == 0:
            raise ValueError(f"Dataset trop petit: total={total_size}, train={train_size}, val={val_size}")
        
        # Création des subsets
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))
        
        self.datasets['train'] = torch.utils.data.Subset(full_dataset, train_indices)
        self.datasets['val'] = torch.utils.data.Subset(full_dataset, val_indices)
        self.datasets['test'] = torch.utils.data.Subset(full_dataset, test_indices)
        
        print(f"✓ Train: {len(self.datasets['train'])} échantillons")
        print(f"✓ Validation: {len(self.datasets['val'])} échantillons")
        print(f"✓ Test: {len(self.datasets['test'])} échantillons")
        
        return full_dataset[0][0].shape[1]  # num_features
    
    def create_dataloaders(self):
        """Crée les DataLoaders"""
        self.loaders['train'] = DataLoader(
            self.datasets['train'],
            batch_size=self.model_config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.loaders['val'] = DataLoader(
            self.datasets['val'],
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.loaders['test'] = DataLoader(
            self.datasets['test'],
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def save_normalization_stats(self, save_dir: Path):
        """Sauvegarde les statistiques de normalisation"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'normalization_stats.npy', self.norm_stats)
        print(f"✓ Stats de normalisation sauvegardées: {save_dir / 'normalization_stats.npy'}")