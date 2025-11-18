from dataclasses import dataclass
from typing import Optional, List
import json
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration du modèle CNN-DA-RGRU"""
    # Architecture
    num_features: int = 1
    conv_channels: int = 64
    hidden_size: int = 128
    output_size: int = 1
    pred_steps: int = 6
    kernel_size: int = 6
    pool_size: int = 2
    
    # Entraînement
    window_size: int = 15
    batch_size: int = 32
    num_epochs: int = 2000
    learning_rate: float = 0.0001
    patience: int = 20
    
    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Autres
    device: str = 'cuda'
    seed: int = 42
    
    def __post_init__(self):
        """Validation de la configuration"""
        assert self.train_ratio + self.val_ratio + self.test_ratio <= 1.0, \
            "Les ratios train/val/test doivent être <= 1.0"
        assert self.window_size > 0, "window_size doit être > 0"
        assert self.pred_steps > 0, "pred_steps doit être > 0"
    
    def save(self, path: Path):
        """Sauvegarde la configuration"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    @classmethod
    def load(cls, path: Path):
        """Charge une configuration"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def __str__(self):
        """Affichage propre de la configuration"""
        lines = ["Configuration du modèle:"]
        for key, value in self.__dict__.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class DataConfig:
    """Configuration des données"""
    target_file: Path
    exog_files: Optional[List[Path]] = None
    has_exog: bool = False
    
    def __post_init__(self):
        # Conversion en Path
        self.target_file = Path(self.target_file)
        
        # Vérification de l'existence du fichier
        if not self.target_file.exists():
            raise FileNotFoundError(f"Fichier cible non trouvé: {self.target_file}")
        
        # Variables exogènes
        if self.exog_files:
            self.exog_files = [Path(f) for f in self.exog_files]
            # Vérification de l'existence
            for f in self.exog_files:
                if not f.exists():
                    raise FileNotFoundError(f"Fichier exogène non trouvé: {f}")
            self.has_exog = True