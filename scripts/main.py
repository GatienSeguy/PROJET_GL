import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import os

from config import ModelConfig, DataConfig
from data_manager import DataManager
from model_CNN_DA_RGRU import CNN_DA_RGRU
from trainer import Trainer
from evaluator import Evaluator
from predictor import Predictor


class CNNDARGRUPipeline:
    """Pipeline principal pour CNN-DA-RGRU"""
    
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        self.model_config = model_config
        self.data_config = data_config
        self.device = torch.device(model_config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialisation
        self.data_manager = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.predictor = None
        
        print("="*60)
        print("CNN-DA-RGRU Pipeline")
        print("="*60)
        print(f"Device: {self.device}")
    
    def setup(self):
        """Configuration initiale"""
        # Seed pour reproductibilité
        torch.manual_seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        
        # Data Manager
        self.data_manager = DataManager(self.data_config, self.model_config)
    
    def prepare_data(self):
        """Prépare les données"""
        # Chargement
        target_data, exog_data = self.data_manager.load_data()
        
        # Création des datasets
        num_features = self.data_manager.prepare_datasets(target_data, exog_data)
        self.model_config.num_features = num_features
        
        # Création des dataloaders
        self.data_manager.create_dataloaders()
        
        return num_features
    
    def build_model(self):
        """Construit le modèle"""
        print("\n" + "="*60)
        print("Construction du modèle")
        print("="*60)
        
        self.model = CNN_DA_RGRU(
            num_features=self.model_config.num_features,
            conv_channels=self.model_config.conv_channels,
            hidden_size=self.model_config.hidden_size,
            output_size=self.model_config.output_size,
            pred_steps=self.model_config.pred_steps,
            kernel_size=self.model_config.kernel_size,
            pool_size=self.model_config.pool_size
        )
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Modèle créé")
        print(f"  Paramètres totaux: {total_params:,}")
        print(f"  Paramètres entraînables: {trainable_params:,}")
    
    def train(self, save_dir: Path = Path('checkpoints')):
        """Entraîne le modèle"""
        self.trainer = Trainer(self.model, self.model_config, self.device, save_dir)
        
        train_losses, val_losses = self.trainer.train(
            self.data_manager.loaders['train'],
            self.data_manager.loaders['val']
        )
        
        # Sauvegarde de la configuration
        self.model_config.save(save_dir / 'config.json')
        self.data_manager.save_normalization_stats(save_dir)
        
        print("\n" + "="*60)
        print("Entraînement terminé")
        print("="*60)
        print(f"Meilleure val loss: {min(val_losses):.6f}")
        print(f"Modèles sauvegardés dans: {save_dir}/")
    
    #####
    def evaluate(self, save_dir: Path = Path('results')):
        """Évalue le modèle avec visualisation complète"""
        self.evaluator = Evaluator(self.model, self.model_config, self.device)
        
        def denormalize_fn(values):
            return values * self.data_manager.norm_stats['std'] + self.data_manager.norm_stats['mean']
        
        # Évaluation sur test
        test_results = self.evaluator.evaluate_on_loader(
            self.data_manager.loaders['test'],
            denormalize_fn,
            phase_name="Test"
        )
        
        # Graphiques détaillés test
        self.evaluator.plot_test_detailed(test_results, save_dir)
        
        # Graphique complet avec train/val/test
        full_dataset = self.data_manager.datasets['train'].dataset
        train_indices = self.data_manager.datasets['train'].indices
        val_indices = self.data_manager.datasets['val'].indices
        test_indices = self.data_manager.datasets['test'].indices
        
        self.evaluator.plot_complete_analysis(
            full_dataset,
            train_indices,
            val_indices,
            test_indices,
            denormalize_fn,
            save_dir
        )
        
        # Sauvegarde JSON
        results_to_save = {
            'test_metrics_per_step': test_results['metrics_per_step'],
            'test_global_metrics': test_results['global_metrics']
        }
        
        save_dir = Path(save_dir)
        with open(save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=4)
        
        self.log(f"\n✓ Évaluation terminée")


    ####
    
    def predict(self, data: dict, exog_data: list = None, save_dir: Path = Path('predictions')):
        """Fait une prédiction"""
        self.predictor = Predictor(
            self.model,
            self.model_config,
            self.data_manager.norm_stats,
            self.device
        )
        
        predictions = self.predictor.predict(data, exog_data)
        
        # Visualisation
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.predictor.plot_prediction(
            data,
            predictions,
            save_path=save_dir / f'forecast_{timestamp_str}.png'
        )
        
        # Sauvegarde
        with open(save_dir / f'predictions_{timestamp_str}.json', 'w') as f:
            json.dump(predictions, f, indent=4)
        
        return predictions
    
    def load_model(self, checkpoint_path: Path):
        """Charge un modèle pré-entraîné"""
        print(f"\nChargement du modèle depuis {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Modèle chargé (époque {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.6f})")


def get_data_file():
    """Interface pour sélectionner le fichier de données"""
    print("\n" + "="*60)
    print("Sélection du fichier de données")
    print("="*60)
    
    # Proposition de chemins courants
    common_paths = [
        Path('PROJET_GL/Datas/EURO.json'),
        Path('Datas/EURO.json'),
        Path('data/EURO.json'),
        Path('EURO.json')
    ]
    
    print("\nChemins suggérés:")
    for i, path in enumerate(common_paths, 1):
        exists = "✓" if path.exists() else "✗"
        print(f"  {i}. {exists} {path}")
    
    print(f"\n  {len(common_paths)+1}. Entrer un chemin personnalisé")
    print(f"  {len(common_paths)+2}. Rechercher automatiquement")
    
    choice = input(f"\nChoisir (1-{len(common_paths)+2}): ").strip()
    
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(common_paths):
            selected_path = common_paths[choice_num - 1]
            if selected_path.exists():
                return selected_path
            else:
                print(f"✗ Fichier non trouvé: {selected_path}")
                return get_data_file()
        
        elif choice_num == len(common_paths) + 1:
            # Chemin personnalisé
            custom_path = input("Entrer le chemin du fichier: ").strip()
            custom_path = Path(custom_path)
            if custom_path.exists():
                return custom_path
            else:
                print(f"✗ Fichier non trouvé: {custom_path}")
                return get_data_file()
        
        elif choice_num == len(common_paths) + 2:
            # Recherche automatique
            print("\nRecherche de fichiers EURO.json...")
            found_files = list(Path('.').rglob('EURO.json'))
            
            if not found_files:
                print("✗ Aucun fichier EURO.json trouvé")
                return get_data_file()
            
            print(f"\n{len(found_files)} fichier(s) trouvé(s):")
            for i, f in enumerate(found_files, 1):
                print(f"  {i}. {f}")
            
            file_choice = int(input(f"\nChoisir (1-{len(found_files)}): ").strip())
            if 1 <= file_choice <= len(found_files):
                return found_files[file_choice - 1]
            else:
                return get_data_file()
    
    except (ValueError, IndexError):
        print("✗ Choix invalide")
        return get_data_file()


def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("CNN-DA-RGRU - Prédiction de Séries Temporelles")
    print("="*60)
    print(f"Répertoire de travail: {Path.cwd()}")
    
    # Sélection du fichier de données
    data_file = get_data_file()
    print(f"\n✓ Fichier sélectionné: {data_file}")
    
    # Configuration du modèle
    model_config = ModelConfig(
        # Architecture
        conv_channels=64,
        hidden_size=256,
        pred_steps=6,
        kernel_size=6,
        pool_size=2,
        
        # Entraînement
        window_size=15,
        batch_size=32,
        num_epochs=2000,
        learning_rate=0.00005,
        patience=100,
        
        # Split
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        
        # Autres
        device='cuda',
        seed=42
    )
    
    # Configuration des données
    data_config = DataConfig(
        target_file=data_file,
        exog_files=None,
        has_exog=False
    )
    
    print("\n" + str(model_config))
    
    # Création du pipeline
    pipeline = CNNDARGRUPipeline(model_config, data_config)
    
    # Mode interactif
    print("\n" + "="*60)
    print("Menu Principal")
    print("="*60)
    print("1. Entraîner un nouveau modèle")
    print("2. Évaluer un modèle existant")
    print("3. Faire une prédiction")
    print("4. Pipeline complet (train + eval + predict)")
    
    choice = input("\nChoisissez une option (1-4): ").strip()
    
    try:
        if choice == '1':
            # Entraînement
            pipeline.setup()
            pipeline.prepare_data()
            pipeline.build_model()
            pipeline.train()
            
        elif choice == '2':
            # Évaluation
            pipeline.setup()
            pipeline.prepare_data()
            pipeline.build_model()
            pipeline.load_model(Path('checkpoints/best_model.pth'))
            pipeline.evaluate()
            
        elif choice == '3':
            # Prédiction
            pipeline.setup()
            
            # Chargement des données pour prédiction
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            # Chargement du modèle
            config_saved = ModelConfig.load(Path('checkpoints/config.json'))
            pipeline.model_config = config_saved
            pipeline.data_manager = DataManager(data_config, config_saved)
            
            # Chargement des stats de normalisation
            pipeline.data_manager.norm_stats = np.load(
                'checkpoints/normalization_stats.npy',
                allow_pickle=True
            ).item()
            
            pipeline.build_model()
            pipeline.load_model(Path('checkpoints/best_model.pth'))
            pipeline.predict(data)
            
        elif choice == '4':
            # Pipeline complet
            pipeline.setup()
            pipeline.prepare_data()
            pipeline.build_model()
            pipeline.train()
            pipeline.evaluate()
            
            # Prédiction sur les dernières données
            with open(data_file, 'r') as f:
                data = json.load(f)
            pipeline.predict(data)
            
        else:
            print("✗ Option invalide")
            return
        
        print("\n" + "="*60)
        print("✓ Terminé avec succès!")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ ERREUR")
        print("="*60)
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()