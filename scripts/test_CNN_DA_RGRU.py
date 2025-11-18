import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from model_CNN_DA_RGRU import CNN_DA_RGRU
from scripts.trainer import TimeSeriesDataset

def calculate_metrics(y_true, y_pred):
    """
    Calcule les métriques RMSE, MAE et MAPE
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def test_model(model, test_data, exog_data, config, device, norm_stats):
    """
    Test du modèle sur données non vues
    
    Args:
        model: modèle entraîné
        test_data: données de test {"timestamp": [...], "values": [...]}
        exog_data: variables exogènes
        config: configuration du modèle
        device: 'cuda' ou 'cpu'
        norm_stats: statistiques de normalisation
    
    Returns:
        results: dictionnaire avec prédictions et métriques
    """
    # Création du dataset de test
    test_dataset = TimeSeriesDataset(
        test_data,
        config['window_size'],
        config['pred_steps'],
        exog_data
    )
    
    # Application des mêmes statistiques de normalisation que pour l'entraînement
    test_dataset.mean = norm_stats['mean']
    test_dataset.std = norm_stats['std']
    test_dataset.values_norm = (test_dataset.values - test_dataset.mean) / test_dataset.std
    
    if exog_data is not None:
        test_dataset.exog_mean = norm_stats['exog_mean']
        test_dataset.exog_std = norm_stats['exog_std']
        test_dataset.exog_norm = (test_dataset.exog - test_dataset.exog_mean) / test_dataset.exog_std
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_timestamps = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            x, y = test_dataset[i]
            x = x.unsqueeze(0).to(device)  # Ajouter dimension batch
            
            # Prédiction
            pred = model(x)
            pred = pred.squeeze(0).cpu().numpy()
            
            # Dénormalisation
            pred_denorm = test_dataset.denormalize(pred)
            y_denorm = test_dataset.denormalize(y.numpy())
            
            all_predictions.append(pred_denorm)
            all_targets.append(y_denorm)
            
            # Timestamps correspondants
            start_idx = i + config['window_size']
            end_idx = start_idx + config['pred_steps']
            all_timestamps.append(test_dataset.timestamps[start_idx:end_idx])
    
    # Conversion en arrays
    predictions = np.array(all_predictions)  # (num_samples, pred_steps, 1)
    targets = np.array(all_targets)
    
    # Calcul des métriques pour chaque step
    results = {
        'predictions': predictions,
        'targets': targets,
        'timestamps': all_timestamps,
        'metrics_per_step': []
    }
    
    for step in range(config['pred_steps']):
        y_true = targets[:, step, 0]
        y_pred = predictions[:, step, 0]
        metrics = calculate_metrics(y_true, y_pred)
        results['metrics_per_step'].append(metrics)
        
        print(f"\nMétriques pour step {step + 1}:")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE: {metrics['MAE']:.6f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # Métriques globales (moyenne sur tous les steps)
    all_targets_flat = targets.reshape(-1)
    all_predictions_flat = predictions.reshape(-1)
    global_metrics = calculate_metrics(all_targets_flat, all_predictions_flat)
    results['global_metrics'] = global_metrics
    
    print(f"\nMétriques globales:")
    print(f"  RMSE: {global_metrics['RMSE']:.6f}")
    print(f"  MAE: {global_metrics['MAE']:.6f}")
    print(f"  MAPE: {global_metrics['MAPE']:.2f}%")
    
    return results


def plot_predictions(results, config, save_dir):
    """
    Visualise les prédictions vs valeurs réelles
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = results['predictions']
    targets = results['targets']
    
    # Plot pour chaque step de prédiction
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for step in range(min(config['pred_steps'], 6)):
        ax = axes[step]
        
        y_true = targets[:, step, 0]
        y_pred = predictions[:, step, 0]
        
        ax.plot(y_true, label='Vérité terrain', alpha=0.7)
        ax.plot(y_pred, label='Prédiction', alpha=0.7)
        ax.set_title(f'Prédiction step {step + 1}')
        ax.set_xlabel('Échantillon')
        ax.set_ylabel('Valeur')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Affichage des métriques
        metrics = results['metrics_per_step'][step]
        textstr = f"RMSE: {metrics['RMSE']:.4f}\nMAE: {metrics['MAE']:.4f}\nMAPE: {metrics['MAPE']:.2f}%"
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot scatter pour chaque step
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for step in range(min(config['pred_steps'], 6)):
        ax = axes[step]
        
        y_true = targets[:, step, 0]
        y_pred = predictions[:, step, 0]
        
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Ligne de référence y=x
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prédiction parfaite')
        
        ax.set_title(f'Scatter plot step {step + 1}')
        ax.set_xlabel('Vérité terrain')
        ax.set_ylabel('Prédiction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Coefficient de corrélation
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        ax.text(0.05, 0.95, f'R² = {corr**2:.4f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot des métriques par step
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    steps = range(1, config['pred_steps'] + 1)
    rmse_values = [m['RMSE'] for m in results['metrics_per_step']]
    mae_values = [m['MAE'] for m in results['metrics_per_step']]
    mape_values = [m['MAPE'] for m in results['metrics_per_step']]
    
    axes[0].plot(steps, rmse_values, marker='o')
    axes[0].set_title('RMSE par step')
    axes[0].set_xlabel('Step de prédiction')
    axes[0].set_ylabel('RMSE')
    axes[0].grid(True)
    
    axes[1].plot(steps, mae_values, marker='o', color='orange')
    axes[1].set_title('MAE par step')
    axes[1].set_xlabel('Step de prédiction')
    axes[1].set_ylabel('MAE')
    axes[1].grid(True)
    
    axes[2].plot(steps, mape_values, marker='o', color='green')
    axes[2].set_title('MAPE par step')
    axes[2].set_xlabel('Step de prédiction')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_per_step.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGraphiques sauvegardés dans {save_dir}")


def main():
    # Chargement de la configuration
    with open('checkpoints/config.json', 'r') as f:
        config = json.load(f)
    
    device = torch.device(config['device'])
    print(f"Utilisation de: {device}")
    
    # Chargement des statistiques de normalisation
    norm_stats = np.load('checkpoints/normalization_stats.npy', allow_pickle=True).item()
    
    # Chargement des données de test
    print("\nChargement des données de test...")
    with open('data/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    # Variables exogènes
    exog_data = []
    for i in range(16):
        with open(f'data/test_exog_{i}.json', 'r') as f:
            exog_data.append(json.load(f))
    
    # Création du modèle
    print("\nChargement du modèle...")
    # Pour obtenir num_features, on crée temporairement un dataset
    temp_dataset = TimeSeriesDataset(test_data, config['window_size'], 
                                     config['pred_steps'], exog_data)
    num_features = temp_dataset[0][0].shape[1]
    
    model = CNN_DA_RGRU(
        num_features=num_features,
        conv_channels=config['conv_channels'],
        hidden_size=config['hidden_size'],
        output_size=1,
        pred_steps=config['pred_steps'],
        kernel_size=config['kernel_size'],
        pool_size=config['pool_size']
    )
    
    # Chargement des poids
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Modèle chargé (époque {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.6f})")
    
    # Test du modèle
    print("\nTest du modèle...")
    results = test_model(model, test_data, exog_data, config, device, norm_stats)
    
    # Visualisation
    print("\nGénération des graphiques...")
    plot_predictions(results, config, save_dir='results')
    
    # Sauvegarde des résultats
    results_to_save = {
        'metrics_per_step': results['metrics_per_step'],
        'global_metrics': results['global_metrics'],
        'predictions': results['predictions'].tolist(),
        'targets': results['targets'].tolist()
    }
    
    with open('results/test_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=4)
    
    print("\nTest terminé!")


if __name__ == "__main__":
    main()