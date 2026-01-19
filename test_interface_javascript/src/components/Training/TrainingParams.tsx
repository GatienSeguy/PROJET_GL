import React from 'react';
import { Settings } from 'lucide-react';
import { useStore } from '../../store/useStore';

export const TrainingParams: React.FC = () => {
  const { config, updateConfig } = useStore();

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white flex items-center">
        <Settings size={20} className="mr-2" />
        Paramètres d'Entraînement
      </h3>

      <div className="space-y-4">
        {/* Nombre d'époques */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Nombre d'époques
          </label>
          <input
            type="number"
            value={config.Parametres_entrainement.nb_epochs}
            onChange={(e) =>
              updateConfig({
                Parametres_entrainement: {
                  ...config.Parametres_entrainement,
                  nb_epochs: parseInt(e.target.value) || 1,
                },
              })
            }
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Batch Size */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Batch Size
          </label>
          <input
            type="number"
            value={config.Parametres_entrainement.batch_size}
            onChange={(e) =>
              updateConfig({
                Parametres_entrainement: {
                  ...config.Parametres_entrainement,
                  batch_size: parseInt(e.target.value) || 1,
                },
              })
            }
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Learning Rate */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Learning Rate
          </label>
          <input
            type="number"
            step="0.0001"
            value={config.Parametres_optimisateur.learning_rate}
            onChange={(e) =>
              updateConfig({
                Parametres_optimisateur: {
                  ...config.Parametres_optimisateur,
                  learning_rate: parseFloat(e.target.value) || 0.001,
                },
              })
            }
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Optimiseur */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Optimiseur
          </label>
          <select
            value={config.Parametres_optimisateur.optimisateur}
            onChange={(e) =>
              updateConfig({
                Parametres_optimisateur: {
                  ...config.Parametres_optimisateur,
                  optimisateur: e.target.value,
                },
              })
            }
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          >
            <option value="Adam">Adam</option>
            <option value="SGD">SGD</option>
            <option value="RMSprop">RMSprop</option>
            <option value="Adagrad">Adagrad</option>
            <option value="Adadelta">Adadelta</option>
          </select>
        </div>

        {/* Decroissance */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Décroissance (Weight Decay)
          </label>
          <input
            type="number"
            step="0.0001"
            min="0"
            value={config.Parametres_optimisateur.decroissance}
            onChange={(e) =>
              updateConfig({
                Parametres_optimisateur: {
                  ...config.Parametres_optimisateur,
                  decroissance: parseFloat(e.target.value) || 0,
                },
              })
            }
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Scheduler */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Scheduler
          </label>
          <select
            value={config.Parametres_optimisateur.scheduler}
            onChange={(e) =>
              updateConfig({
                Parametres_optimisateur: {
                  ...config.Parametres_optimisateur,
                  scheduler: e.target.value,
                },
              })
            }
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          >
            <option value="None">None</option>
            <option value="Plateau">Plateau</option>
            <option value="Cosine">Cosine</option>
            <option value="OneCycle">OneCycle</option>
            <option value="StepLR">StepLR</option>
            <option value="ExponentialLR">ExponentialLR</option>
          </select>
        </div>

        {/* Patience */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Patience (epochs)
          </label>
          <input
            type="number"
            min="1"
            value={config.Parametres_optimisateur.patience}
            onChange={(e) =>
              updateConfig({
                Parametres_optimisateur: {
                  ...config.Parametres_optimisateur,
                  patience: parseInt(e.target.value) || 5,
                },
              })
            }
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Nombre d'époques sans amélioration avant early stopping
          </p>
        </div>

        {/* Fonction de perte */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Fonction de perte
          </label>
          <select
            value={config.Parametres_choix_loss_fct.fonction_perte}
            onChange={(e) =>
              updateConfig({
                Parametres_choix_loss_fct: {
                  ...config.Parametres_choix_loss_fct,
                  fonction_perte: e.target.value,
                },
              })
            }
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          >
            <option value="MSE">MSE (Mean Squared Error)</option>
            <option value="MAE">MAE (Mean Absolute Error)</option>
            <option value="Huber">Huber Loss</option>
          </select>
        </div>
      </div>
    </div>
  );
};
