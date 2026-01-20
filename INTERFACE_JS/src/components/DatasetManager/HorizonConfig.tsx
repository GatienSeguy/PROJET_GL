import React from 'react';
import { Card } from '../Common/Card';
import { useStore } from '../../store/useStore';

export const HorizonConfig: React.FC = () => {
  const { config, updateConfig } = useStore();

  const handleHorizonChange = (value: number) => {
    updateConfig({
      Parametres_temporels: {
        ...config.Parametres_temporels,
        horizon: Math.max(1, value),
      },
    });
  };

  const handleSplitChange = (value: number) => {
    updateConfig({
      Parametres_temporels: {
        ...config.Parametres_temporels,
        portion_decoupage: value / 100,
      },
    });
  };

  return (
    <Card title="Paramètres Temporels">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Horizon temporel
          </label>
          <input
            type="number"
            min="1"
            value={config.Parametres_temporels.horizon}
            onChange={(e) => handleHorizonChange(parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-dark-input text-gray-900 dark:text-white"
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Nombre de pas de temps pour la prédiction
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Train/Test Split: {(config.Parametres_temporels.portion_decoupage * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min="50"
            max="95"
            step="5"
            value={config.Parametres_temporels.portion_decoupage * 100}
            onChange={(e) => handleSplitChange(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
            <span>50%</span>
            <span>95%</span>
          </div>
        </div>
      </div>
    </Card>
  );
};
