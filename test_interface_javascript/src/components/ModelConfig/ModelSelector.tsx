import React from 'react';
import { Wand2 } from 'lucide-react';
import { useStore } from '../../store/useStore';

export const ModelSelector: React.FC = () => {
  const { config, modelConfig, updateConfig, updateModelConfig } = useStore();
  const modelType = config.Parametres_choix_reseau_neurones.modele;

  const handleModelTypeChange = (type: string) => {
    updateConfig({
      Parametres_choix_reseau_neurones: {
        modele: type,
      },
    });
  };

  const handleArchParamChange = (param: string, value: any) => {
    updateModelConfig({
      Parametres_archi_reseau: {
        ...modelConfig.Parametres_archi_reseau,
        [param]: value,
      },
    });
  };

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-white flex items-center">
        <Wand2 size={20} className="mr-2" />
        Configuration du Modèle
      </h3>

      {/* Type de modèle */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-3">
          Type de modèle
        </label>
        <div className="grid grid-cols-3 gap-2">
          {['MLP', 'LSTM', 'CNN'].map((type) => (
            <button
              key={type}
              onClick={() => handleModelTypeChange(type)}
              className={`px-4 py-3 rounded-lg font-medium transition-all ${
                modelType === type
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-slate-800 text-gray-400 hover:bg-slate-700 border border-slate-700'
              }`}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Paramètres communs */}
      <div className="space-y-4">
        {/* Nombre de couches */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Nombre de couches
          </label>
          <input
            type="number"
            min="1"
            max="10"
            value={modelConfig.Parametres_archi_reseau.nb_couches}
            onChange={(e) => handleArchParamChange('nb_couches', parseInt(e.target.value) || 1)}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Hidden Size */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Hidden Size
          </label>
          <input
            type="number"
            min="1"
            max="512"
            value={modelConfig.Parametres_archi_reseau.hidden_size}
            onChange={(e) => handleArchParamChange('hidden_size', parseInt(e.target.value) || 64)}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Paramètres MLP */}
        {modelType === 'MLP' && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Dropout Rate
              </label>
              <input
                type="number"
                min="0"
                max="0.9"
                step="0.1"
                value={modelConfig.Parametres_archi_reseau.dropout_rate}
                onChange={(e) => handleArchParamChange('dropout_rate', parseFloat(e.target.value) || 0)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Fonction d'activation
              </label>
              <select
                value={modelConfig.Parametres_archi_reseau.fonction_activation}
                onChange={(e) => handleArchParamChange('fonction_activation', e.target.value)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="ReLU">ReLU</option>
                <option value="GELU">GELU</option>
                <option value="tanh">tanh</option>
              </select>
            </div>
          </>
        )}

        {/* Paramètres CNN */}
        {modelType === 'CNN' && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Dropout Rate
              </label>
              <input
                type="number"
                min="0"
                max="0.9"
                step="0.1"
                value={modelConfig.Parametres_archi_reseau.dropout_rate}
                onChange={(e) => handleArchParamChange('dropout_rate', parseFloat(e.target.value) || 0)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Fonction d'activation
              </label>
              <select
                value={modelConfig.Parametres_archi_reseau.fonction_activation}
                onChange={(e) => handleArchParamChange('fonction_activation', e.target.value)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="ReLU">ReLU</option>
                <option value="GELU">GELU</option>
                <option value="tanh">tanh</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Kernel Size
              </label>
              <input
                type="number"
                min="1"
                max="11"
                value={modelConfig.Parametres_archi_reseau.kernel_size || 3}
                onChange={(e) => handleArchParamChange('kernel_size', parseInt(e.target.value) || 3)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Stride
              </label>
              <input
                type="number"
                min="1"
                max="5"
                value={modelConfig.Parametres_archi_reseau.stride || 1}
                onChange={(e) => handleArchParamChange('stride', parseInt(e.target.value) || 1)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Padding
              </label>
              <input
                type="number"
                min="0"
                max="5"
                value={modelConfig.Parametres_archi_reseau.padding || 0}
                onChange={(e) => handleArchParamChange('padding', parseInt(e.target.value) || 0)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              />
            </div>
          </>
        )}

        {/* Paramètres LSTM */}
        {modelType === 'LSTM' && (
          <>
            <div className="flex items-center justify-between p-4 bg-slate-800 border border-slate-700 rounded-lg">
              <div>
                <label className="block text-sm font-medium text-white mb-1">
                  Bidirectionnel
                </label>
                <p className="text-xs text-gray-400">
                  Analyse la séquence dans les deux sens
                </p>
              </div>
              <input
                type="checkbox"
                checked={modelConfig.Parametres_archi_reseau.bidirectional || false}
                onChange={(e) => handleArchParamChange('bidirectional', e.target.checked)}
                className="w-5 h-5 rounded bg-slate-700 border-slate-600 text-blue-500 focus:ring-blue-500"
              />
            </div>

            <div className="flex items-center justify-between p-4 bg-slate-800 border border-slate-700 rounded-lg">
              <div>
                <label className="block text-sm font-medium text-white mb-1">
                  Batch First
                </label>
                <p className="text-xs text-gray-400">
                  Format: (batch, sequence, features)
                </p>
              </div>
              <input
                type="checkbox"
                checked={modelConfig.Parametres_archi_reseau.batch_first || false}
                onChange={(e) => handleArchParamChange('batch_first', e.target.checked)}
                className="w-5 h-5 rounded bg-slate-700 border-slate-600 text-blue-500 focus:ring-blue-500"
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
};
