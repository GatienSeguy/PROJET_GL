import React, { useState, useEffect } from 'react';
import { FolderOpen, Save, Trash2, RefreshCw, Download, Check } from 'lucide-react';
import { modelAPI } from '../../services/api';

interface ModelInfo {
  name: string;
  model_type: string;
  dataset: string;
  window_size: number;
  created_at: string;
}

export const ModelManager: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string>('');
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [saveName, setSaveName] = useState('');

  useEffect(() => {
    loadModelsList();
  }, []);

  const loadModelsList = async () => {
    setLoading(true);
    setStatus('Chargement...');
    try {
      const response = await modelAPI.listModels();
      setModels(response.models || []);
      setStatus(`${response.models?.length || 0} modèle(s) trouvé(s)`);
    } catch (error: any) {
      setStatus(`Erreur: ${error.message}`);
      setModels([]);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadModel = async () => {
    if (!selectedModel) {
      alert('Sélectionnez un modèle');
      return;
    }
    
    setLoading(true);
    setStatus('Chargement du modèle...');
    try {
      const response = await modelAPI.loadModel(selectedModel);
      setStatus(`Modèle "${selectedModel}" chargé!`);
      alert(`Modèle "${selectedModel}" chargé avec succès!\n\nType: ${response.model_type}\nFenêtre: ${response.window_size} points`);
    } catch (error: any) {
      setStatus(`Erreur: ${error.message}`);
      alert(`Erreur: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveModel = async () => {
    if (!saveName.trim()) {
      alert('Entrez un nom pour le modèle');
      return;
    }
    
    setLoading(true);
    setStatus('Sauvegarde...');
    try {
      await modelAPI.saveModel(saveName.trim());
      setStatus(`Modèle "${saveName}" sauvegardé!`);
      setSaveModalOpen(false);
      setSaveName('');
      loadModelsList();
    } catch (error: any) {
      setStatus(`Erreur: ${error.message}`);
      alert(`Erreur: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteModel = async () => {
    if (!selectedModel) {
      alert('Sélectionnez un modèle');
      return;
    }
    
    if (!confirm(`Supprimer le modèle "${selectedModel}" ?`)) {
      return;
    }
    
    setLoading(true);
    setStatus('Suppression...');
    try {
      await modelAPI.deleteModel(selectedModel);
      setStatus(`Modèle "${selectedModel}" supprimé`);
      setSelectedModel(null);
      loadModelsList();
    } catch (error: any) {
      setStatus(`Erreur: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white flex items-center">
        <FolderOpen size={20} className="mr-2" />
        Gestion des Modèles
      </h3>

      <div className="space-y-4">
        {/* Liste des modèles */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-gray-300">
              Modèles sauvegardés
            </label>
            <button
              onClick={loadModelsList}
              disabled={loading}
              className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors"
            >
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
              <span>Rafraîchir</span>
            </button>
          </div>
          
          <div className="bg-slate-800 border border-slate-700 rounded-lg max-h-48 overflow-y-auto">
            {models.length === 0 ? (
              <div className="p-4 text-center text-gray-500 text-sm">
                Aucun modèle sauvegardé
              </div>
            ) : (
              models.map((model) => (
                <div
                  key={model.name}
                  onClick={() => setSelectedModel(model.name)}
                  className={`p-3 border-b border-slate-700 last:border-b-0 cursor-pointer transition-colors ${
                    selectedModel === model.name
                      ? 'bg-blue-500/20 border-l-2 border-l-blue-500'
                      : 'hover:bg-slate-700/50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-white font-medium text-sm">{model.name}</span>
                    {selectedModel === model.name && (
                      <Check size={14} className="text-blue-400" />
                    )}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    {model.model_type?.toUpperCase()} • w={model.window_size}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Boutons d'action */}
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={handleLoadModel}
            disabled={loading || !selectedModel}
            className="flex items-center justify-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-700 disabled:text-gray-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            <Download size={14} />
            Charger
          </button>
          
          <button
            onClick={handleDeleteModel}
            disabled={loading || !selectedModel}
            className="flex items-center justify-center gap-2 px-3 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-700 disabled:text-gray-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            <Trash2 size={14} />
            Supprimer
          </button>
        </div>

        {/* Bouton sauvegarder */}
        <button
          onClick={() => setSaveModalOpen(true)}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 text-white rounded-lg text-sm font-medium transition-colors"
        >
          <Save size={14} />
          Sauvegarder le modèle actuel
        </button>

        {/* Status */}
        {status && (
          <div className="text-xs text-gray-400 text-center">
            {status}
          </div>
        )}
      </div>

      {/* Modal de sauvegarde */}
      {saveModalOpen && (
        <>
          <div 
            className="fixed inset-0 z-40 bg-black/50"
            onClick={() => setSaveModalOpen(false)}
          />
          <div className="fixed z-50 bg-slate-800 border border-slate-600 rounded-xl shadow-2xl p-4 w-80"
               style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
            <h4 className="text-white font-semibold mb-3">Sauvegarder le modèle</h4>
            <input
              type="text"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="Nom du modèle"
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm mb-3 focus:outline-none focus:border-blue-500"
            />
            <div className="flex gap-2">
              <button
                onClick={handleSaveModel}
                disabled={loading}
                className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium"
              >
                Sauvegarder
              </button>
              <button
                onClick={() => setSaveModalOpen(false)}
                className="flex-1 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm font-medium"
              >
                Annuler
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
