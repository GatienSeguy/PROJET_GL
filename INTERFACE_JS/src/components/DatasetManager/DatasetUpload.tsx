import React, { useRef, useState } from 'react';
import { Upload, X } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { apiClient } from '../../services/api';

export const DatasetUpload: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { setDatasets } = useStore();

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      // Extraire le nom sans l'extension
      const nameWithoutExt = file.name.replace(/\.json$/i, '');
      setDatasetName(nameWithoutExt);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !datasetName.trim()) {
      setError('Veuillez sélectionner un fichier et entrer un nom');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      // Lire le fichier JSON
      const fileContent = await selectedFile.text();
      const data = JSON.parse(fileContent);

      // Valider la structure
      if (!data.timestamps || !data.values) {
        throw new Error('Structure invalide : timestamps et values requis');
      }

      if (!Array.isArray(data.timestamps) || !Array.isArray(data.values)) {
        throw new Error('timestamps et values doivent être des tableaux');
      }

      if (data.timestamps.length !== data.values.length) {
        throw new Error('timestamps et values doivent avoir la même longueur');
      }

      // Créer le payload
      const payload = {
        payload_name: datasetName.trim(),
        payload_dataset_add: {
          timestamps: data.timestamps,
          values: data.values,
        },
      };

      console.log('📤 Envoi du dataset:', payload);

      // Envoyer au serveur
      const response = await apiClient.addDataset(payload);

      if (response.ok) {
        console.log('✅ Dataset ajouté:', response.stored);

        // Recharger la liste des datasets
        const datasets = await apiClient.getAllDatasets();
        setDatasets(datasets);

        // Fermer la modal
        onClose();
      } else {
        throw new Error('Échec de l\'ajout du dataset');
      }
    } catch (err: any) {
      console.error('❌ Erreur:', err);
      if (err.message.includes('JSON')) {
        setError('Fichier JSON invalide');
      } else if (err.message.includes('Structure')) {
        setError(err.message);
      } else if (err.message.includes('existe déjà')) {
        setError('Un dataset avec ce nom existe déjà');
      } else {
        setError('Erreur lors de l\'ajout du dataset');
      }
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-md w-full space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white">Ajouter un Dataset</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-slate-700 rounded transition-colors"
          >
            <X size={20} className="text-gray-400" />
          </button>
        </div>

        <div className="space-y-4">
          {/* Sélection du fichier */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Fichier JSON
            </label>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleFileSelect}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full px-4 py-3 bg-slate-700 hover:bg-slate-600 border border-slate-600 rounded-lg text-white transition-colors flex items-center justify-center space-x-2"
            >
              <Upload size={20} />
              <span>{selectedFile ? selectedFile.name : 'Sélectionner un fichier'}</span>
            </button>
            <p className="text-xs text-gray-500 mt-1">
              Format attendu : {`{"timestamps": [...], "values": [...]}`}
            </p>
          </div>

          {/* Nom du dataset */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Nom du dataset
            </label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="mon_dataset"
              className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Erreur */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-3">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          {/* Boutons */}
          <div className="flex space-x-3">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
            >
              Annuler
            </button>
            <button
              onClick={handleUpload}
              disabled={!selectedFile || !datasetName.trim() || isUploading}
              className="flex-1 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isUploading ? 'Ajout en cours...' : 'Ajouter'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
