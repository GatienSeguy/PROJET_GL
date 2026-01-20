import React, { useEffect, useState } from 'react';
import { Trash2, Database, Calendar, TrendingUp } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { apiClient } from '../../services/api';
import { DatasetUpload } from './DatasetUpload';

export const DatasetList: React.FC = () => {
  const { datasets, selectedDataset, selectDataset, setDatasets, updateConfig } = useStore();
  const [showUpload, setShowUpload] = useState(false);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const data = await apiClient.getAllDatasets();
      console.log('📊 Datasets chargés:', data);
      setDatasets(data);
    } catch (error) {
      console.error('Erreur chargement datasets:', error);
    }
  };

  const handleSelectDataset = (name: string) => {
    console.log('📊 Dataset sélectionné:', name);
    selectDataset(name);
    
    const datasetInfo = datasets[name];
    console.log('📊 Info du dataset:', datasetInfo);
    
    // Parser pas_temporel : "1j" -> 1, "7j" -> 7, etc.
    let pasTemporel = datasetInfo.pas_temporel;
    if (typeof pasTemporel === 'string') {
      // Extraire le nombre au début de la chaîne
      const match = pasTemporel.match(/^(\d+)/);
      pasTemporel = match ? parseInt(match[1], 10) : 1;
    }
    
    const newConfig = {
      Parametres_temporels: {
        nom_dataset: name,
        horizon: 1,
        dates: datasetInfo.dates,
        pas_temporel: pasTemporel,
        portion_decoupage: 0.8,
      },
    };
    
    console.log('📊 Nouvelle config:', newConfig);
    updateConfig(newConfig);
  };

  const handleDelete = async (name: string) => {
    if (window.confirm(`Supprimer le dataset "${name}" ?`)) {
      try {
        await apiClient.deleteDataset(name);
        await loadDatasets();
        if (selectedDataset === name) {
          selectDataset('');
        }
      } catch (error) {
        console.error('Erreur suppression:', error);
      }
    }
  };

  const datasetList = Object.entries(datasets);

  return (
    <>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white flex items-center">
            <Database size={20} className="mr-2" />
            Gestion des Datasets
          </h3>
          <button
            onClick={() => setShowUpload(true)}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm transition-colors flex items-center space-x-2"
          >
            <span>➕</span>
            <span>Ajouter</span>
          </button>
        </div>

        <div className="space-y-2">
          {datasetList.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <p>Aucun dataset disponible</p>
            </div>
          ) : (
            datasetList.map(([name, info]) => {
              const displayName = info.nom || name;
              const [dateDebut, dateFin] = info.dates || ['?', '?'];
              
              return (
                <div
                  key={name}
                  onClick={() => handleSelectDataset(name)}
                  className={`p-4 rounded-lg cursor-pointer transition-all border ${
                    selectedDataset === name
                      ? 'bg-blue-500/20 border-blue-500'
                      : 'bg-slate-800 border-slate-700 hover:bg-slate-700'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 space-y-2">
                      {/* Nom du dataset */}
                      <div className="flex items-center space-x-2">
                        <TrendingUp size={16} className="text-blue-400" />
                        <p className="font-semibold text-white">{displayName}</p>
                      </div>

                      {/* Dates */}
                      {dateDebut && dateFin && dateDebut !== '?' ? (
                        <div className="flex items-center space-x-2 text-xs">
                          <Calendar size={14} className="text-gray-400" />
                          <span className="text-gray-300">{dateDebut}</span>
                          <span className="text-gray-500">→</span>
                          <span className="text-gray-300">{dateFin}</span>
                        </div>
                      ) : (
                        <div className="text-xs text-gray-500 italic">
                          Dates non disponibles
                        </div>
                      )}

                      {/* Pas temporel */}
                      <div className="text-xs text-gray-400">
                        Pas: <span className="text-gray-300">{info.pas_temporel || '?'}</span>
                      </div>
                    </div>

                    {/* Bouton supprimer */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(name);
                      }}
                      className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
                    >
                      <Trash2 size={16} className="text-red-400" />
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Modal d'upload */}
      {showUpload && (
        <DatasetUpload
          onClose={() => {
            setShowUpload(false);
            loadDatasets();
          }}
        />
      )}
    </>
  );
};
