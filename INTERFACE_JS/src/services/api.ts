import type {
  DatasetInfo,
  GlobalConfig,
  ModelConfig,
  TimeSeriesData,
} from '../types';

// L'API_URL est vide car on utilise le proxy Vite
// Toutes les requêtes /datasets, /model, /predict, /train_full sont proxifiées vers le serveur FastAPI
const API_URL = import.meta.env.VITE_API_URL || '';

console.log(' API_URL configurée:', API_URL || '(proxy Vite)');

// Fonction helper pour faire des requêtes POST
async function simplePost(url: string, data: any) {
  const response = await fetch(API_URL + url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
  }
  
  return response.json();
}

// ====================================
// DATASET API
// ====================================
export const datasetAPI = {
  async getAllDatasets(): Promise<DatasetInfo> {
    console.log(' Appel getAllDatasets');
    return simplePost('/datasets/info_all', {
      message: 'choix dataset',
    });
  },

  async fetchDataset(payload: {
    name: string;
    dates: [string, string];
    pas_temporel: number;
  }): Promise<{ status: string; data: TimeSeriesData }> {
    console.log(' API fetchDataset - Payload envoyé:', JSON.stringify(payload, null, 2));
    const result = await simplePost('/datasets/fetch_dataset', payload);
    console.log(' API fetchDataset - Réponse reçue');
    return result;
  },

  async addDataset(payload: {
    payload_name: string;
    payload_dataset_add: TimeSeriesData;
  }): Promise<{ ok: boolean; stored: string }> {
    return simplePost('/datasets/add_dataset', payload);
  },

  async deleteDataset(name: string): Promise<string> {
    return simplePost('/datasets/data_suppression_proxy', { name });
  },
};

// ====================================
// TRAINING API
// ====================================
export const trainingAPI = {
  async startTraining(
    config: GlobalConfig,
    modelConfig: ModelConfig,
    onEvent: (event: any) => void,
    onError: (error: any) => void,
    onComplete: () => void
  ): Promise<void> {
    console.log(' Tentative de connexion à:', `${API_URL}/train_full`);

    try {
      const response = await fetch(`${API_URL}/train_full`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          payload: config,
          payload_model: modelConfig,
        }),
      });

      console.log('📡 Réponse reçue, status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Erreur HTTP:', response.status, errorText);
        throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No reader available');
      }

      console.log('✅ Streaming démarré');

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log('✅ Streaming terminé');
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const jsonStr = line.slice(6).trim();
              if (jsonStr) {
                const data = JSON.parse(jsonStr);
                onEvent(data);
              }
            } catch (e) {
              console.error('Error parsing event data:', e);
            }
          }
        }
      }

      if (buffer.trim().startsWith('data: ')) {
        try {
          const jsonStr = buffer.slice(6).trim();
          if (jsonStr) {
            const data = JSON.parse(jsonStr);
            onEvent(data);
          }
        } catch (e) {
          console.error('Error parsing final buffer:', e);
        }
      }

      onComplete();
    } catch (error) {
      console.error('❌ Erreur dans startTraining:', error);
      onError(error);
    }
  },

  async stopTraining(): Promise<void> {
    try {
      await fetch(`${API_URL}/stop_training`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    } catch (error) {
      console.error('Erreur lors de l\'arrêt:', error);
    }
  },
};

// ====================================
// MODEL API
// ====================================
export const modelAPI = {
  listModels: async (): Promise<{ models: any[] }> => {
    console.log(' [modelAPI] listModels');
    const response = await fetch(`${API_URL}/model/list`);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || 'Erreur lors du chargement des modèles');
    }
    return response.json();
  },

  loadModel: async (name: string): Promise<any> => {
    console.log(' [modelAPI] loadModel:', name);
    const response = await fetch(`${API_URL}/model/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || 'Erreur lors du chargement du modèle');
    }
    return response.json();
  },

  saveModel: async (name: string): Promise<any> => {
    console.log(' [modelAPI] saveModel:', name);
    const response = await fetch(`${API_URL}/model/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || 'Erreur lors de la sauvegarde du modèle');
    }
    return response.json();
  },

  deleteModel: async (name: string): Promise<any> => {
    console.log(' [modelAPI] deleteModel:', name);
    const response = await fetch(`${API_URL}/model/delete/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || 'Erreur lors de la suppression du modèle');
    }
    return response.json();
  },
};

// ====================================
// PREDICTION API
// ====================================
export interface PredictionResult {
  series_complete: number[];
  predictions: number[];
  pred_low: number[];
  pred_high: number[];
}

export const predictionAPI = {
  predict: async (
    horizon: number,
    confidenceLevel: number = 0.95
  ): Promise<PredictionResult> => {
    console.log(' [predictionAPI] predict horizon:', horizon);

    return new Promise((resolve, reject) => {
      const predictions: number[] = [];
      const pred_low: number[] = [];
      const pred_high: number[] = [];
      let series_complete: number[] = [];

      fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ horizon, confidence_level: confidenceLevel }),
      })
        .then(async (response) => {
          if (!response.ok) {
            const text = await response.text();
            throw new Error(text || 'Erreur lors de la prédiction');
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('Streaming non supporté');
          }

          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (!line.startsWith('data: ')) continue;

              try {
                const msg = JSON.parse(line.slice(6));

                if (msg.type === 'error') {
                  reject(new Error(msg.message));
                  return;
                }

                if (msg.type === 'pred_point') {
                  predictions.push(msg.yhat);
                  pred_low.push(msg.low);
                  pred_high.push(msg.high);
                }

                if (msg.type === 'pred_end') {
                  series_complete = msg.series_complete || [];
                }

                if (msg.type === 'fin_prediction') {
                  resolve({ series_complete, predictions, pred_low, pred_high });
                  return;
                }
              } catch (e) {
                // Ignore JSON parse errors
              }
            }
          }

          // Résolution si fin_prediction n'est pas reçu explicitement
          if (predictions.length > 0) {
            resolve({ series_complete, predictions, pred_low, pred_high });
          } else {
            reject(new Error('Aucune prédiction reçue'));
          }
        })
        .catch(reject);
    });
  },
};

// ====================================
// CLIENT COMBINÉ
// ====================================
export const apiClient = {
  ...datasetAPI,
  ...trainingAPI,
};
