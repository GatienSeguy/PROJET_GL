import type {
  DatasetInfo,
  GlobalConfig,
  ModelConfig,
  TimeSeriesData,
} from '../types';

const API_URL = import.meta.env.VITE_API_URL || '';

console.log('🌐 API_URL configurée:', API_URL);

// Fonction helper pour faire des requêtes POST sans preflight OPTIONS
async function simplePost(url: string, data: any) {
  const response = await fetch(API_URL + url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return response.json();
}

export const datasetAPI = {
  // Récupérer tous les datasets
  async getAllDatasets(): Promise<DatasetInfo> {
    console.log('🌐 Appel getAllDatasets');
    return simplePost('/datasets/info_all', {
      message: 'choix dataset',
    });
  },

  // Récupérer un dataset spécifique (via le serveur IA qui appelle le serveur Data)
  async fetchDataset(payload: {
    name: string;
    dates: [string, string];
    pas_temporel: number;
  }): Promise<{ status: string; data: TimeSeriesData }> {
    console.log('🌐 API fetchDataset - Payload envoyé:', JSON.stringify(payload, null, 2));
    const result = await simplePost('/datasets/fetch_dataset', payload);
    console.log('🌐 API fetchDataset - Réponse reçue');
    return result;
  },

  // Ajouter un dataset
  async addDataset(payload: {
    payload_name: string;
    payload_dataset_add: TimeSeriesData;
  }): Promise<{ ok: boolean; stored: string }> {
    return simplePost('/datasets/add_dataset', payload);
  },

  // Supprimer un dataset
  async deleteDataset(name: string): Promise<string> {
    return simplePost('/datasets/data_suppression_proxy', { name });
  },
};

export const trainingAPI = {
  // Démarrer l'entraînement (streaming)
  async startTraining(
    config: GlobalConfig,
    modelConfig: ModelConfig,
    onEvent: (event: any) => void,
    onError: (error: any) => void,
    onComplete: () => void
  ): Promise<void> {
    console.log('🌐 Tentative de connexion à:', `${API_URL}/train_full`);
    console.log('📦 Payload:', { config, modelConfig });

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

      let buffer = ''; // Buffer pour les chunks fragmentés

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log('✅ Streaming terminé');
          break;
        }

        // Ajouter le nouveau chunk au buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Traiter toutes les lignes complètes
        const lines = buffer.split('\n');
        
        // Garder la dernière ligne incomplète dans le buffer
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
              console.error('Problematic line:', line.substring(0, 200));
            }
          }
        }
      }

      // Traiter le buffer restant s'il y en a
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

  // Arrêter l'entraînement
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

export const apiClient = {
  ...datasetAPI,
  ...trainingAPI,
};
