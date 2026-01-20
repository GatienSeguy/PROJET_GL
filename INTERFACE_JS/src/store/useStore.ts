import { create } from 'zustand';
import type { AppState, GlobalConfig, ModelConfig, DatasetInfo, FinalPlotData, MetricsResponse } from '../types';

const initialConfig: GlobalConfig = {
  Parametres_temporels: {
    nom_dataset: '',
    horizon: 1,
    dates: ['2001-01-01', '2025-01-02'],
    pas_temporel: 1,
    portion_decoupage: 0.8,
  },
  Parametres_choix_reseau_neurones: {
    modele: 'MLP',
  },
  Parametres_choix_loss_fct: {
    fonction_perte: 'MSE',
    params: null,
  },
  Parametres_optimisateur: {
    optimisateur: 'Adam',
    learning_rate: 0.001,
    decroissance: 0.0,
    scheduler: 'None',
    patience: 5,
  },
  Parametres_entrainement: {
    nb_epochs: 1000,
    batch_size: 4,
    clip_gradient: null,
  },
  Parametres_visualisation_suivi: {
    metriques: ['loss'],
  },
};

const initialModelConfig: ModelConfig = {
  Parametres_archi_reseau: {
    nb_couches: 2,
    hidden_size: 64,
    dropout_rate: 0.0,
    fonction_activation: 'ReLU',
  },
};

export const useStore = create<AppState>((set) => ({
  // État initial
  config: initialConfig,
  modelConfig: initialModelConfig,
  datasets: {},
  selectedDataset: null,
  isTraining: false,
  trainingData: [],
  currentEpoch: 0,
  totalEpochs: 1000,
  testingData: null,
  metrics: null,

  // Actions
  updateConfig: (newConfig) =>
    set((state) => {
      const mergedConfig = { ...state.config };
      
      // Deep merge pour chaque clé de premier niveau
      Object.keys(newConfig).forEach((key) => {
        if (typeof newConfig[key] === 'object' && !Array.isArray(newConfig[key])) {
          mergedConfig[key] = { ...state.config[key], ...newConfig[key] };
        } else {
          mergedConfig[key] = newConfig[key];
        }
      });
      
      console.log('🔧 Config mise à jour:', mergedConfig);
      return { config: mergedConfig };
    }),

  updateModelConfig: (newConfig) =>
    set({ modelConfig: newConfig }),

  setDatasets: (datasets: DatasetInfo) =>
    set({ datasets }),

  selectDataset: (name: string) =>
    set({ selectedDataset: name }),

  startTraining: () =>
    set((state) => ({
      isTraining: true,
      trainingData: [],
      currentEpoch: 0,
      totalEpochs: state.config.Parametres_entrainement.nb_epochs,
      testingData: null,
      metrics: null,
    })),

  stopTraining: () =>
    set({ isTraining: false }),

  addTrainingPoint: (epoch: number, loss: number) =>
    set((state) => ({
      trainingData: [...state.trainingData, { epoch, loss }],
      currentEpoch: epoch,
    })),

  setTestingData: (data: FinalPlotData) =>
    set({ testingData: data }),

  setMetrics: (metrics: MetricsResponse) =>
    set({ metrics }),

  reset: () =>
    set({
      config: initialConfig,
      modelConfig: initialModelConfig,
      isTraining: false,
      trainingData: [],
      currentEpoch: 0,
      testingData: null,
      metrics: null,
    }),
}));
