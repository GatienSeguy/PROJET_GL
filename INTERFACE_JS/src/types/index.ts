export interface TimeSeriesData {
  timestamps: string[];
  values: number[];
}

export interface DatasetPacket {
  payload_name: string;
  payload_dataset: TimeSeriesData;
}

export interface DatasetInfo {
  [key: string]: {
    nom?: string;
    dates: [string, string];
    pas_temporel: number | string; // Peut être "1j" ou 1
  };
}

export interface GlobalConfig {
  Parametres_temporels: {
    nom_dataset: string;
    horizon: number;
    dates: [string, string];
    pas_temporel: number;
    portion_decoupage: number;
  };
  Parametres_choix_reseau_neurones: {
    modele: string;
  };
  Parametres_choix_loss_fct: {
    fonction_perte: string;
  };
  Parametres_training: {
    nb_epoques: number;
    batch_size: number;
    learning_rate: number;
    optimiseur: string;
    decroissance: number;
    scheduler: string;
    patience: number;
  };
}

export interface ModelConfig {
  nb_couches?: number;
  hidden_size?: number;
  dropout_rate?: number;
  fonction_activation?: string;
  kernel_size?: number;
  stride?: number;
  padding?: number;
  bidirectional?: boolean;
  batch_first?: boolean;
}

export interface TrainingMetrics {
  epoch: number;
  train_loss: number;
  val_loss?: number;
  lr?: number;
}

export interface TestingResults {
  series_complete: number[];
  val_predictions: number[];
  pred_predictions: number[];
  pred_low: number[];
  pred_high: number[];
  idx_val_start: number;
  idx_test_start: number;
}
