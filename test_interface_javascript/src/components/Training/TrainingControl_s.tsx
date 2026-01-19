import React from 'react';
import { Play, StopCircle } from 'lucide-react';
import { Button } from '../Common/Button';
import { Card } from '../Common/Card';
import { useStore } from '../../store/useStore';
import { trainingAPI, datasetAPI } from '../../services/api';

export const TrainingControl: React.FC = () => {
  const {
    config,
    modelConfig,
    isTraining,
    startTraining,
    stopTraining,
    addTrainingPoint,
    setTestingData,
    setMetrics,
  } = useStore();

  const handleStart = async () => {
    if (!config.Parametres_temporels.nom_dataset) {
      alert('❌ Erreur: Aucun dataset sélectionné !\n\nVeuillez sélectionner un dataset avant de lancer l\'entraînement.');
      return;
    }

    startTraining();

    const valPairs: any[] = [];
    const predPoints: any[] = [];
    let valStart = 0;
    let predStart = 0;
    let serieComplete: number[] = [];
    let valMetrics: any = null;
    let predMetrics: any = null;

    try {
      const datasetPayload = {
        name: config.Parametres_temporels.nom_dataset,
        dates: config.Parametres_temporels.dates,
        pas_temporel: config.Parametres_temporels.pas_temporel,
      };
      
      await datasetAPI.fetchDataset(datasetPayload);

      await trainingAPI.startTraining(
        config,
        modelConfig,
        (event) => {
          if (event.type === 'error') {
            console.error('Erreur serveur:', event.message);
            alert(`❌ Erreur du serveur:\n\n${event.message}`);
            stopTraining();
            return;
          }

          if (event.type === 'epoch') {
            addTrainingPoint(event.epoch || event.epochs, event.avg_loss);
          }
          else if (event.type === 'val_start') {
            valPairs.length = 0;
            valStart = event.idx_start || 0;
          }
          else if (event.type === 'val_pair') {
            valPairs.push(event);
          }
          else if (event.type === 'pred_start') {
            predPoints.length = 0;
            predStart = event.idx_start || 0;
          }
          else if (event.type === 'pred_point') {
            predPoints.push(event);
          }
          else if (event.type === 'val_end') {
            const metricsData = event.metrics || event.validation_metrics || event.val_metrics;
            console.log('📗 Métriques VALIDATION reçues:', metricsData);
            if (metricsData) {
              valMetrics = metricsData;
              const currentMetrics = useStore.getState().metrics || {};
              setMetrics({
                ...currentMetrics,
                validation: metricsData.overall_mean ? metricsData : { overall_mean: metricsData }
              });
            }
          }
          else if (event.type === 'pred_end') {
            const metricsData = event.metrics || event.prediction_metrics || event.pred_metrics;
            console.log('📕 Métriques PRÉDICTION reçues:', metricsData);
            if (metricsData) {
              predMetrics = metricsData;
              const currentMetrics = useStore.getState().metrics || {};
              setMetrics({
                ...currentMetrics,
                prediction: metricsData.overall_mean ? metricsData : { overall_mean: metricsData }
              });
            }
          }
          else if (event.type === 'serie_complete') {
            serieComplete = event.serie || event.values || event.data || [];
          }
          else if (event.type === 'final_plot_data') {
            // Gérer les métriques envoyées avec final_plot_data
            if (event.val_metrics) {
              valMetrics = event.val_metrics;
            }
            if (event.pred_metrics) {
              predMetrics = event.pred_metrics;
            }
            
            // Stocker les métriques combinées
            if (valMetrics || predMetrics) {
              const combinedMetrics: any = {};
              if (valMetrics) combinedMetrics.validation = valMetrics;
              if (predMetrics) combinedMetrics.prediction = predMetrics;
              console.log('📊 Métriques FINALES:', combinedMetrics);
              setMetrics(combinedMetrics);
            }
          }
          else if (event.type === 'fin_pipeline') {
            if (valPairs.length > 0 && predPoints.length > 0) {
              let series_complete = serieComplete;
              let idx_val_start_final = valStart;
              let idx_test_start_final = predStart;
              
              if (!series_complete || series_complete.length === 0) {
                series_complete = [
                  ...valPairs.map(v => v.y[0]),
                  ...predPoints.map(p => p.y)
                ];
                idx_val_start_final = 0;
                idx_test_start_final = valPairs.length;
              }
              
              const testingData = {
                type: 'final_plot_data' as const,
                series_complete,
                val_predictions: valPairs.map(v => v.yhat[0]),
                pred_predictions: predPoints.map(p => p.yhat),
                pred_low: predPoints.map(p => p.low),
                pred_high: predPoints.map(p => p.high),
                idx_val_start: idx_val_start_final,
                idx_test_start: idx_test_start_final,
              };
              
              setTestingData(testingData);
            }
          }
        },
        (error) => {
          console.error('Erreur d\'entraînement:', error);
          stopTraining();
          alert(`Erreur lors de l'entraînement:\n\n${error.message || error}`);
        },
        () => {
          stopTraining();
        }
      );
    } catch (error: any) {
      console.error('Erreur:', error);
      stopTraining();
      alert(`Erreur inattendue:\n\n${error.message || error}`);
    }
  };

  const handleStop = async () => {
    await trainingAPI.stopTraining();
    stopTraining();
  };

  return (
    <Card>
      <div className="flex gap-3">
        {!isTraining ? (
          <Button onClick={handleStart} variant="success" icon={<Play size={18} />} className="flex-1">
            🚀 Lancer l'entraînement
          </Button>
        ) : (
          <Button onClick={handleStop} variant="danger" icon={<StopCircle size={18} />} className="flex-1">
            🛑 Arrêter
          </Button>
        )}
      </div>
    </Card>
  );
};
