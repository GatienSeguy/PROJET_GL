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

    // Variables pour accumuler les données
    const valPairs: any[] = [];
    const predPoints: any[] = [];
    let valStart = 0;
    let predStart = 0;
    let serieComplete: number[] = [];
    let valMetrics: any = null;
    let predMetrics: any = null;

    try {
      // 1. Charger le dataset
      const datasetPayload = {
        name: config.Parametres_temporels.nom_dataset,
        dates: config.Parametres_temporels.dates,
        pas_temporel: config.Parametres_temporels.pas_temporel,
      };
      
      console.log('📊 Chargement du dataset:', datasetPayload);
      await datasetAPI.fetchDataset(datasetPayload);
      console.log('✅ Dataset chargé avec succès');

      // 2. Lancer l'entraînement
      await trainingAPI.startTraining(
        config,
        modelConfig,
        (event) => {
          // Gestion des erreurs
          if (event.type === 'error') {
            console.error('❌ Erreur serveur:', event.message);
            alert(`❌ Erreur du serveur:\n\n${event.message}`);
            stopTraining();
            return;
          }

          // Progression de l'entraînement
          if (event.type === 'epoch') {
            addTrainingPoint(event.epoch || event.epochs, event.avg_loss);
          }

          // === VALIDATION ===
          else if (event.type === 'val_start') {
            valPairs.length = 0;
            valStart = event.idx_start || 0;
            console.log('📗 Validation démarrée à idx:', valStart);
          }
          else if (event.type === 'val_pair') {
            // Le serveur envoie { type: 'val_pair', idx, y, yhat }
            // y et yhat sont des scalaires
            valPairs.push({
              idx: event.idx,
              y: event.y,
              yhat: event.yhat,
            });
          }
          else if (event.type === 'val_end') {
            const metricsData = event.metrics;
            console.log('📗 Validation terminée:', valPairs.length, 'points');
            console.log('📗 Métriques VALIDATION:', metricsData);
            if (metricsData) {
              valMetrics = metricsData;
            }
          }

          // === PRÉDICTION (TEST) ===
          else if (event.type === 'pred_start') {
            predPoints.length = 0;
            predStart = event.idx_start || 0;
            console.log('📕 Prédiction démarrée à idx:', predStart);
          }
          else if (event.type === 'pred_point') {
            // Le serveur envoie { type: 'pred_point', idx, y, yhat, low, high }
            predPoints.push({
              idx: event.idx,
              y: event.y,
              yhat: event.yhat,
              low: event.low,
              high: event.high,
            });
          }
          else if (event.type === 'pred_end') {
            const metricsData = event.metrics;
            console.log('📕 Prédiction terminée:', predPoints.length, 'points');
            console.log('📕 Métriques PRÉDICTION:', metricsData);
            if (metricsData) {
              predMetrics = metricsData;
            }
          }

          // === SÉRIE COMPLÈTE ===
          else if (event.type === 'serie_complete') {
            serieComplete = event.values || event.serie || event.data || [];
            console.log('📊 Série complète reçue:', serieComplete.length, 'points');
          }

          // === DONNÉES FINALES ===
          else if (event.type === 'final_plot_data') {
            console.log('📊 final_plot_data reçu:', {
              series_complete: event.series_complete?.length,
              val_predictions: event.val_predictions?.length,
              pred_predictions: event.pred_predictions?.length,
              idx_val_start: event.idx_val_start,
              idx_test_start: event.idx_test_start,
            });

            // Utiliser directement les données du serveur
            const testingData = {
              type: 'final_plot_data' as const,
              series_complete: event.series_complete || serieComplete,
              val_predictions: event.val_predictions || valPairs.map(v => v.yhat),
              pred_predictions: event.pred_predictions || predPoints.map(p => p.yhat),
              pred_low: event.pred_low || predPoints.map(p => p.low),
              pred_high: event.pred_high || predPoints.map(p => p.high),
              idx_val_start: event.idx_val_start ?? valStart,
              idx_test_start: event.idx_test_start ?? predStart,
            };

            console.log('📊 TestingData à stocker:', {
              series_complete: testingData.series_complete.length,
              val_predictions: testingData.val_predictions.length,
              pred_predictions: testingData.pred_predictions.length,
              idx_val_start: testingData.idx_val_start,
              idx_test_start: testingData.idx_test_start,
            });

            setTestingData(testingData);

            // Stocker les métriques
            if (event.val_metrics) valMetrics = event.val_metrics;
            if (event.pred_metrics) predMetrics = event.pred_metrics;

            if (valMetrics || predMetrics) {
              const combinedMetrics: any = {};
              if (valMetrics) combinedMetrics.validation = valMetrics;
              if (predMetrics) combinedMetrics.prediction = predMetrics;
              console.log('📊 Métriques FINALES:', combinedMetrics);
              setMetrics(combinedMetrics);
            }
          }

          // === FIN DU PIPELINE ===
          else if (event.type === 'fin_pipeline') {
            console.log('✅ Pipeline terminé');
            
            // Si final_plot_data n'a pas été reçu, construire les données manuellement
            if (valPairs.length > 0 || predPoints.length > 0) {
              const currentTestingData = useStore.getState().testingData;
              
              // Seulement si testingData n'a pas été défini par final_plot_data
              if (!currentTestingData || currentTestingData.val_predictions.length === 0) {
                const testingData = {
                  type: 'final_plot_data' as const,
                  series_complete: serieComplete,
                  val_predictions: valPairs.map(v => v.yhat),
                  pred_predictions: predPoints.map(p => p.yhat),
                  pred_low: predPoints.map(p => p.low),
                  pred_high: predPoints.map(p => p.high),
                  idx_val_start: valStart,
                  idx_test_start: predStart,
                };
                
                console.log('📊 TestingData construit manuellement:', {
                  val: testingData.val_predictions.length,
                  pred: testingData.pred_predictions.length,
                });
                
                setTestingData(testingData);
              }
            }
          }
        },
        (error) => {
          console.error('❌ Erreur d\'entraînement:', error);
          stopTraining();
          alert(`Erreur lors de l'entraînement:\n\n${error.message || error}`);
        },
        () => {
          console.log('✅ Streaming terminé');
          stopTraining();
        }
      );
    } catch (error: any) {
      console.error('❌ Erreur:', error);
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
