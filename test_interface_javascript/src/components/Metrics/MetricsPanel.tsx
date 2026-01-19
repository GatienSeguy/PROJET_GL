import React from 'react';
import { BarChart3, TrendingUp, Award } from 'lucide-react';
import { useStore } from '../../store/useStore';

export const MetricsPanel: React.FC = () => {
  const { metrics } = useStore();

  // Enrichir les métriques avec RMSE calculé à partir de MSE
  const enrichMetrics = (data: any) => {
    if (!data) return null;
    
    const enriched = { ...data };
    
    // Calculer RMSE si MSE existe
    if (enriched.MSE !== undefined && enriched.RMSE === undefined) {
      enriched.RMSE = Math.sqrt(enriched.MSE);
    }
    if (enriched.mse !== undefined && enriched.rmse === undefined) {
      enriched.RMSE = Math.sqrt(enriched.mse);
    }
    
    return enriched;
  };

  const getMetricsToDisplay = () => {
    if (!metrics) return { validation: null, prediction: null };

    if (metrics.validation || metrics.prediction) {
      const valRaw = metrics.validation?.overall_mean || metrics.validation || null;
      const predRaw = metrics.prediction?.overall_mean || metrics.prediction || null;
      return { 
        validation: enrichMetrics(valRaw), 
        prediction: enrichMetrics(predRaw) 
      };
    }

    if (metrics.overall_mean) {
      return { validation: enrichMetrics(metrics.overall_mean), prediction: null };
    }

    return { validation: enrichMetrics(metrics), prediction: null };
  };

  const { validation, prediction } = getMetricsToDisplay();

  const metricConfig: { [key: string]: { color: string; label: string; description: string } } = {
    MSE: { color: '#60a5fa', label: 'MSE', description: 'Mean Squared Error' },
    mse: { color: '#60a5fa', label: 'MSE', description: 'Mean Squared Error' },
    MAE: { color: '#c084fc', label: 'MAE', description: 'Mean Absolute Error' },
    mae: { color: '#c084fc', label: 'MAE', description: 'Mean Absolute Error' },
    RMSE: { color: '#fb923c', label: 'RMSE', description: 'Root Mean Squared Error' },
    rmse: { color: '#fb923c', label: 'RMSE', description: 'Root Mean Squared Error' },
    R2: { color: '#2dd4bf', label: 'R²', description: 'Coefficient de détermination' },
    r2: { color: '#2dd4bf', label: 'R²', description: 'Coefficient de détermination' },
    MAPE: { color: '#fbbf24', label: 'MAPE', description: 'Mean Absolute % Error' },
    mape: { color: '#fbbf24', label: 'MAPE', description: 'Mean Absolute % Error' },
  };

  // Ordre d'affichage préféré
  const metricOrder = ['MSE', 'mse', 'MAE', 'mae', 'RMSE', 'rmse', 'R2', 'r2', 'MAPE', 'mape'];

  const formatValue = (value: number): string => {
    if (value === null || value === undefined) return 'N/A';
    if (Math.abs(value) < 0.0001) return value.toExponential(3);
    if (Math.abs(value) > 1000) return value.toFixed(2);
    return value.toFixed(6);
  };

  const renderMetricCard = (metric: string, value: number) => {
    const config = metricConfig[metric] || { color: '#94a3b8', label: metric, description: '' };
    return (
      <div
        key={metric}
        className="bg-slate-800/60 backdrop-blur-sm rounded-lg p-4 border border-slate-700/50 hover:border-slate-600 transition-all"
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-bold uppercase tracking-wide" style={{ color: config.color }}>
            {config.label}
          </span>
        </div>
        <p className="text-2xl font-bold text-white">{formatValue(value)}</p>
        <p className="text-xs text-gray-500 mt-1">{config.description}</p>
      </div>
    );
  };

  const renderSection = (title: string, data: any, icon: React.ReactNode, color: string) => {
    if (!data || typeof data !== 'object') return null;

    // Filtrer et ordonner les métriques
    const entries = metricOrder
      .filter(key => data[key] !== undefined && typeof data[key] === 'number')
      .map(key => [key, data[key]] as [string, number]);

    // Ajouter les métriques non listées dans l'ordre
    Object.entries(data).forEach(([key, value]) => {
      if (typeof value === 'number' && !metricOrder.includes(key)) {
        entries.push([key, value as number]);
      }
    });

    // Dédupliquer (garder MSE pas mse si les deux existent)
    const seen = new Set<string>();
    const uniqueEntries = entries.filter(([key]) => {
      const normalizedKey = key.toUpperCase();
      if (seen.has(normalizedKey)) return false;
      seen.add(normalizedKey);
      return true;
    });

    if (uniqueEntries.length === 0) return null;

    return (
      <div className="bg-slate-800/40 backdrop-blur-sm rounded-xl p-5 border border-slate-700/50">
        <div className="flex items-center mb-4">
          <div className="p-2 rounded-lg mr-3" style={{ backgroundColor: `${color}20` }}>
            {icon}
          </div>
          <h4 className="text-lg font-bold text-white">{title}</h4>
          <span className="ml-auto text-xs text-gray-500">{uniqueEntries.length} métriques</span>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {uniqueEntries.map(([metric, value]) => renderMetricCard(metric, value))}
        </div>
      </div>
    );
  };

  if (!metrics) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <BarChart3 size={64} className="mx-auto mb-4 text-blue-500/30" />
          <p className="text-xl font-bold mb-2 text-gray-200">Aucune métrique disponible</p>
          <p className="text-sm text-gray-400">Lancez un entraînement pour calculer les métriques</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-hidden p-6">
      <div className="h-full flex flex-col max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center mb-6">
          <div className="p-2.5 bg-slate-800/60 rounded-lg mr-3 border border-slate-700/50">
            <BarChart3 size={24} className="text-blue-400" />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-white">Métriques de Performance</h3>
            <p className="text-xs text-gray-400">Évaluation de la qualité des prédictions</p>
          </div>
        </div>

        {/* Métriques en 2 colonnes */}
        <div className="flex-1 grid grid-cols-2 gap-4 overflow-hidden">
          {/* Colonne Validation */}
          <div className="overflow-auto pr-2">
            {validation ? (
              renderSection(
                'Validation (Teacher Forcing)',
                validation,
                <TrendingUp size={20} className="text-emerald-400" />,
                '#10b981'
              )
            ) : (
              <div className="h-full flex items-center justify-center bg-slate-800/20 rounded-xl border border-slate-700/30">
                <p className="text-gray-500 text-sm">Aucune métrique de validation</p>
              </div>
            )}
          </div>

          {/* Colonne Prédiction */}
          <div className="overflow-auto pl-2">
            {prediction ? (
              renderSection(
                'Prédiction (Autorégressive)',
                prediction,
                <Award size={20} className="text-red-400" />,
                '#ef4444'
              )
            ) : (
              <div className="h-full flex items-center justify-center bg-slate-800/20 rounded-xl border border-slate-700/30">
                <p className="text-gray-500 text-sm">Aucune métrique de prédiction</p>
              </div>
            )}
          </div>
        </div>

        {/* Légende */}
        {(validation || prediction) && (
          <div className="mt-4 p-3 bg-slate-800/30 rounded-lg border border-slate-700/40">
            <div className="grid grid-cols-5 gap-2 text-[10px]">
              <div className="text-center">
                <div className="font-bold text-blue-400 mb-0.5">MSE</div>
                <div className="text-gray-500">Erreur quadratique moyenne</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-purple-400 mb-0.5">MAE</div>
                <div className="text-gray-500">Erreur absolue moyenne</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-orange-400 mb-0.5">RMSE</div>
                <div className="text-gray-500">Racine de MSE (√MSE)</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-teal-400 mb-0.5">R²</div>
                <div className="text-gray-500">Coeff. détermination</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-yellow-400 mb-0.5">MAPE</div>
                <div className="text-gray-500">Erreur % absolue</div>
              </div>
            </div>
          </div>
        )}

        {/* Note sur les métriques */}
        <div className="mt-2 text-center text-xs text-gray-600">
          💡 RMSE est calculé automatiquement à partir de MSE. R² et MAPE nécessitent un calcul côté serveur.
        </div>
      </div>
    </div>
  );
};
