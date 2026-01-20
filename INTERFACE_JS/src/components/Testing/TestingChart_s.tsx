import React, { useState, useRef } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Database } from 'lucide-react';
import { useStore } from '../../store/useStore';

export const TestingChart: React.FC = () => {
  const { testingData } = useStore();
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoomState, setZoomState] = useState<{
    xMin: number | null;
    xMax: number | null;
    yMin: number | null;
    yMax: number | null;
  }>({
    xMin: null,
    xMax: null,
    yMin: null,
    yMax: null,
  });

  if (!testingData) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Database size={48} className="mx-auto mb-4 opacity-30 text-gray-400" />
          <p className="text-lg mb-2 text-gray-300">Aucune donnée de test disponible</p>
          <p className="text-sm text-gray-500">Lancez un entraînement pour voir les résultats</p>
        </div>
      </div>
    );
  }

  // DEBUG: Afficher les données reçues
  console.log('📊 [TestingChart] testingData:', {
    series_complete_length: testingData.series_complete?.length,
    val_predictions_length: testingData.val_predictions?.length,
    pred_predictions_length: testingData.pred_predictions?.length,
    idx_val_start: testingData.idx_val_start,
    idx_test_start: testingData.idx_test_start,
  });

  // Construire les données du graphique
  const chartData = testingData.series_complete.map((value, index) => {
    const dataPoint: any = {
      index,
      real: value,
    };

    // Validation (ligne verte) - commence à idx_val_start
    const valIndex = index - testingData.idx_val_start;
    if (valIndex >= 0 && valIndex < testingData.val_predictions.length) {
      dataPoint.validation = testingData.val_predictions[valIndex];
    }

    // Prédiction (ligne rouge) - commence à idx_test_start
    const predIndex = index - testingData.idx_test_start;
    if (predIndex >= 0 && predIndex < testingData.pred_predictions.length) {
      dataPoint.prediction = testingData.pred_predictions[predIndex];
      
      // Intervalle de confiance
      if (testingData.pred_low && testingData.pred_high) {
        dataPoint.confidence = [
          testingData.pred_low[predIndex],
          testingData.pred_high[predIndex]
        ];
      }
    }

    return dataPoint;
  });

  const handleDoubleClick = () => {
    setZoomState({ xMin: null, xMax: null, yMin: null, yMax: null });
  };

  const { xMin, xMax, yMin, yMax } = zoomState;
  const displayData = xMin !== null && xMax !== null
    ? chartData.slice(xMin, xMax + 1)
    : chartData;

  // Calculer les statistiques
  const nVal = testingData.val_predictions?.length || 0;
  const nPred = testingData.pred_predictions?.length || 0;

  return (
    <div className="h-full flex flex-col p-4 space-y-3 select-none">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">📊 Résultats de Test</h3>
      </div>

      {/* Statistiques */}
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-slate-800/50 border border-slate-700/50 p-2 rounded border-l-2 border-l-blue-500">
          <p className="text-xs text-gray-400">
            📈 Série: <span className="text-white font-semibold">{testingData.series_complete.length}</span>
          </p>
        </div>
        <div className="bg-slate-800/50 border border-slate-700/50 p-2 rounded border-l-2 border-l-green-500">
          <p className="text-xs text-gray-400">
            🔗 Valid: <span className="text-green-400 font-semibold">{nVal}</span>
            {nVal > 0 && (
              <span className="text-gray-500 text-[10px] ml-1">
                ({testingData.idx_val_start}→{testingData.idx_val_start + nVal})
              </span>
            )}
          </p>
        </div>
        <div className="bg-slate-800/50 border border-slate-700/50 p-2 rounded border-l-2 border-l-red-500">
          <p className="text-xs text-gray-400">
            📕 Pred: <span className="text-red-400 font-semibold">{nPred}</span>
            {nPred > 0 && (
              <span className="text-gray-500 text-[10px] ml-1">
                ({testingData.idx_test_start}→{testingData.idx_test_start + nPred})
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Légende */}
      <div className="flex items-center justify-center space-x-4 text-xs">
        <div className="flex items-center space-x-1.5">
          <div className="w-3 h-0.5 bg-blue-500"></div>
          <span className="text-gray-300">Réelle</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <div className="w-3 h-0.5 bg-green-500"></div>
          <span className="text-gray-300">Validation (teacher forcing)</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <div className="w-3 h-0.5 bg-red-500"></div>
          <span className="text-gray-300">Prédiction (autorégressive)</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <div className="w-3 h-3 bg-red-500 opacity-20"></div>
          <span className="text-gray-300">IC 95%</span>
        </div>
      </div>

      {/* Instructions */}
      <div className="text-center">
        <p className="text-[11px] text-gray-400 bg-slate-800/30 px-3 py-1.5 rounded inline-block">
          <kbd className="px-1.5 py-0.5 bg-slate-700 rounded text-[10px] text-white font-mono">Double-clic</kbd> reset zoom
        </p>
      </div>

      {/* Graphique */}
      <div
        ref={containerRef}
        className="flex-1 bg-slate-800 border border-slate-700 rounded-xl p-3 relative"
        onDoubleClick={handleDoubleClick}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={displayData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
            <XAxis
              dataKey="index"
              stroke="#94a3b8"
              style={{ fontSize: '11px' }}
              label={{ value: 'Index temporel', position: 'insideBottom', offset: -5, fill: '#94a3b8', fontSize: 11 }}
              domain={['dataMin', 'dataMax']}
            />
            <YAxis
              stroke="#94a3b8"
              style={{ fontSize: '11px' }}
              label={{ value: 'Valeur', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
              domain={yMin !== null && yMax !== null ? [yMin, yMax] : ['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                background: 'rgba(30, 41, 59, 0.95)',
                border: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '8px',
                color: '#f1f5f9',
                fontSize: '11px',
              }}
              formatter={(value: any, name: string) => {
                if (name === 'confidence') return null;
                const label = name === 'real' ? 'Réelle' : name === 'validation' ? 'Validation' : 'Prédiction';
                return [typeof value === 'number' ? value.toFixed(4) : value, label];
              }}
            />

            {/* Ligne de début de validation */}
            {testingData.idx_val_start > 0 && (
              <ReferenceLine
                x={testingData.idx_val_start}
                stroke="#10b981"
                strokeDasharray="3 3"
                strokeWidth={1.5}
                label={{ value: 'Début validation', fill: '#10b981', fontSize: 10, position: 'top' }}
              />
            )}

            {/* Ligne de début de test */}
            {testingData.idx_test_start > 0 && testingData.idx_test_start !== testingData.idx_val_start && (
              <ReferenceLine
                x={testingData.idx_test_start}
                stroke="#ef4444"
                strokeDasharray="3 3"
                strokeWidth={1.5}
                label={{ value: 'Début test', fill: '#ef4444', fontSize: 10, position: 'top' }}
              />
            )}

            {/* Série réelle (bleue) */}
            <Line
              type="monotone"
              dataKey="real"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="real"
              isAnimationActive={false}
            />

            {/* Validation (verte) */}
            <Line
              type="monotone"
              dataKey="validation"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              name="validation"
              isAnimationActive={false}
            />

            {/* Intervalle de confiance (rouge transparent) */}
            <Area
              type="monotone"
              dataKey="confidence"
              stroke="none"
              fill="#ef4444"
              fillOpacity={0.2}
              name="confidence"
              isAnimationActive={false}
            />

            {/* Prédiction (rouge) */}
            <Line
              type="monotone"
              dataKey="prediction"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              name="prediction"
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Info zoom */}
      <div className="text-[10px] text-gray-500 text-center">
        Zoom: X=[{xMin ?? 'auto'}, {xMax ?? 'auto'}] Y=[{yMin?.toFixed(3) ?? 'auto'}, {yMax?.toFixed(3) ?? 'auto'}]
      </div>
    </div>
  );
};
