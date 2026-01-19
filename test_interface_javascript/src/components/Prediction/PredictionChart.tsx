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
import { Sparkles, Play, Loader2, AlertCircle } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { predictionAPI, datasetAPI } from '../../services/api';

interface PredictionData {
  series_complete: number[];
  predictions: number[];
  pred_low: number[];
  pred_high: number[];
}

export const PredictionChart: React.FC = () => {
  const { config } = useStore();
  const containerRef = useRef<HTMLDivElement>(null);
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');
  const [horizon, setHorizon] = useState(config.Parametres_temporels.horizon || 10);

  // État du zoom
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

  // État de la sélection
  const [selection, setSelection] = useState<{
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    active: boolean;
  }>({ x1: 0, y1: 0, x2: 0, y2: 0, active: false });

  const runPrediction = async () => {
    // Vérifier qu'un dataset est sélectionné
    if (!config.Parametres_temporels.nom_dataset) {
      setError('Aucun dataset sélectionné. Veuillez sélectionner un dataset dans l\'onglet Data.');
      return;
    }

    setLoading(true);
    setError(null);
    setPredictionData(null);
    setZoomState({ xMin: null, xMax: null, yMin: null, yMax: null }); // Reset zoom

    try {
      // 1. D'abord charger le dataset sur le serveur IA
      setStatus('Chargement du dataset...');
      console.log('📊 Chargement du dataset:', config.Parametres_temporels.nom_dataset);

      const datasetPayload = {
        name: config.Parametres_temporels.nom_dataset,
        dates: config.Parametres_temporels.dates,
        pas_temporel: config.Parametres_temporels.pas_temporel || 1,
      };

      await datasetAPI.fetchDataset(datasetPayload);
      console.log('✅ Dataset chargé sur le serveur');

      // 2. Lancer la prédiction
      setStatus('Génération des prédictions...');
      console.log('🚀 Lancement de la prédiction avec horizon:', horizon);

      const result = await predictionAPI.predict(horizon, 0.95);

      console.log('✅ Prédiction réussie:', {
        series_complete: result.series_complete?.length,
        predictions: result.predictions?.length,
      });

      setPredictionData({
        series_complete: result.series_complete || [],
        predictions: result.predictions || [],
        pred_low: result.pred_low || [],
        pred_high: result.pred_high || [],
      });
      setStatus('');
    } catch (err: any) {
      console.error('❌ Erreur de prédiction:', err);
      setError(err.message || 'Erreur lors de la prédiction');
      setStatus('');
    } finally {
      setLoading(false);
    }
  };

  // Préparer les données pour le graphique
  const chartData = predictionData
    ? (() => {
        const data: any[] = [];
        const n_history = predictionData.series_complete.length;

        // Données historiques
        predictionData.series_complete.forEach((value, index) => {
          data.push({
            index,
            historique: value,
          });
        });

        // Données de prédiction
        predictionData.predictions.forEach((value, i) => {
          const index = n_history + i;
          data.push({
            index,
            prediction: value,
            confidence: [predictionData.pred_low[i], predictionData.pred_high[i]],
          });
        });

        return data;
      })()
    : [];

  const n_history = predictionData?.series_complete.length || 0;

  // === GESTION DU ZOOM ===
  const handleMouseDown = (e: React.MouseEvent) => {
    if (!e.shiftKey || !containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setSelection({ x1: x, y1: y, x2: x, y2: y, active: true });
    e.preventDefault();
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!selection.active || !containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setSelection({ ...selection, x2: x, y2: y });
  };

  const handleMouseUp = () => {
    if (!selection.active || !containerRef.current || chartData.length === 0) return;

    const rect = containerRef.current.getBoundingClientRect();

    const margin = { left: 65, right: 30, top: 10, bottom: 40 };
    const chartW = rect.width - margin.left - margin.right;
    const chartH = rect.height - margin.top - margin.bottom;

    const relX1 = (selection.x1 - margin.left) / chartW;
    const relX2 = (selection.x2 - margin.left) / chartW;
    const relY1 = (selection.y1 - margin.top) / chartH;
    const relY2 = (selection.y2 - margin.top) / chartH;

    const currentXMin = zoomState.xMin ?? 0;
    const currentXMax = zoomState.xMax ?? chartData.length - 1;

    let currentYMin = zoomState.yMin;
    let currentYMax = zoomState.yMax;

    if (currentYMin === null || currentYMax === null) {
      currentYMin = Infinity;
      currentYMax = -Infinity;
      chartData.forEach(d => {
        [d.historique, d.prediction].forEach(v => {
          if (v !== null && v !== undefined) {
            currentYMin = Math.min(currentYMin!, v);
            currentYMax = Math.max(currentYMax!, v);
          }
        });
      });
    }

    const newXMin = currentXMin + relX1 * (currentXMax - currentXMin);
    const newXMax = currentXMin + relX2 * (currentXMax - currentXMin);
    const newYMax = currentYMax! - relY1 * (currentYMax! - currentYMin!);
    const newYMin = currentYMax! - relY2 * (currentYMax! - currentYMin!);

    const finalXMin = Math.round(Math.min(newXMin, newXMax));
    const finalXMax = Math.round(Math.max(newXMin, newXMax));
    const finalYMin = Math.min(newYMin, newYMax);
    const finalYMax = Math.max(newYMin, newYMax);

    if (finalXMax - finalXMin > 2) {
      setZoomState({
        xMin: finalXMin,
        xMax: finalXMax,
        yMin: finalYMin,
        yMax: finalYMax,
      });
    }

    setSelection({ x1: 0, y1: 0, x2: 0, y2: 0, active: false });
  };

  const handleDoubleClick = () => {
    setZoomState({ xMin: null, xMax: null, yMin: null, yMax: null });
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (chartData.length === 0) return;
    e.preventDefault();

    const { xMin, xMax } = zoomState;
    const currentXMin = xMin ?? 0;
    const currentXMax = xMax ?? chartData.length - 1;

    const range = currentXMax - currentXMin;
    const shift = Math.round(range * 0.1);

    let newXMin = currentXMin;
    let newXMax = currentXMax;

    if (e.deltaY > 0) {
      newXMin = Math.min(currentXMin + shift, chartData.length - range - 1);
      newXMax = newXMin + range;
    } else {
      newXMin = Math.max(currentXMin - shift, 0);
      newXMax = newXMin + range;
    }

    setZoomState({
      ...zoomState,
      xMin: newXMin,
      xMax: newXMax,
    });
  };

  const { xMin, xMax, yMin, yMax } = zoomState;
  const displayData = xMin !== null && xMax !== null
    ? chartData.slice(xMin, xMax + 1)
    : chartData;

  const selectionBox = selection.active ? {
    left: Math.min(selection.x1, selection.x2),
    top: Math.min(selection.y1, selection.y2),
    width: Math.abs(selection.x2 - selection.x1),
    height: Math.abs(selection.y2 - selection.y1),
  } : null;

  return (
    <div className="h-full flex flex-col p-6 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold text-white flex items-center">
          <Sparkles className="mr-3 text-purple-400" size={28} />
          Prédiction Future
        </h3>
      </div>

      {/* Contrôles */}
      <div className="flex items-center gap-4 bg-slate-800 border border-slate-700 rounded-lg p-4">
        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-300">Horizon:</label>
          <input
            type="number"
            min="1"
            max="100"
            value={horizon}
            onChange={(e) => setHorizon(parseInt(e.target.value) || 1)}
            className="w-20 px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-center focus:outline-none focus:border-purple-500"
          />
          <span className="text-sm text-gray-400">pas</span>
        </div>

        <button
          onClick={runPrediction}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 text-white rounded-lg font-medium transition-colors"
        >
          {loading ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Prédiction...
            </>
          ) : (
            <>
              <Play size={18} />
              Lancer la Prédiction
            </>
          )}
        </button>

        {predictionData && (
          <div className="text-sm text-gray-400">
            Historique: {predictionData.series_complete.length} points • Prédictions:{' '}
            {predictionData.predictions.length} points
          </div>
        )}
      </div>

      {/* Légende */}
      <div className="flex items-center justify-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-blue-500"></div>
          <span className="text-gray-300">Historique</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-red-500"></div>
          <span className="text-gray-300">Prédiction</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500/20 border border-red-500/40"></div>
          <span className="text-gray-300">IC 95%</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-0.5 h-4 bg-orange-500"></div>
          <span className="text-gray-300">Aujourd'hui</span>
        </div>
      </div>

      {/* Instructions zoom */}
      {predictionData && (
        <div className="text-center">
          <p className="text-[11px] text-gray-400 bg-slate-800/30 px-3 py-1.5 rounded inline-block">
            <kbd className="px-1.5 py-0.5 bg-slate-700 rounded text-[10px] text-white font-mono">Shift+Glisser</kbd> zoom •
            <kbd className="px-1.5 py-0.5 bg-slate-700 rounded text-[10px] text-white font-mono ml-1.5">Double-clic</kbd> reset •
            <kbd className="px-1.5 py-0.5 bg-slate-700 rounded text-[10px] text-white font-mono ml-1.5">Molette</kbd> déplacer
          </p>
        </div>
      )}

      {/* Graphique */}
      <div
        ref={containerRef}
        className="flex-1 bg-slate-800 border border-slate-700 rounded-xl p-4 relative"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onDoubleClick={handleDoubleClick}
        onWheel={handleWheel}
        onMouseLeave={() => selection.active && setSelection({ ...selection, active: false })}
      >
        {/* Rectangle de sélection */}
        {selectionBox && (
          <div
            className="absolute pointer-events-none z-50"
            style={{
              left: `${selectionBox.left}px`,
              top: `${selectionBox.top}px`,
              width: `${selectionBox.width}px`,
              height: `${selectionBox.height}px`,
              border: '2px solid #a855f7',
              backgroundColor: 'rgba(168, 85, 247, 0.1)',
            }}
          />
        )}

        {!predictionData && !loading && !error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <Sparkles size={48} className="mx-auto mb-4 opacity-30 text-purple-400" />
              <p className="text-lg text-gray-300 mb-2">Prédiction Future</p>
              {config.Parametres_temporels.nom_dataset ? (
                <p className="text-sm text-gray-500">
                  Dataset: <span className="text-purple-400">{config.Parametres_temporels.nom_dataset}</span>
                  <br />
                  Cliquez sur "Lancer la Prédiction"
                </p>
              ) : (
                <p className="text-sm text-red-400">
                  ⚠️ Aucun dataset sélectionné
                  <br />
                  <span className="text-gray-500">Sélectionnez un dataset dans l'onglet Data</span>
                </p>
              )}
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <AlertCircle size={48} className="mx-auto mb-4 text-red-400" />
              <p className="text-red-400 text-lg mb-2">❌ {error}</p>
              <p className="text-sm text-gray-500">
                Vérifiez qu'un modèle est chargé et qu'un dataset est sélectionné
              </p>
            </div>
          </div>
        )}

        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-800/80">
            <div className="text-center">
              <Loader2 size={48} className="mx-auto mb-4 animate-spin text-purple-400" />
              <p className="text-lg text-gray-300">{status || 'Génération des prédictions...'}</p>
            </div>
          </div>
        )}

        {predictionData && chartData.length > 0 && (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={displayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
              <XAxis
                dataKey="index"
                stroke="#94a3b8"
                style={{ fontSize: '12px' }}
                label={{
                  value: 'Index temporel',
                  position: 'insideBottom',
                  offset: -5,
                  fill: '#94a3b8',
                }}
                domain={['dataMin', 'dataMax']}
              />
              <YAxis
                stroke="#94a3b8"
                style={{ fontSize: '12px' }}
                label={{ value: 'Valeur', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                domain={yMin !== null && yMax !== null ? [yMin, yMax] : ['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{
                  background: 'rgba(30, 41, 59, 0.95)',
                  border: '1px solid rgba(148, 163, 184, 0.2)',
                  borderRadius: '8px',
                  color: '#f1f5f9',
                  fontSize: '12px',
                }}
                formatter={(value: any, name: string) => {
                  if (name === 'confidence') return null;
                  const label = name === 'historique' ? 'Historique' : 'Prédiction';
                  return [typeof value === 'number' ? value.toFixed(4) : value, label];
                }}
              />

              {/* Ligne de séparation "Aujourd'hui" */}
              <ReferenceLine
                x={n_history}
                stroke="#f97316"
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{ value: "Aujourd'hui", fill: '#f97316', fontSize: 12, position: 'top' }}
              />

              {/* Série historique */}
              <Line
                type="monotone"
                dataKey="historique"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="historique"
                isAnimationActive={false}
              />

              {/* Intervalle de confiance */}
              <Area
                type="monotone"
                dataKey="confidence"
                stroke="none"
                fill="#ef4444"
                fillOpacity={0.15}
                name="confidence"
                isAnimationActive={false}
              />

              {/* Prédictions */}
              <Line
                type="monotone"
                dataKey="prediction"
                stroke="#ef4444"
                strokeWidth={2.5}
                dot={{ r: 3, fill: 'white', stroke: '#ef4444', strokeWidth: 2 }}
                name="prediction"
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Info zoom */}
      {predictionData && (
        <div className="text-[10px] text-gray-500 text-center">
          Zoom: X=[{xMin ?? 'auto'}, {xMax ?? 'auto'}] Y=[{yMin?.toFixed(3) ?? 'auto'}, {yMax?.toFixed(3) ?? 'auto'}]
        </div>
      )}
    </div>
  );
};
