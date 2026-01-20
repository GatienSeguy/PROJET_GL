import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useStore } from '../../store/useStore';

export const TrainingChart: React.FC = () => {
  const { trainingData, currentEpoch, totalEpochs } = useStore();
  const [isLogScale, setIsLogScale] = useState(false);

  const progress = (currentEpoch / totalEpochs) * 100;

  return (
    <div className="h-full flex flex-col p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold text-white flex items-center">
          Évolution de la Loss
        </h3>
        <label className="flex items-center space-x-2 text-sm">
          <input
            type="checkbox"
            checked={isLogScale}
            onChange={(e) => setIsLogScale(e.target.checked)}
            className="w-4 h-4 rounded bg-slate-700 border-slate-600"
          />
          <span className="text-gray-300">Échelle Log</span>
        </label>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="bg-slate-800 border border-slate-700 p-4 rounded-lg">
          <p className="text-gray-400 text-sm">Epoch</p>
          <p className="text-2xl font-bold text-white">
            {currentEpoch} / {totalEpochs}
          </p>
        </div>
        <div className="bg-slate-800 border border-slate-700 p-4 rounded-lg">
          <p className="text-gray-400 text-sm">Loss actuelle</p>
          <p className="text-2xl font-bold text-red-400">
            {trainingData.length > 0
              ? trainingData[trainingData.length - 1].loss.toFixed(6)
              : '-'}
          </p>
        </div>
        <div className="bg-slate-800 border border-slate-700 p-4 rounded-lg">
          <p className="text-gray-400 text-sm">Progression</p>
          <p className="text-2xl font-bold text-green-400">
            {progress.toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="w-full bg-slate-700 rounded-full h-2">
        <div
          className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="flex-1 bg-slate-800 border border-slate-700 rounded-xl p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={trainingData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
            <XAxis
              dataKey="epoch"
              stroke="#94a3b8"
              style={{ fontSize: '12px' }}
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
            />
            <YAxis
              scale={isLogScale ? 'log' : 'auto'}
              domain={isLogScale ? ['auto', 'auto'] : [0, 'auto']}
              stroke="#94a3b8"
              style={{ fontSize: '12px' }}
              label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
            />
            <Tooltip
              contentStyle={{
                background: 'rgba(30, 41, 59, 0.95)',
                border: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '8px',
                color: '#f1f5f9',
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#ef4444"
              strokeWidth={2}
              dot={{ r: 3, fill: '#ef4444' }}
              activeDot={{ r: 6 }}
              name="Loss"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
