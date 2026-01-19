import React, { useState } from 'react';
import { Timer, RotateCcw } from 'lucide-react';
import { useStore } from '../../store/useStore';
import Calendar from 'react-calendar';
import 'react-calendar/dist/Calendar.css';
import './calendar-dark.css';

type CalendarValue = Date | null;

export const TimeConfig: React.FC = () => {
  const { config, updateConfig, datasets, selectedDataset } = useStore();
  const [showCalendarDebut, setShowCalendarDebut] = useState(false);
  const [showCalendarFin, setShowCalendarFin] = useState(false);

  const handleHorizonChange = (value: number) => {
    updateConfig({
      Parametres_temporels: {
        ...config.Parametres_temporels,
        horizon: Math.max(1, value),
      },
    });
  };

  const handleWindowSizeChange = (value: number) => {
    updateConfig({
      Parametres_temporels: {
        ...config.Parametres_temporels,
        window_size: Math.max(1, value),
      },
    });
  };

  const handleSplitChange = (value: number) => {
    updateConfig({
      Parametres_temporels: {
        ...config.Parametres_temporels,
        portion_decoupage: value / 100,
      },
    });
  };

  const handlePasTemporelChange = (value: number) => {
    updateConfig({
      Parametres_temporels: {
        ...config.Parametres_temporels,
        pas_temporel: Math.max(1, value),
      },
    });
  };

  const handleDateDebutChange = (date: CalendarValue) => {
    if (date) {
      const dateStr = date.toISOString().split('T')[0] + ' 00:00:00';
      updateConfig({
        Parametres_temporels: {
          ...config.Parametres_temporels,
          dates: [dateStr, config.Parametres_temporels.dates[1]],
        },
      });
      setShowCalendarDebut(false);
    }
  };

  const handleDateFinChange = (date: CalendarValue) => {
    if (date) {
      const dateStr = date.toISOString().split('T')[0] + ' 00:00:00';
      updateConfig({
        Parametres_temporels: {
          ...config.Parametres_temporels,
          dates: [config.Parametres_temporels.dates[0], dateStr],
        },
      });
      setShowCalendarFin(false);
    }
  };

  const handleResetDates = () => {
    if (selectedDataset && datasets[selectedDataset]) {
      const datasetInfo = datasets[selectedDataset];
      updateConfig({
        Parametres_temporels: {
          ...config.Parametres_temporels,
          dates: datasetInfo.dates,
        },
      });
    }
  };

  const parseDateString = (dateStr: string): Date => {
    const [datePart] = dateStr.split(' ');
    return new Date(datePart);
  };

  const formatDate = (dateStr: string): string => {
    return dateStr.split(' ')[0];
  };

  const datasetDates = selectedDataset && datasets[selectedDataset] 
    ? datasets[selectedDataset].dates 
    : null;

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white flex items-center">
        <Timer size={20} className="mr-2" />
        Paramètres Temporels
      </h3>

      <div className="space-y-4">
        {/* Horizon temporel */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Horizon temporel
          </label>
          <input
            type="number"
            min="1"
            value={config.Parametres_temporels.horizon}
            onChange={(e) => handleHorizonChange(parseInt(e.target.value) || 1)}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Window Size */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Fenêtre d'observation
          </label>
          <input
            type="number"
            min="1"
            value={config.Parametres_temporels.window_size || 15}
            onChange={(e) => handleWindowSizeChange(parseInt(e.target.value) || 15)}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Pas temporel */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Pas temporel
          </label>
          <input
            type="number"
            min="1"
            value={config.Parametres_temporels.pas_temporel}
            onChange={(e) => handlePasTemporelChange(parseInt(e.target.value) || 1)}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Portion Train/Test */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-gray-300">
              Portion Train/Test
            </label>
            <span className="text-sm text-blue-400">
              {(config.Parametres_temporels.portion_decoupage * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="50"
            max="95"
            step="5"
            value={config.Parametres_temporels.portion_decoupage * 100}
            onChange={(e) => handleSplitChange(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>

        {/* Période de données */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-gray-300">
              Période de données
            </label>
            {datasetDates && (
              <button
                onClick={handleResetDates}
                className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors"
              >
                <RotateCcw size={12} />
                <span>Reset</span>
              </button>
            )}
          </div>
          
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Date de début</label>
              <button
                onClick={() => {
                  setShowCalendarDebut(!showCalendarDebut);
                  setShowCalendarFin(false);
                }}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-left hover:border-blue-500 transition-colors"
              >
                {formatDate(config.Parametres_temporels.dates[0])}
              </button>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Date de fin</label>
              <button
                onClick={() => {
                  setShowCalendarFin(!showCalendarFin);
                  setShowCalendarDebut(false);
                }}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-left hover:border-blue-500 transition-colors"
              >
                {formatDate(config.Parametres_temporels.dates[1])}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Calendrier popup modal */}
      {(showCalendarDebut || showCalendarFin) && (
        <>
          <div 
            className="fixed inset-0 z-40 bg-black/40"
            onClick={() => {
              setShowCalendarDebut(false);
              setShowCalendarFin(false);
            }}
          />
          
          <div 
            className="fixed z-50 bg-slate-800 border border-slate-600 rounded-xl shadow-2xl p-3"
            style={{ 
              top: '50%', 
              left: '50%', 
              transform: 'translate(-50%, -50%)',
              width: '240px'
            }}
          >
            <div className="flex items-center justify-between mb-2 px-1">
              <span className="text-xs font-medium text-gray-300">
                {showCalendarDebut ? 'Date de début' : 'Date de fin'}
              </span>
              <button
                onClick={() => {
                  setShowCalendarDebut(false);
                  setShowCalendarFin(false);
                }}
                className="text-gray-400 hover:text-white text-lg leading-none"
              >
                ×
              </button>
            </div>
            <Calendar
              onChange={showCalendarDebut ? handleDateDebutChange : handleDateFinChange}
              value={parseDateString(
                showCalendarDebut 
                  ? config.Parametres_temporels.dates[0] 
                  : config.Parametres_temporels.dates[1]
              )}
              minDate={datasetDates ? parseDateString(datasetDates[0]) : undefined}
              maxDate={datasetDates ? parseDateString(datasetDates[1]) : undefined}
              locale="fr-FR"
              className="react-calendar-dark react-calendar-mini"
            />
          </div>
        </>
      )}
    </div>
  );
};
