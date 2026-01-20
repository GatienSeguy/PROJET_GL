import React, { useState } from 'react';
import { Calendar as CalendarIcon, Clock, Percent, RotateCcw } from 'lucide-react';
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

  // Calculer les dates min/max du dataset
  const datasetDates = selectedDataset && datasets[selectedDataset] 
    ? datasets[selectedDataset].dates 
    : null;

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-3xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
            <Clock className="text-white" size={20} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Paramètres Temporels</h2>
            <p className="text-sm text-gray-400">Configuration des données temporelles</p>
          </div>
        </div>

        {/* Horizon temporel */}
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-5">
          <label className="block text-sm font-semibold text-white mb-2 flex items-center">
            <Clock size={16} className="mr-2 text-blue-400" />
            Horizon temporel
          </label>
          <input
            type="number"
            min="1"
            value={config.Parametres_temporels.horizon}
            onChange={(e) => handleHorizonChange(parseInt(e.target.value) || 1)}
            className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white text-lg focus:outline-none focus:border-blue-500"
          />
          <p className="text-xs text-gray-400 mt-2">
            Nombre de pas de temps à prédire dans le futur
          </p>
        </div>

        {/* Pas temporel */}
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-5">
          <label className="block text-sm font-semibold text-white mb-2 flex items-center">
            <Clock size={16} className="mr-2 text-purple-400" />
            Pas temporel
          </label>
          <input
            type="number"
            min="1"
            value={config.Parametres_temporels.pas_temporel}
            onChange={(e) => handlePasTemporelChange(parseInt(e.target.value) || 1)}
            className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white text-lg focus:outline-none focus:border-purple-500"
          />
          <p className="text-xs text-gray-400 mt-2">
            Intervalle entre deux points de données (en unités de temps du dataset)
          </p>
        </div>

        {/* Dates avec calendriers */}
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center">
              <CalendarIcon size={16} className="mr-2 text-green-400" />
              Période de données
            </h3>
            {datasetDates && (
              <button
                onClick={handleResetDates}
                className="flex items-center space-x-1 px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs text-gray-300 transition-colors"
              >
                <RotateCcw size={12} />
                <span>Réinitialiser</span>
              </button>
            )}
          </div>

          {datasetDates && (
            <div className="bg-blue-500/10 border border-blue-500/30 rounded p-2 text-xs text-blue-300">
              Période du dataset : {formatDate(datasetDates[0])} → {formatDate(datasetDates[1])}
            </div>
          )}
          
          {/* Date début */}
          <div className="relative">
            <label className="block text-xs text-gray-400 mb-2">Date de début</label>
            <button
              onClick={() => {
                setShowCalendarDebut(!showCalendarDebut);
                setShowCalendarFin(false);
              }}
              className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white text-left hover:border-green-500 transition-colors"
            >
              {formatDate(config.Parametres_temporels.dates[0])}
            </button>
            
            {showCalendarDebut && (
              <>
                {/* Overlay transparent */}
                <div 
                  className="fixed inset-0 z-40"
                  onClick={() => setShowCalendarDebut(false)}
                />
                
                {/* Calendrier compact */}
                <div className="absolute left-0 top-full mt-2 z-50 bg-slate-800 border border-slate-600 rounded-lg shadow-2xl p-3" style={{ width: '280px' }}>
                  <Calendar
                    onChange={handleDateDebutChange}
                    value={parseDateString(config.Parametres_temporels.dates[0])}
                    minDate={datasetDates ? parseDateString(datasetDates[0]) : undefined}
                    maxDate={datasetDates ? parseDateString(datasetDates[1]) : undefined}
                    locale="fr-FR"
                    className="react-calendar-dark react-calendar-compact"
                  />
                </div>
              </>
            )}
          </div>

          {/* Date fin */}
          <div className="relative">
            <label className="block text-xs text-gray-400 mb-2">Date de fin</label>
            <button
              onClick={() => {
                setShowCalendarFin(!showCalendarFin);
                setShowCalendarDebut(false);
              }}
              className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white text-left hover:border-green-500 transition-colors"
            >
              {formatDate(config.Parametres_temporels.dates[1])}
            </button>
            
            {showCalendarFin && (
              <>
                {/* Overlay transparent */}
                <div 
                  className="fixed inset-0 z-40"
                  onClick={() => setShowCalendarFin(false)}
                />
                
                {/* Calendrier compact */}
                <div className="absolute left-0 top-full mt-2 z-50 bg-slate-800 border border-slate-600 rounded-lg shadow-2xl p-3" style={{ width: '280px' }}>
                  <Calendar
                    onChange={handleDateFinChange}
                    value={parseDateString(config.Parametres_temporels.dates[1])}
                    minDate={datasetDates ? parseDateString(datasetDates[0]) : undefined}
                    maxDate={datasetDates ? parseDateString(datasetDates[1]) : undefined}
                    locale="fr-FR"
                    className="react-calendar-dark react-calendar-compact"
                  />
                </div>
              </>
            )}
          </div>
        </div>

        {/* Portion découpage */}
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-5">
          <label className="block text-sm font-semibold text-white mb-3 flex items-center justify-between">
            <span className="flex items-center">
              <Percent size={16} className="mr-2 text-orange-400" />
              Portion Train/Test
            </span>
            <span className="text-lg text-blue-400 font-bold">
              {(config.Parametres_temporels.portion_decoupage * 100).toFixed(0)}%
            </span>
          </label>
          <input
            type="range"
            min="50"
            max="95"
            step="5"
            value={config.Parametres_temporels.portion_decoupage * 100}
            onChange={(e) => handleSplitChange(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-2">
            <span>50% Train</span>
            <span>95% Train</span>
          </div>
          <p className="text-xs text-gray-400 mt-3">
            Pourcentage des données utilisées pour l'entraînement (le reste pour le test)
          </p>
        </div>

        {/* Info dataset sélectionné */}
        {config.Parametres_temporels.nom_dataset && (
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <p className="text-sm text-blue-300">
              <span className="font-semibold">Dataset actif :</span> {config.Parametres_temporels.nom_dataset}
            </p>
            <p className="text-xs text-blue-400 mt-1">
              Période sélectionnée : {formatDate(config.Parametres_temporels.dates[0])} → {formatDate(config.Parametres_temporels.dates[1])}
            </p>
          </div>
        )}

      </div>
    </div>
  );
};
