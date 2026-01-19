import { useState } from 'react';
import { Activity, Layers, TrendingUp, FlaskConical, BarChart3, Sparkles } from 'lucide-react';
import { ModelSelector } from './components/ModelConfig/ModelSelector';
import { ModelManager } from './components/ModelConfig/ModelManager';
import { DatasetList } from './components/DatasetManager/DatasetList';
import { HorizonConfig } from './components/DatasetManager/HorizonConfig';
import { TimeConfig } from './components/TimeConfig/TimeConfig';
import { TrainingChart } from './components/Training/TrainingChart';
import { TrainingControl } from './components/Training/TrainingControl';
import { TrainingParams } from './components/Training/TrainingParams';
import { TestingChart } from './components/Testing/TestingChart';
import { PredictionChart } from './components/Prediction/PredictionChart';
import { NetworkArchitecture } from './components/ModelConfig/NetworkArchitecture';
import { MetricsPanel } from './components/Metrics/MetricsPanel';
import { useStore } from './store/useStore';

type MainTab = 'architecture' | 'training' | 'testing' | 'prediction' | 'metrics';
type SidebarTab = 'data' | 'model' | 'models' | 'train' | 'time';

function App() {
  const [activeTab, setActiveTab] = useState<MainTab>('architecture');
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('model');
  const { isTraining } = useStore();

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-slate-800 to-slate-900">
      {/* Header épuré */}
      <header className="bg-gradient-to-r from-slate-800 to-slate-900 border-b border-slate-700 shadow-xl">
        <div className="px-6 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center shadow-lg">
              <Activity className="text-white" size={20} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">IRMA ML</h1>
              <p className="text-xs text-gray-400">Machine Learning Application</p>
            </div>
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            isTraining
              ? 'bg-green-500/20 text-green-400 border border-green-500/30'
              : 'bg-gray-700 text-gray-300'
          }`}>
            {isTraining ? '🟢 En cours' : '⚪ Prêt'}
          </div>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar fixe et épurée */}
        <aside className="w-80 bg-gradient-to-b from-slate-800/90 to-slate-900/90 border-r border-slate-700 flex flex-col backdrop-blur-sm">
          {/* Onglets sidebar */}
          <div className="grid grid-cols-5 gap-1 p-2 border-b border-slate-700">
            <button
              onClick={() => setSidebarTab('data')}
              className={`px-2 py-2 rounded-lg text-xs font-medium transition-all ${
                sidebarTab === 'data'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              Data
            </button>
            <button
              onClick={() => setSidebarTab('model')}
              className={`px-2 py-2 rounded-lg text-xs font-medium transition-all ${
                sidebarTab === 'model'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              Model
            </button>
            <button
              onClick={() => setSidebarTab('models')}
              className={`px-2 py-2 rounded-lg text-xs font-medium transition-all ${
                sidebarTab === 'models'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              Save
            </button>
            <button
              onClick={() => setSidebarTab('train')}
              className={`px-2 py-2 rounded-lg text-xs font-medium transition-all ${
                sidebarTab === 'train'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              Train
            </button>
            <button
              onClick={() => setSidebarTab('time')}
              className={`px-2 py-2 rounded-lg text-xs font-medium transition-all ${
                sidebarTab === 'time'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              Time
            </button>
          </div>

          {/* Contenu sidebar */}
          <div className="flex-1 overflow-y-auto p-4">
            {sidebarTab === 'data' && <DatasetList />}
            {sidebarTab === 'model' && <ModelSelector />}
            {sidebarTab === 'models' && <ModelManager />}
            {sidebarTab === 'train' && <TrainingParams />}
            {sidebarTab === 'time' && <TimeConfig />}
          </div>

          {/* Bouton d'entraînement */}
          <div className="p-4 border-t border-slate-700">
            <TrainingControl />
          </div>
        </aside>

        {/* Contenu principal large */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Onglets principaux épurés */}
          <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-700 bg-slate-800/50">
            <button
              onClick={() => setActiveTab('architecture')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'architecture'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <Layers size={16} />
              <span>Architecture</span>
            </button>
            <button
              onClick={() => setActiveTab('training')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'training'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <TrendingUp size={16} />
              <span>Training</span>
            </button>
            <button
              onClick={() => setActiveTab('testing')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'testing'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <FlaskConical size={16} />
              <span>Testing</span>
            </button>
            <button
              onClick={() => setActiveTab('prediction')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'prediction'
                  ? 'bg-purple-500 text-white shadow-lg'
                  : 'text-gray-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <Sparkles size={16} />
              <span>Prediction</span>
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'metrics'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <BarChart3 size={16} />
              <span>Metrics</span>
            </button>
          </div>

          {/* Zone de contenu */}
          <div className="flex-1 overflow-hidden">
            {activeTab === 'architecture' && <NetworkArchitecture />}
            {activeTab === 'training' && (
              <div className="h-full p-6">
                <TrainingChart />
              </div>
            )}
            {activeTab === 'testing' && <TestingChart />}
            {activeTab === 'prediction' && <PredictionChart />}
            {activeTab === 'metrics' && (
              <div className="h-full p-6 overflow-y-auto">
                <MetricsPanel />
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
