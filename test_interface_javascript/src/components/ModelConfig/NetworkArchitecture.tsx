import React from 'react';
import { Network } from 'lucide-react';
import { useStore } from '../../store/useStore';

export const NetworkArchitecture: React.FC = () => {
  const { config, modelConfig } = useStore();
  const modelType = config.Parametres_choix_reseau_neurones.modele;
  const archParams = modelConfig.Parametres_archi_reseau;

  const getModelDescription = () => {
    switch (modelType) {
      case 'MLP':
        return "Le MLP (Multi-Layer Perceptron) traite chaque point temporel indépendamment à travers des couches de neurones entièrement connectées. Adapté pour capturer des relations non-linéaires simples entre les variables.";
      case 'LSTM':
        return "Le LSTM (Long Short-Term Memory) possède une mémoire interne qui lui permet de retenir les patterns temporels sur de longues séquences. Idéal pour les séries avec dépendances temporelles complexes.";
      case 'CNN':
        return "Le CNN (Convolutional Neural Network) applique des filtres glissants sur la séquence temporelle pour détecter des motifs locaux récurrents. Efficace pour capturer des patterns périodiques.";
      default:
        return "";
    }
  };

  const getParamDescription = (param: string) => {
    const descriptions: Record<string, string> = {
      nb_couches: "Nombre de couches empilées dans le réseau. Plus de couches = capacité accrue à modéliser des relations complexes, mais risque de surapprentissage.",
      hidden_size: "Dimension de l'espace caché interne. Détermine la 'capacité mémoire' du modèle pour représenter les patterns temporels.",
      dropout_rate_mlp: "Probabilité de désactivation aléatoire des neurones pendant l'entraînement (0.0-0.9). Régularisation essentielle pour éviter le surapprentissage.",
      fonction_activation_mlp: "Fonction mathématique appliquée après chaque couche. ReLU (rapide), GELU (plus lisse), tanh (limite les valeurs entre -1 et 1).",
      bidirectional: "Analyse la séquence dans les deux sens (passé→futur et futur→passé). Double la capacité mais aussi le temps de calcul.",
      batch_first: "Format des données d'entrée : (batch, temps, features) si activé, sinon (temps, batch, features). Affecte uniquement l'interface d'entrée.",
      kernel_size: "Taille de la fenêtre temporelle analysée par chaque filtre. Kernel=3 capture des patterns sur 3 pas de temps consécutifs.",
      stride: "Pas de déplacement du filtre. Stride=1 analyse chaque position, stride=2 saute une position sur deux (réduit la taille de sortie).",
      padding: "Ajoute des zéros aux bords de la séquence pour contrôler la taille de sortie. Padding=0 réduit la taille, padding=(kernel-1)/2 la conserve.",
      fonction_activation_cnn: "Fonction non-linéaire appliquée après chaque convolution. Cruciale pour permettre au réseau d'apprendre des relations complexes.",
      dropout_rate_cnn: "Désactivation aléatoire de filtres entiers pendant l'entraînement. Aide à généraliser sur de nouvelles données.",
    };
    return descriptions[param] || "";
  };

  const renderMLPArchitecture = () => {
    const blocks = [];
    
    blocks.push(
      <div key="input" className="text-center">
        <div className="w-24 h-24 bg-blue-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Input</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">(B, in_dim)</p>
      </div>
    );
    
    blocks.push(<div key="arrow0" className="text-gray-500 text-xl">→</div>);
    
    blocks.push(
      <div key="fc_in" className="text-center">
        <div className="w-24 h-24 bg-purple-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Linear</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">→ {archParams.hidden_size}</p>
      </div>
    );
    
    blocks.push(<div key="arrow1" className="text-gray-500 text-xl">→</div>);
    
    blocks.push(
      <div key="act_in" className="text-center">
        <div className="w-24 h-24 bg-orange-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">{archParams.fonction_activation}</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">activation</p>
      </div>
    );
    
    for (let i = 0; i < (archParams.nb_couches - 1); i++) {
      blocks.push(<div key={`arrow_block${i}`} className="text-gray-500 text-xl">→</div>);
      
      blocks.push(
        <div key={`block${i}`} className="text-center">
          <div className="w-24 h-24 bg-purple-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
            <div className="text-sm">Linear</div>
          </div>
          <p className="text-[10px] text-gray-400 mt-1">{archParams.hidden_size}</p>
        </div>
      );
      
      blocks.push(<div key={`arrow_act${i}`} className="text-gray-500 text-xl">→</div>);
      blocks.push(
        <div key={`act${i}`} className="text-center">
          <div className="w-24 h-24 bg-orange-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
            <div className="text-sm">{archParams.fonction_activation}</div>
          </div>
          <p className="text-[10px] text-gray-400 mt-1">activation</p>
        </div>
      );
    }
    
    blocks.push(<div key="arrow_out" className="text-gray-500 text-xl">→</div>);
    
    blocks.push(
      <div key="fc_out" className="text-center">
        <div className="w-24 h-24 bg-purple-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Linear</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">→ out_dim</p>
      </div>
    );
    
    blocks.push(<div key="arrow_final" className="text-gray-500 text-xl">→</div>);
    
    blocks.push(
      <div key="output" className="text-center">
        <div className="w-24 h-24 bg-green-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Output</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">(B, out_dim)</p>
      </div>
    );
    
    return blocks;
  };

  const renderLSTMArchitecture = () => {
    const blocks = [];
    const direction = archParams.bidirectional ? '(Bi)' : '';
    
    blocks.push(
      <div key="input" className="text-center">
        <div className="w-24 h-24 bg-blue-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Input</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">(B,T,in_dim)</p>
      </div>
    );
    
    blocks.push(<div key="arrow0" className="text-gray-500 text-xl">→</div>);
    
    blocks.push(
      <div key="lstm_in" className="text-center">
        <div className="w-24 h-24 bg-red-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">LSTM{direction}</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">h={archParams.hidden_size}</p>
      </div>
    );
    
    for (let i = 0; i < (archParams.nb_couches - 1); i++) {
      blocks.push(<div key={`arrow${i + 1}`} className="text-gray-500 text-xl">→</div>);
      
      blocks.push(
        <div key={`lstm${i}`} className="text-center">
          <div className="w-24 h-24 bg-red-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
            <div className="text-sm">LSTM{direction}</div>
          </div>
          <p className="text-[10px] text-gray-400 mt-1">h={archParams.hidden_size}</p>
        </div>
      );
    }
    
    blocks.push(<div key="arrow_fc" className="text-gray-500 text-xl">→</div>);
    
    const factor = archParams.bidirectional ? 2 : 1;
    blocks.push(
      <div key="fc_out" className="text-center">
        <div className="w-24 h-24 bg-purple-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Linear</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">{archParams.hidden_size * factor}→out</p>
      </div>
    );
    
    blocks.push(<div key="arrow_out" className="text-gray-500 text-xl">→</div>);
    
    blocks.push(
      <div key="output" className="text-center">
        <div className="w-24 h-24 bg-green-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Output</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">(B,T,out)</p>
      </div>
    );
    
    return blocks;
  };

  const renderCNNArchitecture = () => {
    const blocks = [];
    
    blocks.push(
      <div key="input" className="text-center">
        <div className="w-24 h-24 bg-blue-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Input</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">(B,in,T)</p>
      </div>
    );
    
    blocks.push(<div key="arrow0" className="text-gray-500 text-xl">→</div>);
    
    blocks.push(
      <div key="conv_in" className="text-center">
        <div className="w-24 h-24 bg-indigo-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Conv1d</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">k={archParams.kernel_size}</p>
      </div>
    );
    
    blocks.push(<div key="arrow_act0" className="text-gray-500 text-xl">→</div>);
    
    blocks.push(
      <div key="act_in" className="text-center">
        <div className="w-24 h-24 bg-orange-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">{archParams.fonction_activation}</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">activation</p>
      </div>
    );
    
    for (let i = 0; i < (archParams.nb_couches - 1); i++) {
      blocks.push(<div key={`arrow_conv${i}`} className="text-gray-500 text-xl">→</div>);
      
      blocks.push(
        <div key={`conv${i}`} className="text-center">
          <div className="w-24 h-24 bg-indigo-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
            <div className="text-sm">Conv1d</div>
          </div>
          <p className="text-[10px] text-gray-400 mt-1">k={archParams.kernel_size}</p>
        </div>
      );
      
      blocks.push(<div key={`arrow_bn${i}`} className="text-gray-500 text-xl">→</div>);
      
      blocks.push(
        <div key={`bn${i}`} className="text-center">
          <div className="w-24 h-24 bg-cyan-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
            <div className="text-sm">BatchNorm</div>
          </div>
          <p className="text-[10px] text-gray-400 mt-1">norm</p>
        </div>
      );
      
      blocks.push(<div key={`arrow_act${i}`} className="text-gray-500 text-xl">→</div>);
      
      blocks.push(
        <div key={`act${i}`} className="text-center">
          <div className="w-24 h-24 bg-orange-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
            <div className="text-sm">{archParams.fonction_activation}</div>
          </div>
          <p className="text-[10px] text-gray-400 mt-1">activation</p>
        </div>
      );
    }
    
    blocks.push(<div key="arrow_out" className="text-gray-500 text-xl">→</div>);
    
    blocks.push(
      <div key="conv_out" className="text-center">
        <div className="w-24 h-24 bg-indigo-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Conv1d</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">→ out</p>
      </div>
    );
    
    blocks.push(<div key="arrow_final" className="text-gray-500 text-xl">→</div>);
    
    blocks.push(
      <div key="output" className="text-center">
        <div className="w-24 h-24 bg-green-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
          <div className="text-sm">Output</div>
        </div>
        <p className="text-[10px] text-gray-400 mt-1">(B,out,T')</p>
      </div>
    );
    
    return blocks;
  };

  const renderArchitecture = () => {
    switch (modelType) {
      case 'MLP':
        return renderMLPArchitecture();
      case 'LSTM':
        return renderLSTMArchitecture();
      case 'CNN':
        return renderCNNArchitecture();
      default:
        return [];
    }
  };

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
            <Network className="text-white" size={20} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Architecture du Réseau</h2>
            <p className="text-sm text-gray-400">Modèle {modelType}</p>
          </div>
        </div>

        {/* Description pédagogique */}
        <div className="bg-slate-800 border border-slate-700/50 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <div className="text-xl">📚</div>
            <div className="flex-1">
              <h4 className="text-white font-semibold mb-2">Comment fonctionne ce modèle ?</h4>
              <p className="text-sm text-gray-300 leading-relaxed">
                {getModelDescription()}
              </p>
            </div>
          </div>
        </div>

        {/* Schéma d'architecture */}
        <div className="bg-slate-800 border border-slate-700/50 rounded-lg p-8">
          <div className="overflow-x-auto">
            <div className="flex items-center justify-center space-x-6 min-w-max py-4">
              {renderArchitecture()}
            </div>
          </div>
        </div>

        {/* Paramètres */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h3 className="text-white font-semibold mb-4 flex items-center">
            <span className="text-xl mr-2">⚙️</span>
            Paramètres et leur rôle
          </h3>
          
          <div className="space-y-3">
            {/* Nombre de couches */}
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-medium">Nombre de couches</span>
                <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm font-bold">
                  {archParams.nb_couches}
                </span>
              </div>
              <p className="text-xs text-gray-400 leading-relaxed">
                {getParamDescription('nb_couches')}
              </p>
            </div>
            
            {/* Hidden Size */}
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-medium">Hidden Size</span>
                <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm font-bold">
                  {archParams.hidden_size}
                </span>
              </div>
              <p className="text-xs text-gray-400 leading-relaxed">
                {getParamDescription('hidden_size')}
              </p>
            </div>
            
            {/* Paramètres MLP */}
            {modelType === 'MLP' && archParams.fonction_activation && (
              <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-white font-medium">Fonction d'activation</span>
                  <span className="px-3 py-1 bg-yellow-500/20 text-yellow-400 rounded-full text-sm font-mono">
                    {archParams.fonction_activation}
                  </span>
                </div>
                <p className="text-xs text-gray-400 leading-relaxed">
                  {getParamDescription('fonction_activation_mlp')}
                </p>
              </div>
            )}
            
            {modelType === 'MLP' && archParams.dropout_rate !== undefined && (
              <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-white font-medium">Dropout Rate</span>
                  <span className="px-3 py-1 bg-orange-500/20 text-orange-400 rounded-full text-sm font-bold">
                    {archParams.dropout_rate}
                  </span>
                </div>
                <p className="text-xs text-gray-400 leading-relaxed">
                  {getParamDescription('dropout_rate_mlp')}
                </p>
              </div>
            )}
            
            {/* Paramètres LSTM */}
            {modelType === 'LSTM' && (
              <>
                <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-medium">Bidirectionnel</span>
                    <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                      archParams.bidirectional 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-gray-500/20 text-gray-400'
                    }`}>
                      {archParams.bidirectional ? 'Oui' : 'Non'}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    {getParamDescription('bidirectional')}
                  </p>
                </div>
                
                <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-medium">Batch First</span>
                    <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                      archParams.batch_first 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-gray-500/20 text-gray-400'
                    }`}>
                      {archParams.batch_first ? 'Oui' : 'Non'}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    {getParamDescription('batch_first')}
                  </p>
                </div>
              </>
            )}
            
            {/* Paramètres CNN */}
            {modelType === 'CNN' && (
              <>
                <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-medium">Kernel Size</span>
                    <span className="px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm font-bold">
                      {archParams.kernel_size || 3}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    {getParamDescription('kernel_size')}
                  </p>
                </div>
                
                <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-medium">Stride</span>
                    <span className="px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm font-bold">
                      {archParams.stride || 1}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    {getParamDescription('stride')}
                  </p>
                </div>
                
                <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-medium">Padding</span>
                    <span className="px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm font-bold">
                      {archParams.padding || 0}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    {getParamDescription('padding')}
                  </p>
                </div>
                
                {archParams.fonction_activation && (
                  <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-medium">Fonction d'activation</span>
                      <span className="px-3 py-1 bg-yellow-500/20 text-yellow-400 rounded-full text-sm font-mono">
                        {archParams.fonction_activation}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400 leading-relaxed">
                      {getParamDescription('fonction_activation_cnn')}
                    </p>
                  </div>
                )}
                
                {archParams.dropout_rate !== undefined && (
                  <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-medium">Dropout Rate</span>
                      <span className="px-3 py-1 bg-orange-500/20 text-orange-400 rounded-full text-sm font-bold">
                        {archParams.dropout_rate}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400 leading-relaxed">
                      {getParamDescription('dropout_rate_cnn')}
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};
