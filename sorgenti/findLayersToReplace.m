% findLayersToReplace(lgraph) trova il layer di classificazione e il layer
% addestrabile ad esso precedente (FC o CONV) nel grafo specificato da
% 'lgraph'

function [learnableLayer,classLayer] = findLayersToReplace(lgraph)

if ~isa(lgraph,'nnet.cnn.LayerGraph')
    error('Argument must be a LayerGraph object.')
end

src = string(lgraph.Connections.Source);
dst = string(lgraph.Connections.Destination);
layerNames = string({lgraph.Layers.Name}');

% Trova il layer di classificazione (unico in un grafo)
% Nel caso di GoogLeNet, isClassificationLayer e' un logical array 144x1 con
% tutti 0 tranne che l'1 in posizione 144.
isClassificationLayer = arrayfun(@(l) ...
    (isa(l,'nnet.cnn.layer.ClassificationOutputLayer')|...
    isa(l,'nnet.layer.ClassificationLayer')), lgraph.Layers);

if sum(isClassificationLayer) ~= 1
    error('Il grafo della rete deve avere un unico layer di classificazione')
end
% Il classification layer cosi' trovato
classLayer = lgraph.Layers(isClassificationLayer);

% Posizione del classification layer all'interno del grafo
currentLayerIdx = find(isClassificationLayer);
while true
    
    if numel(currentLayerIdx) ~= 1
        message = "Il grafo della rete deve avere un unico learnable"+...
            "layer prima del layer di classificazione";
        error(message)
    end
    
    currentLayerType = class(lgraph.Layers(currentLayerIdx));
    isLearnableLayer = ismember(currentLayerType, ...
        ['nnet.cnn.layer.FullyConnectedLayer','nnet.cnn.layer.Convolution2DLayer']);
    
    if isLearnableLayer
        learnableLayer =  lgraph.Layers(currentLayerIdx);
        return
    end
    
    currentDstIdx = find(layerNames(currentLayerIdx) == dst);
    currentLayerIdx = find(src(currentDstIdx) == layerNames);
    
end

end

