% lgraph = createLgraphUsingConnections(layers,connections) crea il grafo
% di una rete con strati specificati nel layer array 'layers' e connessi
% con le connessioni specificate nella tabella 'connections'.

function lgraph = createLgraphUsingConnections(layers,connections)

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

end

