% layers = freezeWeights(layers) setta il learning rate dei layer
% specificati nel layer array 'layers' a 0.

function layers = freezeWeights(layers)

for ii = 1:numel(layers)   %Cicla da 1 al num. di layer da freezare
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            %se presenti, setta a 0 i campi WeigthLearnRateFactor e BiasLearnRateFactor
            layers(ii).(propName) = 0;
        end
    end
end

end

