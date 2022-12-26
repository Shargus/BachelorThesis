%GoogLeNet

clear all
clc

%% Preparazione dataset

% Metto dataset in un oggetto di tipo datastore
datasetPath = 'Dataset Taranto';
cropDS = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Split in datastore di train e validation
[cropTrain,cropValidation] = splitEachLabel(cropDS,0.7,'randomized');


%% Inizializzazione CNN

% Carico la rete GoogLeNet
net = googlenet;
% Dimensioni immagine di input
inputSize = net.Layers(1).InputSize;
%numClasses: numero di categorie di classificazione (2)
numClasses = numel(categories(cropTrain.Labels));

%% Sostituzione last learnable layer e classification layer

%Grafo della rete originale
lgraph = layerGraph(net);

% Trova automaticamente i layer da rimpiazzare
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);

elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

%Rimpiazza il learnable layer nel grafo
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% Crea un nuovo classification layer, che avra' 0 classi (le classi saranno
% automaticamente associate a questo layer in fase di trainNetwork)
newClassLayer = classificationLayer('Name','new_classoutput');

%Rimpiazza il learnable layer nel grafo
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);


%% Freeze dei weigths dei primi 3 layer convoluzionali

%Strati e connessioni della rete originale
layers = lgraph.Layers;
connections = lgraph.Connections;

% Blocco il learning rate di pesi e bias
layers(1:10) = freezeWeights(layers(1:8));
% Riconnetto tutti i layer nell'ordine originario
lgraph = createLgraphUsingConnections(layers,connections);


%% Image pre-processing e augmentation

pixelRange = [-60 60];
angleRange = [-20,20];
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',angleRange, ...
    'RandXTranslation',pixelRange);

% Training set aumentato e ridimensionato 224x224
cropAugmentedTrain = augmentedImageDatastore(inputSize(1:2),cropTrain,...
    'DataAugmentation',augmenter);
% Validation set ridimensionato 224x224
cropAugmentedValidation = augmentedImageDatastore(...
    inputSize(1:2),cropValidation);

%% TRAINING

miniBatchSize = 20;
valFrequency = floor(numel(cropAugmentedTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', cropAugmentedValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'CheckpointPath','.\Checkpoint GoogLeNet');

TL_net = trainNetwork(cropAugmentedTrain,lgraph,options);


%% Classificazione immagini del validation set

[prediction,probs] = classify(TL_net,cropAugmentedValidation);
accuracy = mean(prediction == cropValidation.Labels)


%% Matrice di confusione

plotConfusionMatrix(prediction,cropValidation.Labels)
saveas(gcf,'confMat GoogLeNet.jpg');


%% Salvataggio

%Salvataggio workspace
save('workspace_googlenet.mat');
%Salvataggio della rete addestrata
save('TL_googlenet.mat','TL_net');
