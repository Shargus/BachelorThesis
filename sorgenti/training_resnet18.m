%ResNet-18

clear all
clc

%% Preparazione dataset

%Metto dataset in un oggetto di tipo datastore
datasetPath = 'Dataset Taranto';
cropDS = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Split in datastore di train e validation
[cropTrain,cropValidation] = splitEachLabel(cropDS,0.7,'randomized');


%% Inizializzazione CNN

% Carico la rete ResNet-18
net = resnet18;
% Dimensioni immagine di input
inputSize = net.Layers(1).InputSize;
%numClasses: numero di categorie di classificazione (2)
numClasses = numel(categories(cropTrain.Labels));

%Grafo della rete originale
lgraph = layerGraph(net);

% learnableLayer = lgraph.Layers(70);
% softmaxLayer = lgraph.Layers(71);
% classLayer = lgraph.Layers(72);

% Layer rimpiazzanti
newLearnableLayer = fullyConnectedLayer(numClasses,'Name','new_fc1000',...
    'WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
newSoftmaxLayer = softmaxLayer('Name','new_prob');
newClassLayer = classificationLayer(...
    'Name','new_ClassificationLayer_predictions');

% Rimpiazzo degli ultimi 3 layer
lgraph = replaceLayer(lgraph,lgraph.Layers(70).Name,newLearnableLayer);
lgraph = replaceLayer(lgraph,lgraph.Layers(71).Name,newSoftmaxLayer);
lgraph = replaceLayer(lgraph,lgraph.Layers(72).Name,newClassLayer);


%% Freeze dei weigths del primo layer convoluzionale

%Strati e connessioni della rete originale
layers = lgraph.Layers;
connections = lgraph.Connections;

% Blocca il learning rate di pesi e bias
layers(1:3) = freezeWeights(layers(1:3));
% Riconnetti tutti i layer nell'ordine originario
lgraph = createLgraphUsingConnections(layers,connections);


%% Image pre-processing e augmentation

pixelRange = [-60 60];
angleRange = [-20,20];
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',angleRange, ...
    'RandXTranslation',pixelRange);

% Training set aumentato e ridimensionato 224x224
cropAugmentedTrain = augmentedImageDatastore(inputSize(1:2),cropTrain,'DataAugmentation',augmenter);
% Validation set ridimensionato 224x224
cropAugmentedValidation = augmentedImageDatastore(inputSize(1:2),cropValidation);


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
    'ExecutionEnvironment','parallel', ...
    'CheckpointPath','.\Checkpoint ResNet-50');

TL_net = trainNetwork(cropAugmentedTrain,lgraph,options);


%% Classificazione immagini del validation set

[prediction,probs] = classify(TL_net,cropAugmentedValidation);
accuracy = mean(prediction == cropValidation.Labels)


%% Matrice di confusione

plotConfusionMatrix(prediction,cropValidation.Labels)
saveas(gcf,'confMat ResNet18.jpg');


%% Salvataggio

%Salvataggio workspace
save('workspace_resnet18.mat');
%Salvataggio della rete addestrata
save('TL_resnet18.mat','TL_net');
