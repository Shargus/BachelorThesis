%ResNet-50

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

%Carico la rete ResNet-50
net = resnet50;
% Dimensioni immagine di input
inputSize = net.Layers(1).InputSize;
%numClasses: numero di categorie di classificazione (2)
numClasses = numel(categories(cropTrain.Labels));

%Grafo della rete originale
lgraph = layerGraph(net);

% learnableLayer = lgraph.Layers(175);
% softmaxLayer = lgraph.Layers(175);
% classLayer = lgraph.Layers(177);

% Layer rimpiazzanti
newLearnableLayer = fullyConnectedLayer(numClasses,'Name','new_fc1000',...
    'WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
newSoftmaxLayer = softmaxLayer('Name','new_fc1000_softmax');
newClassLayer = classificationLayer(...
    'Name','new_ClassificationLayer_fc1000');

% Rimpiazzo degli ultimi 3 layer
lgraph = replaceLayer(lgraph,lgraph.Layers(175).Name,newLearnableLayer);
lgraph = replaceLayer(lgraph,lgraph.Layers(176).Name,newSoftmaxLayer);
lgraph = replaceLayer(lgraph,lgraph.Layers(177).Name,newClassLayer);

% Plotta il nuovo grafo ottenuto
plot(lgraph); ylim([0,10]);


%% Freeze dei weigths del primo layer convoluzionale

%Strati e connessioni della rete originale
layers = lgraph.Layers;
connections = lgraph.Connections;

% Blocca il learning rate di pesi e bias nei primi 10 layer (the initial 'stem' of the ResNet network)
layers(1:2) = freezeWeights(layers(1:2));
% Riconnetti tutti i layer nell'ordine originario
lgraph = createLgraphUsingConnections(layers,connections);  %PERMETTE DI AGGIRARE LA SOLA LETTURA DEGLI ATTRIBUTI


%% Re-addestramento ResNet-50 - Image pre-processing e augmentation

pixelRange = [-60 60];
angleRange = [-20,20];
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',angleRange, ...
    'RandXTranslation',pixelRange);

%Training set aumentato e ridimensionato 224x224
cropAugmentedTrain = augmentedImageDatastore(inputSize(1:2),cropTrain,'DataAugmentation',augmenter);
%Validation set ridimensionato 224x224
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
saveas(gcf,'confMat ResNet50.jpg');

%% Salvataggio

%Salvataggio workspace
save('workspace_resnet50.mat');
%Salvataggio della rete addestrata
save('TL_resnet50.mat','TL_net');
