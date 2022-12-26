%AlexNet

clear all
clc

%% Preparazione dataset

% Metto dataset con i crop in un oggetto di tipo datastore
datasetPath = 'Dataset Taranto';
cropDS = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Split in datastore di train e validation
[cropTrain,cropValidation] = splitEachLabel(cropDS,0.7,'randomized');


%% Inizializzazione CNN

% Carico la rete AlexNet
net = alexnet;
% Dimensioni immagine di input
inputSize = net.Layers(1).InputSize;
%numClasses: numero di categorie di classificazione (2)
numClasses = numel(categories(cropTrain.Labels));

% ESTRAZIONE DEI LAYER DA TRASFERIRE
% Gli ultimi tre layer della AlexNet pre-addestrata sono configurati per
% 1000 classi. Questi tre layer devono essere ri-configurati per il nuovo
% problema di classificazione (=> per il nuovo dataset). Questi layer sono:
% 1) 'fc8' FC layer --> last learnable layer (=ultimo layer provvisto di
% parametri addestrabili, e quindi di learning rate)
% 2) 'prob' Softmax layer (sta per "probabilities"; applica la softmax
% function)
% 3) 'output' Classification Output Layer (calcola la cross-entropy loss)

% Estrae tutti i layer tranne gli ultimi tre (che saranno da sostituire),
% quindi i primi 22
layersTransfer = net.Layers(1:end-3);


%% Freeze dei weigths dei primi 2 layer convoluzionali

layersTransfer(2).WeightLearnRateFactor = 0;
layersTransfer(2).BiasLearnRateFactor = 0;

layersTransfer(6).WeightLearnRateFactor = 0;
layersTransfer(6).BiasLearnRateFactor = 0;


%% Ricostruzione della rete

%layers e' il nuovo "grafo" della rete, da addestrare
layers = [
    
    layersTransfer                              %i primi 22
    
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,'Name','new_fc8'),    %nota il lr abbastanza alto
    softmaxLayer('Name','new_prob'),            %calcola la probabilita' per ogni classe
    classificationLayer('Name','new_output')];	%calcola la cross-entropy loss Li

%NB: i nuovi layer introdotti inferiscono il numero di classi (OutputSize) dall'OutputSize del layer precedente (il. For example, to specify the number of classes K of the
%network, include a fully connected layer with output size K and a softmax
%layer before the classification layer.

% NB:
% -il nuovo Fully Connected Layer ha OutputSize=numClasses=2 e
% InputSize='auto', cioe' in fase di trainNetwork legge l'OutputSize del
% Fully Connected Layer a lui precedente (ci sono in questo caso i layer
% ReLU e Dropout tra i due Fully Connected in questione), che nel caso di
% AlexNet e' 4096.
% -il nuovo Classification Layer ha OutputSize='auto', letto in fase di
% trainNetwork dall'OutputSize=numClasses=2 del Fully Connected Layer a lui
% precedente, e anche Classes='auto', impostato a ['No Pinna','Pinna'] dal
% cropAugmentedValidation (in qualche modo...)


%% Image pre-processing e augmentation

pixelRange = [-60 60];
angleRange = [-20,20];
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',angleRange, ...
    'RandXTranslation',pixelRange);

%Training set aumentato e ridimensionato 227x227
cropAugmentedTrain = augmentedImageDatastore(inputSize(1:2),cropTrain,...
    'DataAugmentation',augmenter);
%Validation set ridimensionato 227x227
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
    'ExecutionEnvironment','parallel', ...
    'CheckpointPath','.\Checkpoint AlexNet');

TL_net = trainNetwork(cropAugmentedTrain,layers,options);


%% Classificazione immagini del validation set

[prediction,probs] = classify(TL_net,cropAugmentedValidation);
accuracy = mean(prediction == cropValidation.Labels)


%% Matrice di confusione

plotConfusionMatrix(prediction,cropValidation.Labels)
saveas(gcf,'confMat AlexNet.jpg');


%% Salvataggio

%Salvataggio workspace
save('workspace_alexnet.mat');
%Salvataggio della rete addestrata
save('TL_alexnet.mat','TL_net');
