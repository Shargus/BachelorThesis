clear
clc

%% DESCRIZIONE
% Questo programma effettua le seguenti operazioni:
% 1. test sul dataset delle azzorre con AlexNet, GoogLeNet, ResNet18
% 2. creazione di un file excel con due colonne, contenenti ciascuna gli
%    indirizzi delle immagini classificate come 'Pinna' e come 'No Pinna'


%% INIZIALIZZAZIONE DATASET

datasetPath = 'D:\Dati utente\Desktop\Tesi\Lavoro\Dataset Azzorre';
datasetDS = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% CARICAMENTO RETI + TEST + OUTPUT FILE EXCEL

netsList_temp = dir('TL_*');
% Escludi le reti da non usare nel major voting (in questo caso ResNet-50)
j=1;
for i=1:numel(netsList_temp)
    if string(netsList_temp(i).name)~="TL_resnet50.mat"
        netsList(j) = netsList_temp(i);
        j=j+1;
    end
end
        
n = numel(netsList);

for i=1:n
    %Caricamento della rete i-esima
    net_toExtract = load(netsList(i).name);
    name_asCellArray = fieldnames(net_toExtract);
    name = name_asCellArray{1};
    net = net_toExtract.(name);
    
    %Ridimensionamento del dataset all'inputSize della rete i-esima
    inputSize = net.Layers(1).InputSize;
    datasetDS_resize = augmentedImageDatastore(inputSize(1:2),datasetDS);
    
    %Classificazione con rete i-esima
    [prediction,probs] = classify(net,datasetDS_resize);
    %predictionProbs = (max(probs')');
    accuracy = mean(prediction == datasetDS.Labels)

    %Salvataggio risultati della rete i-esima in un file excel
    results = table(datasetDS.Files,datasetDS.Labels,prediction,probs(:,1),probs(:,2), ...
        'VariableNames',{'Crop','TrueClass','Prediction','Prob_No_Pinna','Prob_Pinna'});
    netName = netsList(i).name(4:end-4);
    outputExcel = ['Risultati Azzorre ',netName,'.xls'];
    writetable(results,outputExcel);
    
    %Stampa matrice di confusione relativa alla rete i-esima
    plotConfusionMatrix(prediction,datasetDS.Labels)
    saveas(gcf,['confMat Azzorre ',netName,'.png'])
end
