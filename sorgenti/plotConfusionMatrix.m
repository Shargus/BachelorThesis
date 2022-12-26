function plotConfusionMatrix(predictedLabels,trueLabels)
% Costruisce e plotta la matrice di confusione per i dati specificati
% predictedLabels: risultati della classificazione (predizioni)
% trueLabels: classi effettive delle immagini

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(trueLabels,predictedLabels,'FontSize',15);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

end
