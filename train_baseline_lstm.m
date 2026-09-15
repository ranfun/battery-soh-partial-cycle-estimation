%% train_baseline_lstm.m
% LSTM baseline — same features as PI-NN, standard MSE loss
% Each sample fed as [features x 1] sequence to LSTM

clear; clc;
dataDir = fullfile(fileparts(mfilename('fullpath')), 'data', 'cleaned_dataset');
load(fullfile(dataDir, 'data_processed.mat'));

inputSize = size(XTrain, 2);

% Convert rows to cell array of column vectors for trainNetwork
toSeq = @(X) cellfun(@(r) r', num2cell(X, 2), 'UniformOutput', false);
XTr = toSeq(XTrain);
XVl = toSeq(XVal);
XTe = toSeq(XTest);

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(64, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

opts = trainingOptions('adam', ...
    'MaxEpochs',           200, ...
    'MiniBatchSize',       32, ...
    'InitialLearnRate',    0.001, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 60, ...
    'ValidationData',      {XVl, YVal}, ...
    'ValidationFrequency', 30, ...
    'GradientThreshold',   1, ...
    'Shuffle',             'every-epoch', ...
    'Plots',               'training-progress', ...
    'Verbose',             false);

net_lstm = trainNetwork(XTr, YTrain, layers, opts);

YPred_lstm = predict(net_lstm, XTe);
YTest_dn   = YTest;

[rmse_lstm, mae_lstm, mape_lstm] = calc_metrics(YTest_dn, YPred_lstm);
fprintf('=== LSTM Baseline ===\n');
fprintf('RMSE: %.4f | MAE: %.4f | MAPE: %.2f%%\n', rmse_lstm, mae_lstm, mape_lstm);

save(fullfile(dataDir,'results_lstm.mat'), ...
     'net_lstm','YPred_lstm','YTest_dn','rmse_lstm','mae_lstm','mape_lstm');
