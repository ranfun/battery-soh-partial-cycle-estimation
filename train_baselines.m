%% train_baselines.m
% Baseline 1: Linear Regression
% Baseline 2: MLP

clear; clc;
dataDir = fullfile(fileparts(mfilename('fullpath')), 'data', 'cleaned_dataset');
load(fullfile(dataDir, 'data_processed.mat'));

%% --- Baseline 1: Linear Regression ---
A        = [XTrain, ones(size(XTrain,1),1)];
beta     = A \ YTrain(:);
YPred_lr = [XTest, ones(size(XTest,1),1)] * beta;
YTest_dn = YTest;   % already in [0,1]

[rmse_lr, mae_lr, mape_lr] = calc_metrics(YTest_dn, YPred_lr);
fprintf('=== Linear Regression ===\n');
fprintf('RMSE: %.4f | MAE: %.4f | MAPE: %.2f%%\n', rmse_lr, mae_lr, mape_lr);
save(fullfile(dataDir,'results_lr.mat'), 'YPred_lr','YTest_dn','rmse_lr','mae_lr','mape_lr');

%% --- Baseline 2: MLP ---
inputSize = size(XTrain, 2);

layers_mlp = [
    featureInputLayer(inputSize)
    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(0.1)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

opts = trainingOptions('adam', ...
    'MaxEpochs',           150, ...
    'MiniBatchSize',       32, ...
    'InitialLearnRate',    0.001, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 50, ...
    'ValidationData',      {XVal, YVal}, ...
    'ValidationFrequency', 30, ...
    'GradientThreshold',   1, ...
    'Shuffle',             'every-epoch', ...
    'Plots',               'training-progress', ...
    'Verbose',             false);

net_mlp  = trainNetwork(XTrain, YTrain, layers_mlp, opts);
YPred_mlp = predict(net_mlp, XTest);

[rmse_mlp, mae_mlp, mape_mlp] = calc_metrics(YTest_dn, YPred_mlp);
fprintf('=== MLP Baseline ===\n');
fprintf('RMSE: %.4f | MAE: %.4f | MAPE: %.2f%%\n', rmse_mlp, mae_mlp, mape_mlp);
save(fullfile(dataDir,'results_mlp.mat'), 'net_mlp','YPred_mlp','YTest_dn','rmse_mlp','mae_mlp','mape_mlp');
