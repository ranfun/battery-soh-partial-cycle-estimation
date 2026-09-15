%% ablation_fullcycle_only.m
% Ablation Variant 1: PI-NN with full-cycle features only (no multi-truncation)
% Uses only the 100% truncation features (columns 25-32) + cycle index (col 33)

clear; clc;
dataDir = fullfile(fileparts(mfilename('fullpath')), 'data', 'cleaned_dataset');
load(fullfile(dataDir, 'data_processed.mat'));

% Use only full-cycle features (truncation level 4 = 100%) + cycle index
XTrain_abl = XTrain(:, 25:33);
XVal_abl   = XVal(:,   25:33);
XTest_abl  = XTest(:,  25:33);

inputSize  = size(XTrain_abl, 2);   % 9 features
numEpochs  = 200;
miniBatch  = 32;
learnRate  = 0.001;
lambda     = 0.05;

layers = [
    featureInputLayer(inputSize,   'Name','input')
    fullyConnectedLayer(128,       'Name','fc1')
    reluLayer(                     'Name','relu1')
    dropoutLayer(0.2,              'Name','drop1')
    fullyConnectedLayer(64,        'Name','fc2')
    reluLayer(                     'Name','relu2')
    dropoutLayer(0.1,              'Name','drop2')
    fullyConnectedLayer(32,        'Name','fc3')
    reluLayer(                     'Name','relu3')
    fullyConnectedLayer(1,         'Name','fc_out')
];

net    = dlnetwork(layerGraph(layers));
avgG   = []; avgSqG = []; iter = 0;
N      = size(XTrain_abl, 1);
nBatch = floor(N / miniBatch);

for epoch = 1:numEpochs
    idx = randperm(N);
    for b = 1:nBatch
        iter = iter + 1;
        bi   = idx((b-1)*miniBatch+1 : b*miniBatch);
        XDL  = dlarray(single(XTrain_abl(bi,:))', 'CB');
        YDL  = dlarray(single(YTrain(bi))',        'CB');
        Cb   = CycTrain(bi);
        Bb   = BatIDTrain(bi);
        [loss, grads, ~, ~] = dlfeval(@piLoss, net, XDL, YDL, Cb, Bb, lambda);
        currentLR = learnRate * (0.5^floor(epoch/60));
        [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, iter, currentLR);
    end
    if mod(epoch,40) == 0
        fprintf('Epoch %d/%d done\n', epoch, numEpochs);
    end
end

XTe_dl   = dlarray(single(XTest_abl)', 'CB');
YPred_abl = double(extractdata(predict(net, XTe_dl)))';
YTest_dn  = YTest;

[rmse_abl1, mae_abl1, ~] = calc_metrics(YTest_dn, YPred_abl);
fprintf('=== Ablation 1: Full-Cycle Only + Physics ===\n');
fprintf('RMSE: %.4f | MAE: %.4f\n', rmse_abl1, mae_abl1);
save(fullfile(dataDir,'results_ablation1.mat'), 'YPred_abl','rmse_abl1','mae_abl1');

function [loss, grads, mse_loss, phy_loss] = piLoss(net, X, Y, cycNums, batIDs, lambda)
    YPred = forward(net, X);
    p = stripdims(YPred); y = stripdims(Y);
    mse_loss = mean((p(:) - y(:)).^2, 'all');
    uBats = unique(batIDs); phy_loss = dlarray(single(0)); nPairs = 0;
    for k = 1:numel(uBats)
        mask = strcmp(batIDs, uBats{k});
        if sum(mask) < 2; continue; end
        [~, si] = sort(cycNums(mask)); idx = find(mask); idx = idx(si);
        pk = p(idx); diffs = pk(2:end) - pk(1:end-1);
        phy_loss = phy_loss + sum(max(dlarray(zeros(size(diffs),'single')), diffs).^2, 'all');
        nPairs = nPairs + numel(diffs);
    end
    if nPairs > 0; phy_loss = phy_loss / nPairs; end
    loss  = mse_loss + lambda * phy_loss;
    grads = dlgradient(loss, net.Learnables);
end
