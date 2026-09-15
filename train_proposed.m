%% train_proposed.m
% Proposed: Physics-Informed Neural Network for Battery SoH Estimation
% Novel contributions:
%   1. Multi-truncation features (30/50/75/100%) encode partial-cycle observability
%   2. Physics-informed loss: penalizes non-monotonic SoH predictions
%      (capacity fade is irreversible — SoH must decrease over cycle life)

clear; clc;
dataDir = fullfile(fileparts(mfilename('fullpath')), 'data', 'cleaned_dataset');
load(fullfile(dataDir, 'data_processed.mat'));

inputSize  = size(XTrain, 2);
numEpochs  = 200;
miniBatch  = 32;
learnRate  = 0.001;
lambda     = 0.05;   % small physics weight — regularizer, not dominant term

%% Build dlnetwork
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
lossLog = zeros(numEpochs, 2);

N      = size(XTrain, 1);
nBatch = floor(N / miniBatch);

for epoch = 1:numEpochs
    idx = randperm(N);
    mse_ep = 0; phy_ep = 0;

    for b = 1:nBatch
        iter = iter + 1;
        bi   = idx((b-1)*miniBatch+1 : b*miniBatch);

        XDL = dlarray(single(XTrain(bi,:))', 'CB');
        YDL = dlarray(single(YTrain(bi))',   'CB');
        Cb  = CycTrain(bi);
        Bb  = BatIDTrain(bi);

        [loss, grads, mse_v, phy_v] = dlfeval(@piLoss, net, XDL, YDL, Cb, Bb, lambda);

        mse_ep = mse_ep + double(extractdata(mse_v));
        phy_ep = phy_ep + double(extractdata(phy_v));

        currentLR = learnRate * (0.5^floor(epoch/60));
        [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, iter, currentLR);
    end

    lossLog(epoch,:) = [mse_ep/nBatch, phy_ep/nBatch];
    if mod(epoch,40) == 0
        fprintf('Epoch %d/%d — MSE: %.5f | Physics: %.5f\n', ...
                epoch, numEpochs, lossLog(epoch,1), lossLog(epoch,2));
    end
end

%% Predict
XTe_dl     = dlarray(single(XTest)', 'CB');
YPred_dl   = predict(net, XTe_dl);
YPred_dn   = double(extractdata(YPred_dl))';
YTest_dn   = YTest;

[rmse_prop, mae_prop, mape_prop] = calc_metrics(YTest_dn, YPred_dn);
fprintf('=== Physics-Informed NN (Proposed) ===\n');
fprintf('RMSE: %.4f | MAE: %.4f | MAPE: %.2f%%\n', rmse_prop, mae_prop, mape_prop);

save(fullfile(dataDir,'results_proposed.mat'), ...
     'net','YPred_dn','YTest_dn','rmse_prop','mae_prop','mape_prop','lossLog');

%% Loss plot
figure;
plot(lossLog(:,1),'b-','DisplayName','MSE Loss'); hold on;
plot(lossLog(:,2),'r--','DisplayName','Physics Loss');
xlabel('Epoch'); ylabel('Loss'); legend; grid on;
title('Physics-Informed Training Loss');
saveas(gcf, fullfile(dataDir,'fig_pi_loss.png'));

%% -----------------------------------------------------------------------
function [loss, grads, mse_loss, phy_loss] = piLoss(net, X, Y, cycNums, batIDs, lambda)
    YPred = forward(net, X);
    p = stripdims(YPred);
    y = stripdims(Y);

    mse_loss = mean((p(:) - y(:)).^2, 'all');

    % Physics: penalize SoH increase within same battery only
    uBats    = unique(batIDs);
    phy_loss = dlarray(single(0));
    nPairs   = 0;
    for k = 1:numel(uBats)
        mask = strcmp(batIDs, uBats{k});
        if sum(mask) < 2; continue; end
        [~, si] = sort(cycNums(mask));
        idx = find(mask); idx = idx(si);
        pk = p(idx);
        diffs = pk(2:end) - pk(1:end-1);
        phy_loss = phy_loss + sum(max(dlarray(zeros(size(diffs),'single')), diffs).^2, 'all');
        nPairs = nPairs + numel(diffs);
    end
    if nPairs > 0; phy_loss = phy_loss / nPairs; end

    loss  = mse_loss + lambda * phy_loss;
    grads = dlgradient(loss, net.Learnables);
end
