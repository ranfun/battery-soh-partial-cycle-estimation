%% eval_lambda_sensitivity.m
% Test lambda values: 0, 0.01, 0.05, 0.1, 0.3, 0.5
% Reports RMSE/MAE for each — produces sensitivity table for paper

clear; clc;
dataDir = fullfile(fileparts(mfilename('fullpath')), 'data', 'cleaned_dataset');
load(fullfile(dataDir, 'data_processed.mat'));

lambdas   = [0, 0.01, 0.05, 0.1, 0.3, 0.5];
rmse_lam  = zeros(1, numel(lambdas));
mae_lam   = zeros(1, numel(lambdas));
inputSize = size(XTrain, 2);
numEpochs = 200; miniBatch = 32; learnRate = 0.001;

rng(1);  % fixed seed for fair comparison

for li = 1:numel(lambdas)
    lambda = lambdas(li);
    fprintf('Training lambda = %.3f\n', lambda);

    layers = [
        featureInputLayer(inputSize,'Name','input')
        fullyConnectedLayer(128,   'Name','fc1') reluLayer('Name','relu1') dropoutLayer(0.2,'Name','drop1')
        fullyConnectedLayer(64,    'Name','fc2') reluLayer('Name','relu2') dropoutLayer(0.1,'Name','drop2')
        fullyConnectedLayer(32,    'Name','fc3') reluLayer('Name','relu3')
        fullyConnectedLayer(1,     'Name','fc_out')
    ];

    net = dlnetwork(layerGraph(layers));
    avgG = []; avgSqG = []; iter = 0;
    N = size(XTrain,1); nBatch = floor(N/miniBatch);

    for epoch = 1:numEpochs
        idx = randperm(N);
        for b = 1:nBatch
            iter = iter + 1;
            bi   = idx((b-1)*miniBatch+1 : b*miniBatch);
            XDL  = dlarray(single(XTrain(bi,:))','CB');
            YDL  = dlarray(single(YTrain(bi))','CB');
            [loss, grads, ~, ~] = dlfeval(@piLoss, net, XDL, YDL, CycTrain(bi), BatIDTrain(bi), lambda);
            currentLR = learnRate * (0.5^floor(epoch/60));
            [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, iter, currentLR);
        end
    end

    YPred = double(extractdata(predict(net, dlarray(single(XTest)','CB'))))';
    [rmse_lam(li), mae_lam(li), ~] = calc_metrics(YTest, YPred);
    fprintf('  lambda=%.3f: RMSE=%.4f MAE=%.4f\n', lambda, rmse_lam(li), mae_lam(li));
end

fprintf('\n=== Lambda Sensitivity ===\n');
fprintf('%-8s %8s %8s\n','Lambda','RMSE','MAE');
for li = 1:numel(lambdas)
    fprintf('%-8.3f %8.4f %8.4f\n', lambdas(li), rmse_lam(li), mae_lam(li));
end

save(fullfile(dataDir,'results_lambda.mat'),'lambdas','rmse_lam','mae_lam');

% Plot
figure;
plot(lambdas, rmse_lam, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor','b'); hold on;
plot(lambdas, mae_lam,  'rs-', 'LineWidth', 1.5, 'MarkerFaceColor','r');
xlabel('\lambda (physics loss weight)'); ylabel('Error');
legend('RMSE','MAE','Location','best'); grid on;
title('Sensitivity to Physics Loss Weight \lambda');
saveas(gcf, fullfile(dataDir,'re/fig_lambda_sensitivity.png'));

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
        phy_loss = phy_loss + sum(max(dlarray(zeros(size(diffs),'single')), diffs).^2,'all');
        nPairs = nPairs + numel(diffs);
    end
    if nPairs > 0; phy_loss = phy_loss / nPairs; end
    loss  = mse_loss + lambda * phy_loss;
    grads = dlgradient(loss, net.Learnables);
end
