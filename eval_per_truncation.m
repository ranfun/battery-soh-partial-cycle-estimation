%% eval_per_truncation.m
% Evaluate PI-NN at each truncation level independently
% Zero-pads features from other truncation levels to isolate each level's contribution
% Reports RMSE/MAE at 30%, 50%, 75%, 100% cycle completion

clear; clc;
dataDir = fullfile(fileparts(mfilename('fullpath')), 'data', 'cleaned_dataset');
load(fullfile(dataDir, 'data_processed.mat'));
load(fullfile(dataDir, 'results_proposed.mat'));

% Feature layout: [trunc1(1:8), trunc2(9:16), trunc3(17:24), trunc4(25:32), cycIdx(33)]
truncRanges = {1:8, 9:16, 17:24, 25:32};
truncLabels = {'30%', '50%', '75%', '100%'};
nLevels     = numel(truncRanges);

fprintf('%-10s %8s %8s\n', 'Truncation', 'RMSE', 'MAE');
fprintf('%s\n', repmat('-', 1, 28));

rmse_trunc = zeros(1, nLevels);
mae_trunc  = zeros(1, nLevels);

for t = 1:nLevels
    % Zero-pad all features except current truncation level + cycle index
    XTest_masked = zeros(size(XTest), 'like', XTest);
    XTest_masked(:, truncRanges{t}) = XTest(:, truncRanges{t});
    XTest_masked(:, 33)             = XTest(:, 33);   % always keep cycle index

    XTe_dl   = dlarray(single(XTest_masked)', 'CB');
    YPred_t  = double(extractdata(predict(net, XTe_dl)))';
    YTest_dn = YTest;

    [rmse_trunc(t), mae_trunc(t), ~] = calc_metrics(YTest_dn, YPred_t);
    fprintf('%-10s %8.4f %8.4f\n', truncLabels{t}, rmse_trunc(t), mae_trunc(t));
end

% Also evaluate full model (all features)
XTe_dl   = dlarray(single(XTest)', 'CB');
YPred_full = double(extractdata(predict(net, XTe_dl)))';
[rmse_full, mae_full, ~] = calc_metrics(YTest, YPred_full);
fprintf('%-10s %8.4f %8.4f\n', 'Full', rmse_full, mae_full);

save(fullfile(dataDir,'results_per_truncation.mat'), ...
     'rmse_trunc','mae_trunc','truncLabels','rmse_full','mae_full');
