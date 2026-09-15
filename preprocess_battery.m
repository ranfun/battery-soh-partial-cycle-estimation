%% preprocess_battery.m
% Load NASA battery dataset, compute SoH, build partial-cycle feature matrix
% Split: within-battery (70/15/15 per battery) — no cross-battery leakage

clear; clc;

dataDir = fullfile(fileparts(mfilename('fullpath')), 'data', 'cleaned_dataset');
csvDir  = fullfile(dataDir, 'data');

%% Load metadata
meta = readtable(fullfile(dataDir, 'metadata.csv'));
dischargeMask = strcmp(meta.type, 'discharge') & ~isnan(meta.Capacity);
dischargeMeta = meta(dischargeMask, :);

%% Filter outlier batteries (init capacity < 0.5 Ah)
batteries = unique(dischargeMeta.battery_id);
allBatCap = arrayfun(@(b) dischargeMeta.Capacity(find(strcmp(dischargeMeta.battery_id, batteries{b}),1,'first')), ...
                     1:numel(batteries));
batteries = batteries(allBatCap >= 0.5);
fprintf('Using %d batteries after filtering.\n', numel(batteries));

truncLevels = [0.30, 0.50, 0.75, 1.00];
nTrunc      = numel(truncLevels);

XTrain=[]; YTrain=[]; CycTrain=[]; BatIDTrain={};
XVal=[];   YVal=[];
XTest=[];  YTest=[];  CycTest=[];
BatTrain_ids = {};

for b = 1:numel(batteries)
    batID   = batteries{b};
    batRows = dischargeMeta(strcmp(dischargeMeta.battery_id, batID), :);
    batRows = sortrows(batRows, 'test_id');
    nCycles = height(batRows);
    initCap = batRows.Capacity(1);

    Xbat = []; Ybat = []; Cbat = [];

    for c = 1:nCycles
        fname    = batRows.filename{c};
        capacity = batRows.Capacity(c);
        SoH      = capacity / initCap;   % naturally in [0,1]

        fpath = fullfile(csvDir, fname);
        if ~isfile(fpath); continue; end
        cyc = readtable(fpath);

        cols   = {'Voltage_measured','Current_measured','Temperature_measured','Voltage_load','Time'};
        cyc    = cyc(:, cols);
        cyc    = fillmissing(cyc, 'linear');
        cycMat = table2array(cyc);
        T      = size(cycMat, 1);
        if T < 10; continue; end

        featVec = zeros(1, nTrunc * 8 + 1);
        for t = 1:nTrunc
            nPts = max(5, floor(truncLevels(t) * T));
            seg  = cycMat(1:nPts, :);
            V    = seg(:,1); I = seg(:,2); Temp = seg(:,3); ts = seg(:,5);
            dV   = diff(V);

            f(1) = mean(V);
            f(2) = min(V);
            f(3) = mean(I);
            f(4) = mean(Temp);
            f(5) = trapz(ts, abs(I));
            f(6) = mean(dV);
            f(7) = std(V);
            f(8) = (V(1)-V(end)) / max(ts(end), 1e-6);
            featVec(1, (t-1)*8+1 : t*8) = f;
        end
        featVec(end) = c / nCycles;   % normalized cycle index

        Xbat(end+1,:) = featVec;
        Ybat(end+1)   = SoH;
        Cbat(end+1)   = c;
        BatTrain_ids{end+1} = batID;
    end

    % Per-battery 70/15/15 split
    n    = size(Xbat,1);
    nTr  = floor(0.70*n);
    nVl  = floor(0.15*n);

    XTrain   = [XTrain;  Xbat(1:nTr,:)];
    YTrain   = [YTrain;  Ybat(1:nTr)'];
    CycTrain = [CycTrain; Cbat(1:nTr)'];
    BatIDTrain = [BatIDTrain; repmat({batID}, nTr, 1)];

    XVal    = [XVal;    Xbat(nTr+1:nTr+nVl,:)];
    YVal    = [YVal;    Ybat(nTr+1:nTr+nVl)'];

    XTest   = [XTest;   Xbat(nTr+nVl+1:end,:)];
    YTest   = [YTest;   Ybat(nTr+nVl+1:end)'];
    CycTest = [CycTest; Cbat(nTr+nVl+1:end)'];
end

%% Normalize features only (Y is already [0,1])
[X_all_norm, X_ps] = mapminmax([XTrain; XVal; XTest]', 0, 1);
X_all_norm = X_all_norm';
nTr = size(XTrain,1);
nVl = size(XVal,1);

XTrain = X_all_norm(1:nTr, :);
XVal   = X_all_norm(nTr+1:nTr+nVl, :);
XTest  = X_all_norm(nTr+nVl+1:end, :);

Y_ps = [];  % not used — SoH already in [0,1]

fprintf('Feature size: %dx%d\n', size([XTrain;XVal;XTest]));
fprintf('Train: %d | Val: %d | Test: %d\n', size(XTrain,1), size(XVal,1), size(XTest,1));

save(fullfile(dataDir,'data_processed.mat'), ...
     'XTrain','YTrain','XVal','YVal','XTest','YTest', ...
     'X_ps','Y_ps','CycTrain','CycTest','truncLevels','BatIDTrain');
