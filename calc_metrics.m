%% calc_metrics.m
function [rmse, mae, mape] = calc_metrics(YTrue, YPred)
    YTrue = YTrue(:); YPred = YPred(:);
    err   = YTrue - YPred;
    rmse  = sqrt(mean(err.^2));
    mae   = mean(abs(err));
    nz    = YTrue ~= 0;
    mape  = mean(abs(err(nz) ./ YTrue(nz))) * 100;
end
