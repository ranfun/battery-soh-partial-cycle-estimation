%% plot_results.m
clear; clc;
dataDir = fullfile(fileparts(mfilename('fullpath')), 'data', 'cleaned_dataset');
load(fullfile(dataDir,'data_processed.mat'));
load(fullfile(dataDir,'results_lr.mat'));
load(fullfile(dataDir,'results_mlp.mat'));
load(fullfile(dataDir,'results_proposed.mat'));

nPlot = min(300, length(YTest_dn));
t = 1:nPlot;

%% Figure 1: SoH prediction comparison
figure('Position',[100 100 1000 400]);
plot(t, YTest_dn(t),    'k-',  'LineWidth',1.5, 'DisplayName','Actual SoH'); hold on;
plot(t, YPred_lr(t),    'b--', 'LineWidth',1,   'DisplayName','Linear Regression');
plot(t, YPred_mlp(t),   'r-',  'LineWidth',1,   'DisplayName','MLP');
plot(t, YPred_dn(t),    'g-',  'LineWidth',1.5, 'DisplayName','PI-NN (Proposed)');
xlabel('Test Sample Index'); ylabel('State of Health (SoH)');
title('Battery SoH Estimation: Model Comparison');
legend('Location','best'); grid on; ylim([0 1.1]);
saveas(gcf, fullfile(dataDir,'fig_soh_comparison.png'));

%% Figure 2: Metrics bar chart
models   = {'Linear Reg','MLP','PI-NN'};
rmse_all = [rmse_lr, rmse_mlp, rmse_prop];
mae_all  = [mae_lr,  mae_mlp,  mae_prop];

figure('Position',[100 100 700 300]);
subplot(1,2,1);
b1 = bar(rmse_all); set(gca,'XTickLabel',models,'FontSize',10);
title('RMSE (lower = better)'); ylabel('SoH Error'); grid on;
b1.FaceColor = 'flat'; b1.CData = [0.3 0.5 0.9; 0.9 0.5 0.3; 0.3 0.8 0.3];
subplot(1,2,2);
b2 = bar(mae_all);  set(gca,'XTickLabel',models,'FontSize',10);
title('MAE (lower = better)');  ylabel('SoH Error'); grid on;
b2.FaceColor = 'flat'; b2.CData = [0.3 0.5 0.9; 0.9 0.5 0.3; 0.3 0.8 0.3];
sgtitle('Performance Comparison Across Models');
saveas(gcf, fullfile(dataDir,'fig_metrics.png'));

%% Figure 3: Scatter — proposed model actual vs predicted
figure('Position',[100 100 500 500]);
scatter(YTest_dn, YPred_dn, 15, 'filled', 'MarkerFaceAlpha',0.5, 'MarkerFaceColor',[0.2 0.6 0.2]);
hold on;
mn = min([YTest_dn; YPred_dn]); mx = max([YTest_dn; YPred_dn]);
plot([mn mx],[mn mx],'r--','LineWidth',1.5);
xlabel('Actual SoH'); ylabel('Predicted SoH');
title('PI-NN: Actual vs Predicted SoH');
grid on; axis equal;
saveas(gcf, fullfile(dataDir,'fig_scatter.png'));

%% Figure 4: SoH degradation curve — one battery from test set
% Show how well the model tracks the full degradation trajectory
figure('Position',[100 100 700 350]);
plot(YTest_dn,  'k-',  'LineWidth',1.5, 'DisplayName','Actual'); hold on;
plot(YPred_dn,  'g--', 'LineWidth',1.5, 'DisplayName','PI-NN');
plot(YPred_mlp, 'r:',  'LineWidth',1,   'DisplayName','MLP');
xlabel('Test Sample Index'); ylabel('SoH');
title('SoH Degradation Trajectory Tracking');
legend('Location','best'); grid on;
saveas(gcf, fullfile(dataDir,'fig_trajectory.png'));

%% Summary table
fprintf('\n========= RESULTS SUMMARY =========\n');
fprintf('%-15s %8s %8s %10s\n','Model','RMSE','MAE','MAPE(%)');
fprintf('%-15s %8.4f %8.4f %10.2f\n','Linear Reg', rmse_lr,   mae_lr,   mape_lr);
fprintf('%-15s %8.4f %8.4f %10.2f\n','MLP',        rmse_mlp,  mae_mlp,  mape_mlp);
fprintf('%-15s %8.4f %8.4f %10.2f\n','PI-NN',      rmse_prop, mae_prop, mape_prop);
fprintf('====================================\n');
fprintf('Improvement over MLP — RMSE: %.1f%% | MAE: %.1f%%\n', ...
        (rmse_mlp-rmse_prop)/rmse_mlp*100, (mae_mlp-mae_prop)/mae_mlp*100);
