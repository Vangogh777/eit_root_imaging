%% 诊断脚本 - 检查数据生成是否正确
% 在MATLAB中运行此脚本

clear; close all; clc;

%% 添加EIDORS路径
eidors_path = 'D:\eidors-v3.11ceshi\eidors-v3.11\eidors';
addpath(eidors_path);
cd('D:\EITProject');

%% 加载数据
load('eit_eidors_dataset.mat');

%% 诊断信息
fprintf('========== 数据诊断 ==========\n');
fprintf('网格单元数: %d\n', size(elements, 1));
fprintf('样本数: %d\n', size(voltages, 1));
fprintf('包含物参数范围:\n');
fprintf('  半径: %.1f - %.1f mm\n', min(inclusion_params(:,3))*1000, max(inclusion_params(:,3))*1000);
fprintf('  X位置: %.1f - %.1f mm\n', min(inclusion_params(:,1))*1000, max(inclusion_params(:,1))*1000);
fprintf('  Y位置: %.1f - %.1f mm\n', min(inclusion_params(:,2))*1000, max(inclusion_params(:,2))*1000);

% 检查包含物覆盖了多少单元
nonzero_counts = sum(conductivities ~= config.sigma_background, 2);
fprintf('包含物覆盖单元数:\n');
fprintf('  最小: %d 单元\n', min(nonzero_counts));
fprintf('  最大: %d 单元\n', max(nonzero_counts));
fprintf('  平均: %.1f 单元\n', mean(nonzero_counts));

if min(nonzero_counts) < 5
    fprintf('\n警告: 包含物太小，可能只覆盖了很少的单元！\n');
    fprintf('建议: 增大包含物半径或使用更细的网格\n');
end

%% 可视化几个样本
figure('Position', [100, 100, 1600, 600]);

% 选择包含物大小不同的样本
[~, sort_idx] = sort(nonzero_counts);
sample_indices = [sort_idx(1), sort_idx(round(end/2)), sort_idx(end)];  % 最小、中等、最大

for k = 1:3
    idx = sample_indices(k);

    % 电导率分布
    subplot(2, 3, k);
    patch('Faces', double(elements), 'Vertices', double(nodes), ...
          'FaceVertexCData', conductivities(idx, :)', ...
          'FaceColor', 'interp', 'EdgeColor', 'none');
    axis equal; axis off; colorbar;
    title(sprintf('样本 #%d (覆盖%d单元)', idx, nonzero_counts(idx)));

    % 标记包含物边界
    hold on;
    cx = inclusion_params(idx, 1);
    cy = inclusion_params(idx, 2);
    r = inclusion_params(idx, 3);
    theta_plot = linspace(0, 2*pi, 100);
    plot(cx + r*cos(theta_plot), cy + r*sin(theta_plot), 'r--', 'LineWidth', 2);
    hold off;

    % 电压测量
    subplot(2, 3, k+3);
    plot(voltages(idx, :), 'b-o', 'MarkerSize', 3);
    xlabel('测量索引'); ylabel('电压 (V)');
    title(sprintf('R=%.1fmm, sigma=%.3f', r*1000, inclusion_params(idx,4)));
    grid on;
end

sgtitle('EIDORS数据集诊断 - 左列最小覆盖，右列最大覆盖');
saveas(gcf, 'D:\EITProject\diagnosis.png');
fprintf('\n诊断图已保存到: D:\EITProject\diagnosis.png\n');
