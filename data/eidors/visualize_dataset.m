%% 可视化已生成的数据集
% 运行此脚本前请先运行 generate_eit_dataset_simple.m
% 使用方法: 在MATLAB命令行输入 run('D:\EITProject\visualize_dataset.m')

clear; close all; clc;

%% 添加EIDORS路径
eidors_path = 'D:\eidors-v3.11ceshi\eidors-v3.11\eidors';
addpath(eidors_path);

% 设置工作目录
project_path = 'D:\EITProject';
cd(project_path);

%% 加载数据
fprintf('加载数据集...\n');
load('eit_eidors_dataset.mat');

%% 可视化
fprintf('生成可视化...\n');

figure('Position', [100, 100, 1400, 400], 'Name', 'EIDORS数据集样本');

% 选择3个样本展示
sample_indices = [1, randi(config.n_samples), randi(config.n_samples)];

for k = 1:3
    idx = sample_indices(k);

    % 电导率分布
    subplot(2, 3, k);
    patch('Faces', double(elements), 'Vertices', double(nodes), ...
          'FaceVertexCData', conductivities(idx, :)', ...
          'FaceColor', 'interp', 'EdgeColor', 'none');
    axis equal; axis off;
    colorbar;
    title(sprintf('样本 #%d: 电导率分布', idx));

    % 标记包含物中心
    hold on;
    cx = inclusion_params(idx, 1);
    cy = inclusion_params(idx, 2);
    r = inclusion_params(idx, 3);
    theta_plot = linspace(0, 2*pi, 100);
    plot(cx + r*cos(theta_plot), cy + r*sin(theta_plot), 'r--', 'LineWidth', 1.5);
    hold off;

    % 电压测量
    subplot(2, 3, k+3);
    plot(voltages(idx, :), 'b-o', 'LineWidth', 1.2, 'MarkerSize', 3);
    xlabel('测量索引');
    ylabel('电压 (V)');
    title(sprintf('边界电压 (R=%.1fmm)', r*1000));
    grid on;
    xlim([0, config.n_meas_total]);
end

% 保存图像
saveas(gcf, 'D:\EITProject\eidors_samples_visualization.png');
fprintf('可视化已保存到: D:\EITProject\eidors_samples_visualization.png\n');

fprintf('\n完成！\n');
