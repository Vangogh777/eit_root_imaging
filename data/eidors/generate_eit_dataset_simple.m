%% EIDORS EIT Training Dataset Generator
% 生成用于神经网络的EIT训练数据集
% 输入：边界电压测量 (1×256)
% 输出：电导率分布
%
% 作者：Claude Code Assistant
% 日期：2025-06-13
%
% 使用方法:
%   1. 修改下方的EIDORS路径
%   2. 运行脚本: generate_eit_dataset_simple

clear; close all; clc;

%% 添加EIDORS路径 (根据你的安装位置修改)
addpath('/path/to/eidors');  % <<<< 修改为你的EIDORS安装路径 >>>>
eidors_on;

%% ==================== 配置参数 ====================
config = struct();

% 网格参数
config.n_electrodes = 16;       % 电极数量
config.radius = 0.1;            % 圆形区域半径 (m) = 10cm

% 电导率参数
config.sigma_background = 0.01; % 背景(土壤)电导率 S/m
config.sigma_inclusion = 0.05;  % 包含物(根系)电导率 S/m

% 数据集参数
config.n_samples = 10000;       % 样本数量

% 输出路径
config.output_path = './eit_eidors_dataset.mat';

%% ==================== 创建EIDORS正向模型 ====================
fprintf('====================================\n');
fprintf('创建EIDORS正向模型...\n');
fprintf('====================================\n');

% 方法1: 使用mk_common_model (推荐)
fwd_model = mk_common_model('c2c', config.n_electrodes);
fwd_model = fwd_model.fwd_model;

% 方法2: 如果方法1失败，使用mk_circ_model
% fwd_model = mk_circ_model([], config.n_electrodes, 'A2c0.05');

% 设置激励模式：相邻激励 (adjacent stimulation)
% '{ad}' 表示相邻激励，测量也是相邻电极
stimulation = mk_stim_patterns(config.n_electrodes, 1, '{ad}', '{ad}');
fwd_model.stimulation = stimulation;

% 获取网格信息
nodes = fwd_model.nodes;
elements = fwd_model.elems;
n_nodes = size(nodes, 1);
n_elements = size(elements, 1);

% 计算测量数量
n_meas_total = 0;
for i = 1:length(fwd_model.stimulation)
    n_meas_total = n_meas_total + size(fwd_model.stimulation(i).meas_pattern, 1);
end
config.n_meas_total = n_meas_total;

fprintf('网格信息:\n');
fprintf('  节点数: %d\n', n_nodes);
fprintf('  单元数: %d\n', n_elements);
fprintf('  总测量数: %d\n', n_meas_total);
fprintf('  电极数: %d\n', config.n_electrodes);

%% ==================== 计算单元中心坐标 ====================
elem_centers = zeros(n_elements, 2);
for e = 1:n_elements
    elem_nodes = elements(e, :);
    elem_centers(e, :) = mean(nodes(elem_nodes, :), 1);
end

%% ==================== 初始化数据存储 ====================
voltages = zeros(config.n_samples, config.n_meas_total);
conductivities = zeros(config.n_samples, n_elements);
inclusion_masks = zeros(config.n_samples, n_elements);

% 存储包含物参数（用于验证）
inclusion_params = zeros(config.n_samples, 4); % [cx, cy, radius, sigma]

%% ==================== 主数据生成循环 ====================
fprintf('\n====================================\n');
fprintf('开始生成 %d 个样本...\n', config.n_samples);
fprintf('====================================\n');

tic;
for i = 1:config.n_samples
    % 进度显示
    if mod(i, 1000) == 0 || i == 1
        fprintf('进度: %d/%d (%.1f%%)\n', i, config.n_samples, 100*i/config.n_samples);
    end

    % ===== 生成单一圆形包含物 =====

    % 随机位置 (在圆形区域内)
    r = sqrt(rand()) * config.radius * 0.7;  % sqrt保证均匀分布
    theta = rand() * 2 * pi;
    cx = r * cos(theta);
    cy = r * sin(theta);

    % 随机半径 (5mm - 25mm)
    radius = rand() * config.radius * 0.2 + config.radius * 0.05;

    % 随机电导率 (在范围内变化)
    sigma_inc = config.sigma_inclusion * (0.8 + 0.4 * rand());

    % 初始化电导率分布为背景值
    sigma = config.sigma_background * ones(n_elements, 1);
    mask = zeros(n_elements, 1);

    % 计算每个单元中心到包含物中心的距离
    dist = sqrt((elem_centers(:, 1) - cx).^2 + (elem_centers(:, 2) - cy).^2);

    % 在包含物内的单元设置为包含物电导率
    in_inclusion = dist < radius;
    sigma(in_inclusion) = sigma_inc;
    mask(in_inclusion) = 1;

    % 存储包含物参数
    inclusion_params(i, :) = [cx, cy, radius, sigma_inc];

    % ===== 正向求解 =====

    % 创建EIDORS图像对象
    img = mk_image(fwd_model, sigma);

    % 计算边界电压
    data = fwd_solve(img);

    % 存储结果
    voltages(i, :) = data.meas';
    conductivities(i, :) = sigma';
    inclusion_masks(i, :) = mask';
end

elapsed_time = toc;
fprintf('数据生成完成！总耗时: %.2f 秒 (%.2f ms/样本)\n', elapsed_time, 1000*elapsed_time/config.n_samples);

%% ==================== 添加噪声（可选）====================
noise_level = 0.001;  % 0.1% 相对噪声
voltages_noisy = voltages + noise_level * randn(size(voltages)) .* abs(voltages);

%% ==================== 保存数据集 ====================
fprintf('\n保存数据集到: %s\n', config.output_path);

% 确保输出目录存在
output_dir = fileparts(config.output_path);
if ~isempty(output_dir) && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存为MAT文件
save(config.output_path, ...
    'voltages', 'voltages_noisy', ...
    'conductivities', 'inclusion_masks', 'inclusion_params', ...
    'nodes', 'elements', 'elem_centers', ...
    'config', 'fwd_model', ...
    '-v7.3');

fprintf('数据集保存完成！\n');

%% ==================== 数据统计 ====================
fprintf('\n====================================\n');
fprintf('数据集统计\n');
fprintf('====================================\n');
fprintf('样本数量: %d\n', config.n_samples);
fprintf('电压维度: %d (应该接近256)\n', config.n_meas_total);
fprintf('电导率维度: %d\n', n_elements);
fprintf('电压范围: [%.6f, %.6f] V\n', min(voltages(:)), max(voltages(:)));
fprintf('电导率范围: [%.4f, %.4f] S/m\n', min(conductivities(:)), max(conductivities(:)));
fprintf('包含物半径范围: [%.1f, %.1f] mm\n', ...
    min(inclusion_params(:, 3))*1000, max(inclusion_params(:, 3))*1000);

%% ==================== 可视化样本 ====================
fprintf('\n生成可视化...\n');

figure('Position', [100, 100, 1400, 400], 'Name', 'EIDORS数据集样本');

% 选择3个样本展示
sample_indices = [1, randi(config.n_samples), randi(config.n_samples)];

for k = 1:3
    idx = sample_indices(k);

    % 电导率分布
    subplot(2, 3, k);
    patch('Faces', elements, 'Vertices', nodes, ...
          'FaceVertexCData', conductivities(idx, :), ...
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
saveas(gcf, 'eidors_samples_visualization.png');
fprintf('可视化已保存到: eidors_samples_visualization.png\n');

fprintf('\n====================================\n');
fprintf('全部完成！\n');
fprintf('====================================\n');
fprintf('输出文件:\n');
fprintf('  - 数据集: %s\n', config.output_path);
fprintf('  - 可视化: eidors_samples_visualization.png\n');
fprintf('\n下一步: 使用 convert_eidors_data.py 将MAT文件转换为PyTorch格式\n');
