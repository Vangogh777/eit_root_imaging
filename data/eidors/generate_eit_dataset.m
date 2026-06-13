%% EIDORS EIT Training Dataset Generator
% 生成用于神经网络的EIT训练数据集
% 输入：边界电压测量 (1×256)
% 输出：电导率分布
%
% 作者：Claude Code Assistant
% 日期：2025-06-13

clear; close all; clc;

%% 添加EIDORS路径 (根据你的安装位置修改)
addpath('/path/to/eidors');  % 修改为你的EIDORS安装路径
eidors_on;

%% 配置参数
config = struct();

% 网格参数
config.n_electrodes = 16;       % 电极数量
config.radius = 0.1;            % 圆形区域半径 (m)
config.n_stim = 16;             % 激励模式数量
config.n_meas_per_stim = 16;    % 每个激励模式的测量数
config.n_meas_total = 256;      % 总测量数 (16 × 16)

% 电导率参数
config.sigma_background = 0.01; % 背景(土壤)电导率 S/m
config.sigma_inclusion = 0.05;  % 包含物(根系)电导率 S/m
config.sigma_min = 0.005;       % 最小电导率
config.sigma_max = 0.1;         % 最大电导率

% 数据集参数
config.n_samples = 10000;       % 样本数量
config.n_inclusions_max = 5;    % 最大包含物数量
config.inclusion_type = 'random'; % 'random', 'circular', 'ellipse', 'root'

% 输出路径
config.output_path = './data/eit_eidors_dataset.mat';

%% 创建EIDORS正向模型
fprintf('创建EIDORS正向模型...\n');

% 创建2D圆形FEM模型
fwd_model = mk_common_model('c2c', config.n_electrodes);
fwd_model = fwd_model.fwd_model;

% 设置激励模式：相邻激励 (adjacent pattern)
stimulation = mk_stim_patterns(config.n_electrodes, 1, '{ad}', '{ad}');
fwd_model.stimulation = stimulation;

% 设置电极参数
fwd_model.n_electrodes = config.n_electrodes;
fwd_model.electrode = [];
for i = 1:config.n_electrodes
    angle = 2*pi*(i-1)/config.n_electrodes;
    fwd_model.electrode(i).nodes = i;
    fwd_model.electrode(i).z_contact = 0.01;  % 接触阻抗
end

% 计算测量数量
config.n_meas_total = calc_n_meas(fwd_model);
fprintf('总测量数: %d\n', config.n_meas_total);

% 存储节点和单元信息
nodes = fwd_model.nodes;
elements = fwd_model.elems;
n_nodes = size(nodes, 1);
n_elements = size(elements, 1);
fprintf('节点数: %d, 单元数: %d\n', n_nodes, n_elements);

%% 初始化数据存储
voltages = zeros(config.n_samples, config.n_meas_total);
conductivities = zeros(config.n_samples, n_elements);
inclusion_masks = zeros(config.n_samples, n_elements);  % 包含物掩码

%% 生成随机包含物的辅助函数
generate_inclusion = @(sigma_bg, sigma_inc, type) create_random_inclusion(...
    fwd_model, sigma_bg, sigma_inc, config, type);

%% 主数据生成循环
fprintf('开始生成 %d 个样本...\n', config.n_samples);
tic;

for i = 1:config.n_samples
    % 进度显示
    if mod(i, 1000) == 0
        fprintf('进度: %d/%d (%.1f%%)\n', i, config.n_samples, 100*i/config.n_samples);
    end

    % 生成随机电导率分布
    [sigma_dist, mask] = generate_inclusion(config.sigma_background, ...
        config.sigma_inclusion, config.inclusion_type);

    % 正向求解：计算边界电压
    img = mk_image(fwd_model, sigma_dist);
    data = fwd_solve(img);

    % 存储数据
    voltages(i, :) = data.meas';
    conductivities(i, :) = sigma_dist';
    inclusion_masks(i, :) = mask';
end

elapsed_time = toc;
fprintf('数据生成完成！耗时: %.2f 秒\n', elapsed_time);

%% 添加噪声（可选）
noise_level = 0.001;  % 0.1% 高斯噪声
voltages_noisy = voltages + noise_level * randn(size(voltages)) .* abs(voltages);

%% 保存数据集
fprintf('保存数据集到: %s\n', config.output_path);

% 确保输出目录存在
[~, ~, ~] = mkdir(fileparts(config.output_path));

% 保存为MAT文件
save(config.output_path, 'voltages', 'voltages_noisy', 'conductivities', ...
    'inclusion_masks', 'nodes', 'elements', 'config', 'fwd_model', '-v7.3');

fprintf('数据集保存完成！\n');

%% 数据统计
fprintf('\n=== 数据集统计 ===\n');
fprintf('样本数量: %d\n', config.n_samples);
fprintf('电压维度: %d\n', config.n_meas_total);
fprintf('电导率维度: %d\n', n_elements);
fprintf('电压范围: [%.6f, %.6f]\n', min(voltages(:)), max(voltages(:)));
fprintf('电导率范围: [%.4f, %.4f] S/m\n', min(conductivities(:)), max(conductivities(:)));

%% 可视化样本
figure('Position', [100, 100, 1200, 400]);

% 显示一个样本的电导率分布
subplot(1, 3, 1);
sample_idx = randi(config.n_samples);
show_fem(fwd_model, conductivities(sample_idx, :));
title('电导率分布');
colorbar;
colormap(jet);

% 显示对应的电压测量
subplot(1, 3, 2);
plot(voltages(sample_idx, :), 'b-o', 'LineWidth', 1.5);
title('边界电压测量');
xlabel('测量索引');
ylabel('电压 (V)');
grid on;

% 显示电导率直方图
subplot(1, 3, 3);
histogram(conductivities(:), 50);
title('电导率分布直方图');
xlabel('电导率 (S/m)');
ylabel('频数');

sgtitle(sprintf('样本 #%d 示例', sample_idx));

%% 辅助函数定义

function [sigma, mask] = create_random_inclusion(fwd_model, sigma_bg, sigma_inc, config, type)
%CREATE_RANDOM_INCLUSION 创建随机包含物的电导率分布
%
% 输入:
%   fwd_model   - EIDORS正向模型
%   sigma_bg    - 背景电导率
%   sigma_inc   - 包含物电导率
%   config      - 配置参数
%   type        - 包含物类型: 'random', 'circular', 'ellipse', 'root'
%
% 输出:
%   sigma       - 电导率分布向量
%   mask        - 包含物掩码 (1=包含物, 0=背景)

    n_elements = size(fwd_model.elems, 1);
    sigma = sigma_bg * ones(n_elements, 1);
    mask = zeros(n_elements, 1);

    % 计算单元中心坐标
    nodes = fwd_model.nodes;
    elements = fwd_model.elems;

    elem_centers = zeros(n_elements, 2);
    for e = 1:n_elements
        elem_centers(e, :) = mean(nodes(elements(e, :), :), 1);
    end

    % 根据类型生成包含物
    switch type
        case 'random'
            % 随机圆形包含物
            n_inclusions = randi([1, config.n_inclusions_max]);

            for k = 1:n_inclusions
                % 随机位置和大小
                r = rand() * config.radius * 0.7;
                theta = rand() * 2 * pi;
                cx = r * cos(theta);
                cy = r * sin(theta);

                radius = rand() * config.radius * 0.15 + config.radius * 0.05;

                % 找到包含物内的单元
                dist = sqrt((elem_centers(:, 1) - cx).^2 + ...
                           (elem_centers(:, 2) - cy).^2);
                in_inclusion = dist < radius;

                % 随机电导率变化
                sigma_val = sigma_inc * (0.8 + 0.4 * rand());
                sigma(in_inclusion) = sigma_val;
                mask(in_inclusion) = 1;
            end

        case 'circular'
            % 单个圆形包含物
            r = rand() * config.radius * 0.6;
            theta = rand() * 2 * pi;
            cx = r * cos(theta);
            cy = r * sin(theta);
            radius = rand() * config.radius * 0.2 + config.radius * 0.1;

            dist = sqrt((elem_centers(:, 1) - cx).^2 + ...
                       (elem_centers(:, 2) - cy).^2);
            in_inclusion = dist < radius;

            sigma(in_inclusion) = sigma_inc;
            mask(in_inclusion) = 1;

        case 'ellipse'
            % 椭圆形包含物
            r = rand() * config.radius * 0.5;
            theta = rand() * 2 * pi;
            cx = r * cos(theta);
            cy = r * sin(theta);

            a = rand() * config.radius * 0.2 + config.radius * 0.05;  % 长轴
            b = rand() * config.radius * 0.15 + config.radius * 0.03; % 短轴
            phi = rand() * pi;  % 旋转角度

            % 旋转坐标
            x_rot = (elem_centers(:, 1) - cx) * cos(phi) + ...
                    (elem_centers(:, 2) - cy) * sin(phi);
            y_rot = -(elem_centers(:, 1) - cx) * sin(phi) + ...
                     (elem_centers(:, 2) - cy) * cos(phi);

            in_inclusion = (x_rot.^2 / a^2 + y_rot.^2 / b^2) < 1;

            sigma(in_inclusion) = sigma_inc;
            mask(in_inclusion) = 1;

        case 'root'
            % 模拟根系结构
            n_roots = randi([2, 5]);

            % 主根
            main_root_angle = rand() * 2 * pi;
            main_root_length = rand() * config.radius * 0.6 + config.radius * 0.2;
            main_root_width = config.radius * 0.03;

            % 沿主根路径设置电导率
            for t = 0:main_root_width:main_root_length
                x = t * cos(main_root_angle);
                y = t * sin(main_root_angle);

                dist = sqrt((elem_centers(:, 1) - x).^2 + ...
                           (elem_centers(:, 2) - y).^2);
                in_root = dist < main_root_width;

                sigma(in_root) = sigma_inc;
                mask(in_root) = 1;
            end

            % 侧根
            for k = 1:n_roots-1
                branch_pos = rand() * main_root_length * 0.8;
                branch_angle = main_root_angle + (rand() - 0.5) * pi/2;
                branch_length = rand() * config.radius * 0.3 + config.radius * 0.1;
                branch_width = main_root_width * 0.6;

                for t = 0:branch_width:branch_length
                    x = branch_pos * cos(main_root_angle) + t * cos(branch_angle);
                    y = branch_pos * sin(main_root_angle) + t * sin(branch_angle);

                    dist = sqrt((elem_centers(:, 1) - x).^2 + ...
                               (elem_centers(:, 2) - y).^2);
                    in_root = dist < branch_width;

                    sigma(in_root) = sigma_inc * (0.8 + 0.2 * rand());
                    mask(in_root) = 1;
                end
            end
    end
end

function n = calc_n_meas(fwd_model)
%CALC_N_MEAS 计算总测量数量
    n = 0;
    for i = 1:length(fwd_model.stimulation)
        n = n + length(fwd_model.stimulation(i).meas_pattern);
    end
end

function show_fem(fwd_model, sigma)
%SHOW_FEM 显示FEM网格上的电导率分布

    nodes = fwd_model.nodes;
    elements = fwd_model.elems;

    % 创建三角形补片
    patch('Faces', elements, 'Vertices', nodes, ...
          'FaceVertexCData', sigma, ...
          'FaceColor', 'interp', 'EdgeColor', 'none');
    axis equal;
    axis off;
end
