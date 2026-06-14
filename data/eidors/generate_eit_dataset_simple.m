%% EIDORS EIT 数据生成脚本 - 最终修复版
% 桶直径: 20cm, 包含物直径: 2-6cm
%
% 使用方法: 复制此脚本到 D:\EITProject\ 然后运行

clear; close all; clc;

%% ==================== 路径配置 ====================
eidors_path = 'D:\eidors-v3.11ceshi\eidors-v3.11\eidors';
addpath(eidors_path);
cd('D:\EITProject');

%% ==================== 参数配置 ====================
config = struct();
config.n_electrodes = 16;           % 电极数量
config.radius = 0.1;                % 桶半径 10cm (直径20cm)
config.sigma_background = 0.01;     % 背景电导率 S/m
config.sigma_inclusion = 0.05;      % 包含物电导率 S/m
config.n_samples = 10000;           % 样本数量

% 包含物尺寸: 直径2-6cm (半径1-3cm)
config.inclusion_radius_min = 0.01; % 最小半径 1cm
config.inclusion_radius_max = 0.03; % 最大半径 3cm

%% ==================== 创建细网格 ====================
fprintf('====================================\n');
fprintf('创建EIDORS细网格正向模型\n');
fprintf('====================================\n');

% 尝试不同级别的网格密度
% c2c = coarse, c2c2 = medium, c2c4 = fine, c2c8 = very fine
model_names = {'c2c8', 'c2c4', 'c2c2', 'c2c'};
fwd_model = [];

for i = 1:length(model_names)
    try
        fprintf('尝试: mk_common_model(''%s'', %d)...\n', model_names{i}, config.n_electrodes);
        model_struct = mk_common_model(model_names{i}, config.n_electrodes);
        fwd_model = model_struct.fwd_model;
        n_elem = size(fwd_model.elems, 1);
        fprintf('  成功! 单元数: %d\n', n_elem);
        if n_elem >= 1000
            fprintf('  网格足够细，采用此模型\n');
            break;
        end
    catch ME
        fprintf('  失败: %s\n', ME.message);
    end
end

if isempty(fwd_model)
    error('无法创建EIDORS网格模型');
end

% 获取网格信息
nodes = fwd_model.nodes;
elements = fwd_model.elems;
n_nodes = size(nodes, 1);
n_elements = size(elements, 1);

fprintf('\n最终网格信息:\n');
fprintf('  节点数: %d\n', n_nodes);
fprintf('  单元数: %d\n', n_elements);

% 计算单元中心坐标
elem_centers = zeros(n_elements, 2);
for e = 1:n_elements
    elem_nodes = elements(e, :);
    elem_centers(e, :) = mean(nodes(elem_nodes, :), 1);
end

% 计算典型单元尺寸
elem_areas = zeros(n_elements, 1);
for e = 1:n_elements
    v1 = nodes(elements(e,1), :) - nodes(elements(e,3), :);
    v2 = nodes(elements(e,2), :) - nodes(elements(e,3), :);
    elem_areas(e) = abs(v1(1)*v2(2) - v1(2)*v2(1)) / 2;
end
typical_elem_size = sqrt(mean(elem_areas));
fprintf('  典型单元尺寸: %.1f mm\n', typical_elem_size * 1000);

if typical_elem_size * 1000 > 15
    fprintf('\n警告: 网格仍然较粗，包含物可能不够精确\n');
end

%% ==================== 设置激励模式 ====================
stimulation = mk_stim_patterns(config.n_electrodes, 1, '{ad}', '{ad}');
fwd_model.stimulation = stimulation;

% 计算测量数
n_meas_total = 0;
for i = 1:length(stimulation)
    n_meas_total = n_meas_total + size(stimulation(i).meas_pattern, 1);
end
config.n_meas_total = n_meas_total;
fprintf('  总测量数: %d\n', n_meas_total);

%% ==================== 生成数据 ====================
fprintf('\n====================================\n');
fprintf('开始生成 %d 个样本...\n', config.n_samples);
fprintf('====================================\n');

voltages = zeros(config.n_samples, n_meas_total);
conductivities = zeros(config.n_samples, n_elements);
inclusion_params = zeros(config.n_samples, 4);  % [cx, cy, radius, sigma]

tic;
for i = 1:config.n_samples
    if mod(i, 1000) == 0 || i == 1
        fprintf('进度: %d/%d (%.1f%%)\n', i, config.n_samples, 100*i/config.n_samples);
    end

    % ===== 生成包含物 =====
    % 随机半径 (1cm - 3cm)
    radius = rand() * (config.inclusion_radius_max - config.inclusion_radius_min) + config.inclusion_radius_min;

    % 随机位置 (确保包含物完全在区域内)
    max_center_dist = config.radius - radius - 0.005;
    r_pos = sqrt(rand()) * max_center_dist;
    theta_pos = rand() * 2 * pi;
    cx = r_pos * cos(theta_pos);
    cy = r_pos * sin(theta_pos);

    % 随机电导率
    sigma_inc = config.sigma_inclusion * (0.8 + 0.4 * rand());

    % ===== 设置电导率分布 =====
    sigma = config.sigma_background * ones(n_elements, 1);

    % 计算每个单元中心到包含物中心的距离
    dist = sqrt((elem_centers(:, 1) - cx).^2 + (elem_centers(:, 2) - cy).^2);

    % 在包含物内的单元
    in_inclusion = dist < radius;
    sigma(in_inclusion) = sigma_inc;

    % 存储参数
    inclusion_params(i, :) = [cx, cy, radius, sigma_inc];

    % ===== 正向求解 =====
    img = mk_image(fwd_model, sigma);
    data = fwd_solve(img);

    voltages(i, :) = data.meas';
    conductivities(i, :) = sigma';
end

elapsed_time = toc;
fprintf('完成! 耗时: %.1f 秒\n', elapsed_time);

%% ==================== 数据统计 ====================
coverage_counts = sum(conductivities ~= config.sigma_background, 2);

fprintf('\n====================================\n');
fprintf('数据统计\n');
fprintf('====================================\n');
fprintf('样本数: %d\n', config.n_samples);
fprintf('电压维度: %d\n', n_meas_total);
fprintf('电导率维度: %d\n', n_elements);
fprintf('包含物直径: %.1f - %.1f cm\n', ...
    2*config.inclusion_radius_min*100, 2*config.inclusion_radius_max*100);
fprintf('包含物覆盖单元数:\n');
fprintf('  最小: %d\n', min(coverage_counts));
fprintf('  最大: %d\n', max(coverage_counts));
fprintf('  平均: %.1f\n', mean(coverage_counts));

%% ==================== 保存数据 ====================
output_path = 'D:\EITProject\eit_eidors_dataset.mat';
fprintf('\n保存到: %s\n', output_path);
save(output_path, 'voltages', 'conductivities', 'inclusion_params', ...
    'nodes', 'elements', 'elem_centers', 'config', 'fwd_model', '-v7.3');
fprintf('保存完成!\n');

%% ==================== 可视化 ====================
fprintf('\n生成可视化...\n');

figure('Position', [50, 50, 1600, 600]);

% 显示3个样本
[~, sort_idx] = sort(coverage_counts);
samples = [sort_idx(1), sort_idx(round(end/2)), sort_idx(end)];

for k = 1:3
    idx = samples(k);

    % 电导率分布 (使用trisurf)
    subplot(2, 3, k);
    h = trisurf(double(elements), nodes(:,1)*100, nodes(:,2)*100, conductivities(idx,:)');
    shading flat; axis equal; view(2);
    colorbar; colormap jet;
    caxis([config.sigma_background*0.9, config.sigma_inclusion*1.1]);
    title(sprintf('#%d: %d单元', idx, coverage_counts(idx)));

    % 电压
    subplot(2, 3, k+3);
    plot(voltages(idx, :), 'b-o', 'MarkerSize', 3);
    xlabel('测量'); ylabel('V');
    title(sprintf('直径=%.0fcm', 2*inclusion_params(idx,3)*100));
    grid on;
end

sgtitle(sprintf('网格=%d单元, 单元尺寸=%.1fmm', n_elements, typical_elem_size*1000));
saveas(gcf, 'D:\EITProject\visualization.png');
fprintf('可视化保存到: D:\EITProject\visualization.png\n');

fprintf('\n全部完成!\n');
