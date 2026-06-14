%% EIDORS 测试脚本 - 生成10个样本并可视化
% 包含物: 随机大小、随机位置
% 电导率: 背景0.01 S/m, 包含物0.05 S/m (5倍)

clear; close all; clc;

%% 初始化随机种子 - 确保每次运行都不同
rng('shuffle');  % 使用当前时间作为种子
fprintf('随机种子已初始化\n');

%% 路径
addpath('D:\eidors-v3.11ceshi\eidors-v3.11\eidors');
cd('D:\EITProject');

%% 参数
n_electrodes = 16;
radius_bucket = 0.1;           % 桶半径 10cm (直径20cm)
sigma_bg = 0.01;               % 背景电导率 S/m
sigma_inc = 0.05;              % 包含物电导率 S/m (背景的5倍)

% 包含物尺寸范围 (直径)
min_diameter = 0.02;           % 最小直径 2cm
max_diameter = 0.06;           % 最大直径 6cm

%% 创建网格
fprintf('创建网格...\n');
m = mk_common_model('c2c', n_electrodes);
fwd = m.fwd_model;

nodes = fwd.nodes;
elems = fwd.elems;
n_elem = size(elems, 1);

% 单元中心
centers = zeros(n_elem, 2);
for e = 1:n_elem
    centers(e,:) = mean(nodes(elems(e,:),:), 1);
end

% 激励模式
fwd.stimulation = mk_stim_patterns(n_electrodes, 1, '{ad}', '{ad}');

fprintf('网格: %d 单元\n', n_elem);

%% 生成10个样本
n_samples = 10;
figure('Position', [50, 50, 1800, 900]);

fprintf('\n===== 样本参数 (每次运行应该不同) =====\n');
fprintf('%-6s %-10s %-14s %-10s\n', '编号', '直径(cm)', '中心位置(mm)', '覆盖单元');

for s = 1:n_samples
    % 随机包含物大小 (直径2-6cm)
    diameter = min_diameter + rand() * (max_diameter - min_diameter);
    r_inc = diameter / 2;  % 半径

    % 随机包含物位置 (确保完全在区域内)
    max_r = radius_bucket - r_inc - 0.005;  % 留5mm边距
    r_pos = sqrt(rand()) * max_r;           % sqrt保证均匀分布
    theta_pos = rand() * 2 * pi;
    cx = r_pos * cos(theta_pos);
    cy = r_pos * sin(theta_pos);

    % 电导率分布
    sigma = sigma_bg * ones(n_elem, 1);
    dist = sqrt((centers(:,1)-cx).^2 + (centers(:,2)-cy).^2);
    sigma(dist < r_inc) = sigma_inc;

    % 正向求解
    img = mk_image(fwd, sigma);
    data = fwd_solve(img);
    v = data.meas';

    % 可视化电导率
    subplot(2, 5, s);
    patch('Faces', double(elems), 'Vertices', nodes*100, ...
          'FaceVertexCData', sigma, ...
          'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.2);
    axis equal; axis off;
    caxis([sigma_bg*0.9, sigma_inc*1.1]);
    colormap jet;
    colorbar;

    % 标记包含物边界 (红色圆圈)
    hold on;
    th = linspace(0,2*pi,50);
    plot(cx*100 + r_inc*100*cos(th), cy*100 + r_inc*100*sin(th), 'r-', 'LineWidth', 2);
    % 标记包含物中心 (红色点)
    plot(cx*100, cy*100, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
    hold off;

    n_covered = sum(dist < r_inc);
    title(sprintf('#%d: D=%.1fcm', s, diameter*100));

    % 打印信息
    fprintf('%-6d %-10.1f (%.1f, %.1f)   %-10d\n', ...
        s, diameter*100, cx*1000, cy*1000, n_covered);
end

sgtitle(sprintf('EIDORS - 背景%.3f S/m, 包含物%.3f S/m (5倍)', sigma_bg, sigma_inc));
saveas(gcf, 'D:\EITProject\test_10samples.png');
fprintf('\n图片保存到: D:\EITProject\test_10samples.png\n');
fprintf('\n提示: 红色圆圈是包含物边界，红点是包含物中心\n');