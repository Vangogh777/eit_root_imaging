"""
ConvSpatialEIT 网络架构图绘制脚本 (现代风格)
生成精美的架构图用于论文和展示
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(1, 1, figsize=(20, 12), facecolor='#0f172a')
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_facecolor('#0f172a')

# 颜色定义 - 现代配色
colors = {
    'input': '#334155',
    'encoder': '#3b82f6',
    'blc': '#f59e0b',
    'fusion': '#10b981',
    'backbone': '#ec4899',
    'output': '#8b5cf6',
    'data': '#22d3ee',
    'text': '#e2e8f0',
    'text_dim': '#94a3b8',
    'bg': '#1e293b',
    'border': '#475569',
}

def draw_module(ax, x, y, w, h, color, title, lines, alpha=0.15):
    """绘制模块框 - 现代风格"""
    # 背景渐变效果 (通过多层实现)
    for i in range(5):
        alpha_i = alpha * (1 - i * 0.15)
        offset = i * 0.02
        box = FancyBboxPatch(
            (x - offset, y - offset), w + offset*2, h + offset*2,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            facecolor=color, edgecolor='none', alpha=alpha_i,
            linewidth=0
        )
        ax.add_patch(box)

    # 主边框
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.3",
        facecolor='none', edgecolor=color, linewidth=2.5, alpha=0.8
    )
    ax.add_patch(box)

    # 标题
    ax.text(x + w/2, y + h - 0.35, title,
            ha='center', va='top', fontsize=13, fontweight='bold',
            color=colors['text'])

    # 分隔线
    ax.plot([x + 0.3, x + w - 0.3], [y + h - 0.55, y + h - 0.55],
            color=color, linewidth=1, alpha=0.4)

    # 内部文字
    line_y = y + h - 0.85
    for line in lines:
        ax.text(x + w/2, line_y, line,
                ha='center', va='top', fontsize=9.5,
                color=colors['text_dim'])
        line_y -= 0.38

def draw_data(ax, x, y, w, h, name, shape, highlight=False):
    """绘制数据块"""
    border_color = colors['output'] if highlight else colors['border']
    border_width = 2.5 if highlight else 1.5

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=colors['input'], edgecolor=border_color,
        linewidth=border_width, alpha=0.9
    )
    ax.add_patch(box)

    # 数据名称
    ax.text(x + w/2, y + h*0.65, name,
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=colors['data'])

    # 维度
    ax.text(x + w/2, y + h*0.35, shape,
            ha='center', va='center', fontsize=9,
            color=colors['text_dim'])

def arrow(ax, x1, y1, x2, y2, color='#64748b', style='->', animated=False):
    """绘制箭头"""
    # 发光效果
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color, lw=4, alpha=0.2,
                    connectionstyle='arc3,rad=0'
                ))
    # 主箭头
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color, lw=1.8, alpha=0.8,
                    connectionstyle='arc3,rad=0'
                ))

# ==================== 标题 ====================
ax.text(10, 11.5, 'ConvSpatialEIT Architecture',
        ha='center', va='center', fontsize=26, fontweight='bold',
        color=colors['text'])
ax.text(10, 10.9, 'Spatial-Frequency Shared Network with Graph Neural Refinement',
        ha='center', va='center', fontsize=12,
        color=colors['text_dim'])

# ==================== 输入 ====================
draw_data(ax, 0.5, 4.5, 1.8, 1.0, 'V', '(B, 6, 208)')
ax.text(1.4, 4.2, 'Multi-freq', ha='center', fontsize=8, color=colors['text_dim'])

# ==================== FreqCrossAttention ====================
draw_module(ax, 3, 2.5, 2.8, 2.2, colors['encoder'], 'FreqCrossAttn',
            ['MultiheadAttention',
             'n_heads = 8',
             'LayerNorm',
             'GELU activation'])

# ==================== ConvEncoder ====================
draw_module(ax, 3, 5.0, 2.8, 1.8, colors['encoder'], 'ConvEncoder',
            ['Conv1d × 4 layers',
             'hidden_dim = 256',
             'BatchNorm + GELU'])

# ==================== 中间数据 h ====================
draw_data(ax, 6.5, 4.5, 1.5, 0.9, 'h', '(B, 6, 256)')

# ==================== GridSampler + PE ====================
draw_module(ax, 8.5, 2.0, 3.0, 2.5, colors['blc'], 'GridSampler + PE',
            ['Sample to mesh nodes',
             'Bilinear interpolation',
             'Fourier Position Enc',
             'dim_pe = 35',
             'Concat to features'])

# ==================== 中间数据 x_nodes ====================
draw_data(ax, 12.0, 2.8, 1.6, 0.9, 'x_nodes', '(B, N, D)')

# ==================== SimpleGNN × 4 ====================
draw_module(ax, 8.5, 5.0, 3.0, 2.5, colors['fusion'], 'SimpleGNN × 4',
            ['EdgeConv MessagePassing',
             'edge_net: MLP(→32→1)',
             'Softmax attention agg',
             'Residual connection',
             'LayerNorm'])

# ==================== 中间数据 h_gnn ====================
draw_data(ax, 12.0, 5.8, 1.6, 0.9, 'h_gnn', '(B, N, D)')

# ==================== OutputHead ====================
draw_module(ax, 14.2, 3.5, 2.5, 2.5, colors['output'], 'OutputHead',
            ['MLP: 256→128→64',
             '→ n_elems',
             'LayerNorm',
             'Sigmoid activation',
             'Scale to [0.01, 0.05]'])

# ==================== 输出 ====================
draw_data(ax, 17.2, 4.3, 1.8, 1.0, 'σ', '(B, 11466)', highlight=True)
ax.text(18.1, 4.0, 'Conductivity', ha='center', fontsize=8, color=colors['text_dim'])

# ==================== 绘制箭头 ====================
# Input → FreqCrossAttn
arrow(ax, 2.3, 5.0, 3.0, 4.2, colors['encoder'])

# Input → ConvEncoder
arrow(ax, 2.3, 5.0, 3.0, 5.9, colors['encoder'])

# FreqCrossAttn → h
arrow(ax, 5.8, 3.6, 6.5, 4.95, colors['encoder'])

# ConvEncoder → h
arrow(ax, 5.8, 5.9, 6.5, 5.0, colors['encoder'])

# h → GridSampler
arrow(ax, 8.0, 4.95, 8.5, 3.8, colors['blc'])

# h → SimpleGNN
arrow(ax, 8.0, 5.0, 8.5, 6.25, colors['fusion'])

# GridSampler → x_nodes
arrow(ax, 11.5, 3.25, 12.0, 3.25, colors['blc'])

# x_nodes → SimpleGNN (feedback)
ax.annotate('', xy=(12.8, 5.8), xytext=(12.8, 3.7),
            arrowprops=dict(arrowstyle='->', color=colors['text_dim'],
                           lw=1.5, ls='--', alpha=0.5))

# SimpleGNN → h_gnn
arrow(ax, 11.5, 6.25, 12.0, 6.25, colors['fusion'])

# x_nodes → OutputHead
arrow(ax, 13.6, 3.5, 14.2, 4.5, colors['output'])

# h_gnn → OutputHead
arrow(ax, 13.6, 6.0, 14.2, 5.5, colors['output'])

# OutputHead → Output
arrow(ax, 16.7, 4.8, 17.2, 4.8, colors['output'])

# ==================== 参数信息框 ====================
info_box = FancyBboxPatch(
    (0.5, 0.3), 19, 1.4,
    boxstyle="round,pad=0.02,rounding_size=0.2",
    facecolor=colors['input'], edgecolor=colors['border'],
    linewidth=1, alpha=0.6
)
ax.add_patch(info_box)

ax.text(10, 1.5, 'Model Parameters', ha='center', fontsize=12,
        fontweight='bold', color=colors['text'])

# 参数列
params = [
    ('Input', ['B: batch size', '6 frequencies', '208 measurements']),
    ('Mesh', ['N: 5859 nodes', 'E: 11466 elements', '147260 edges']),
    ('Hidden', ['D: 256 dim', 'n_heads: 8', '4 GNN layers']),
    ('Output', ['σ: conductivity', 'range: [0.01, 0.05] S/m', '']),
    ('Params', ['~6.1M total', 'FP16: ~12MB', 'FP32: ~24MB']),
]

for i, (title, items) in enumerate(params):
    x_pos = 2 + i * 3.8
    ax.text(x_pos, 1.15, title, fontsize=10, fontweight='bold',
            color=colors['data'])
    for j, item in enumerate(items):
        if item:
            ax.text(x_pos, 0.95 - j*0.2, item, fontsize=8,
                    color=colors['text_dim'])

# ==================== 图例 ====================
legend_items = [
    (colors['encoder'], 'Frequency Encoding'),
    (colors['blc'], 'Spatial Sampling'),
    (colors['fusion'], 'Graph Neural Net'),
    (colors['output'], 'Output Projection'),
]

for i, (color, label) in enumerate(legend_items):
    x = 0.5 + i * 4.8
    rect = Rectangle((x, 11.0), 0.8, 0.15, facecolor=color, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + 1.0, 11.08, label, fontsize=9, color=colors['text_dim'], va='center')

plt.tight_layout()

# 保存
output_dir = '/home/ubuntu/EIT/eit_root_imaging/docs'
plt.savefig(f'{output_dir}/architecture_modern.png',
            dpi=200, bbox_inches='tight', facecolor='#0f172a',
            pad_inches=0.2)
plt.savefig(f'{output_dir}/architecture_modern.pdf',
            bbox_inches='tight', facecolor='#0f172a',
            pad_inches=0.2)

print(f"✅ 架构图已保存:")
print(f"   PNG: {output_dir}/architecture_modern.png")
print(f"   PDF: {output_dir}/architecture_modern.pdf")
