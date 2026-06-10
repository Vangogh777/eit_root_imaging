"""
SF-SBLC 网络架构图绘制脚本 (改进版)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

# 使用英文字体避免中文问题
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')

# 颜色定义
colors = {
    'input': '#FFF9C4',
    'encoder': '#BBDEFB',
    'blc': '#FFE0B2',
    'fusion': '#C8E6C9',
    'backbone': '#F8BBD0',
    'output': '#FFCDD2',
    'attention': '#E1BEE7',
    'data': '#E0E0E0',
}

def draw_module(ax, x, y, w, h, color, title, lines, border_color='#333'):
    """绘制模块框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                         facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.4, title, ha='center', va='top', fontsize=12, fontweight='bold')

    # 绘制内部线条
    line_y = y + h - 0.9
    for line in lines:
        ax.text(x + w/2, line_y, line, ha='center', va='top', fontsize=9, color='#444')
        line_y -= 0.45

def draw_data(ax, x, y, w, h, name, shape):
    """绘制数据块"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=colors['data'], edgecolor='#666', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.65, name, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(x + w/2, y + h*0.35, shape, ha='center', va='center', fontsize=9, color='#555')

def arrow(ax, x1, y1, x2, y2):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8),
                annotation_clip=False)

# ==================== 标题 ====================
ax.text(9, 13.5, 'SF-SBLC Network Architecture', ha='center', va='center', fontsize=20, fontweight='bold')
ax.text(9, 12.9, 'Spatial-Frequency Shared and Base Layer Correction', ha='center', va='center', fontsize=13, color='#666')

# ==================== 输入 ====================
draw_data(ax, 1, 10.5, 3, 1.2, 'Voltage V', '(B, 6, 256)')
ax.text(2.5, 10.2, 'Multi-frequency measurements', ha='center', fontsize=9, color='#888')

# ==================== SharedEncoder ====================
draw_module(ax, 5.5, 9.5, 4.5, 3, colors['encoder'], 'SharedEncoder',
            ['Input Projection: Linear -> GELU',
             'Shared 1D Conv x4 layers',
             'BatchNorm + GELU + Dropout',
             'Cross-Frequency Attention',
             '(MultiheadAttention, 8 heads)',
             'Output Projection'])

# ==================== 中间数据 h ====================
draw_data(ax, 11, 10.5, 2.5, 1, 'h', '(B, 6, 512)')

# ==================== BLC ====================
draw_module(ax, 1, 5.5, 4.5, 3.2, colors['blc'], 'BLC (Base Layer Correction)',
            ['Global Pooling: AdaptiveAvgPool1d',
             'Base Extractor: Linear -> Tanh',
             'Correction Predictor: MLP',
             'Frequency-specific Correction',
             'Gate: Sigmoid',
             'Output: h + delta * gate'])

# ==================== h_corrected ====================
draw_data(ax, 6.5, 6.8, 2.5, 0.9, 'h_corrected', '(B, 6, 512)')

# ==================== FrequencyFusion ====================
draw_module(ax, 10, 5.5, 4.5, 3.2, colors['fusion'], 'FrequencyFusionDecoder',
            ['Frequency Self-Attention',
             'Feature Concat: (F x D) -> D',
             'Fusion MLP: 3072 -> 1024 -> 512',
             'CBAM Attention Enhancement',
             'Dual-path Output:',
             '  main + residual * sigmoid(gate)'])

# ==================== sigma_fused ====================
draw_data(ax, 15, 6.8, 2.5, 0.9, 'sigma_fused', '(B, 1500)')

# ==================== ResNetBackbone ====================
draw_module(ax, 6, 1.5, 5, 3, colors['backbone'], 'ResNetBackbone',
            ['Input Projection: Linear',
             'Residual Block x8:',
             '  Linear -> GELU -> Dropout',
             '  -> Linear -> LayerNorm',
             '  Residual Connection',
             'Output Head: 512 -> 256 -> 1500'])

# ==================== sigma_refined ====================
draw_data(ax, 12, 2.8, 2.5, 0.9, 'sigma_refined', '(B, 1500)')

# ==================== 融合节点 ====================
fusion_circle = Circle((14.5, 2), 0.4, facecolor='#78909C', edgecolor='#37474F', linewidth=2)
ax.add_patch(fusion_circle)
ax.text(14.5, 2, '+', ha='center', va='center', fontsize=18, fontweight='bold', color='white')

# ==================== 输出 ====================
draw_data(ax, 16, 1.3, 2, 1.2, 'sigma', '(B, 1500)')
ax.text(17, 1.0, 'Conductivity', ha='center', fontsize=9, color='#888')

# ==================== 绘制箭头 ====================
# 输入 -> SharedEncoder
arrow(ax, 4, 11.1, 5.5, 11.1)

# SharedEncoder -> h
arrow(ax, 10, 11, 11, 11)

# h -> BLC (分叉)
arrow(ax, 12.25, 10.5, 12.25, 9.5)
ax.plot([12.25, 3.25], [9.5, 9.5], color='#333', lw=1.8)
arrow(ax, 3.25, 9.5, 3.25, 8.7)

# h -> FrequencyFusion
arrow(ax, 13.5, 10.5, 13.5, 8.7)

# BLC -> h_corrected
arrow(ax, 5.5, 7.1, 6.5, 7.25)

# h_corrected -> FrequencyFusion
arrow(ax, 9, 7.25, 10, 7.25)

# h_corrected -> ResNetBackbone (pool路径)
ax.plot([7.75, 7.75], [6.8, 4.5], color='#333', lw=1.8)
arrow(ax, 7.75, 4.5, 7.75, 4.5)
ax.text(8.2, 5.5, 'mean(F)', fontsize=9, color='#888')

# FrequencyFusion -> sigma_fused
arrow(ax, 14.5, 6.8, 15, 7.25)

# ResNetBackbone -> sigma_refined
arrow(ax, 11, 3, 12, 3.25)

# sigma_fused -> fusion
arrow(ax, 16.25, 6.8, 16.25, 2)
ax.plot([16.25, 14.9], [2, 2], color='#333', lw=1.8)

# sigma_refined -> fusion
arrow(ax, 14.5, 2.8, 14.5, 2.4)

# fusion -> output
arrow(ax, 14.9, 2, 16, 1.9)

# ==================== 全局残差连接 ====================
ax.plot([3.25, 3.25], [5.5, 1.5], color='#888', lw=1.5, ls='--')
ax.plot([3.25, 14.1], [1.5, 1.5], color='#888', lw=1.5, ls='--')
arrow(ax, 14.1, 1.5, 14.1, 1.6)
ax.text(8.5, 1.2, 'Global Residual (base_map)', fontsize=9, color='#888', ha='center', style='italic')

# ==================== 图例 ====================
legend_x = 0.5
legend_y = 0.3

boxes = [
    (colors['encoder'], 'Encoder'),
    (colors['blc'], 'Correction'),
    (colors['fusion'], 'Decoder'),
    (colors['backbone'], 'Backbone'),
    (colors['data'], 'Data'),
]

for i, (color, label) in enumerate(boxes):
    x = legend_x + i * 3.2
    ax.add_patch(FancyBboxPatch((x, legend_y), 0.6, 0.4, boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor='#333', linewidth=1))
    ax.text(x + 1.2, legend_y + 0.2, label, fontsize=10, va='center')

# ==================== 参数说明 ====================
info_x = 15
info_y = 10.5
info_box = FancyBboxPatch((info_x, info_y), 2.8, 2.5, boxstyle="round,pad=0.02",
                          facecolor='#FAFAFA', edgecolor='#BDBDBD', linewidth=1)
ax.add_patch(info_box)
ax.text(info_x + 1.4, info_y + 2.3, 'Parameters', fontsize=11, fontweight='bold', ha='center')
ax.text(info_x + 0.2, info_y + 1.9, 'B: Batch size', fontsize=9)
ax.text(info_x + 0.2, info_y + 1.5, 'F: 6 frequencies', fontsize=9)
ax.text(info_x + 0.2, info_y + 1.1, 'M: 256 measurements', fontsize=9)
ax.text(info_x + 0.2, info_y + 0.7, 'D: 512 hidden dim', fontsize=9)
ax.text(info_x + 0.2, info_y + 0.3, 'E: ~1500 elements', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/vangogh/455/王楠师姐/EITProject/eit_root_imaging/docs/network_architecture.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/vangogh/455/王楠师姐/EITProject/eit_root_imaging/docs/network_architecture.pdf',
            bbox_inches='tight', facecolor='white')
print("Done!")
