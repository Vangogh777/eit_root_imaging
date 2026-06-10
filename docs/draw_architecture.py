"""
SF-SBLC 网络架构图绘制脚本
===========================
绘制 EIT 植物根部无监督成像系统的网络架构
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.patheffects as path_effects
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# 颜色定义
colors = {
    'input': '#E8F4FD',
    'encoder': '#B8D4E8',
    'blc': '#FFE5B4',
    'fusion': '#C8E6C9',
    'backbone': '#F8BBD9',
    'output': '#FFCDD2',
    'attention': '#E1BEE7',
    'arrow': '#455A64',
}

def draw_box(ax, x, y, w, h, color, text, fontsize=10, bold=False):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
                         facecolor=color, edgecolor='#333333', linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, wrap=True)

def draw_arrow(ax, start, end, color='#455A64', style='->', lw=1.5):
    """绘制箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                annotation_clip=False)

def draw_data_block(ax, x, y, w, h, label, shape_text=''):
    """绘制数据块"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.65, label, ha='center', va='center', fontsize=9, fontweight='bold')
    if shape_text:
        ax.text(x + w/2, y + h*0.35, shape_text, ha='center', va='center', fontsize=8, color='#666')

# ==================== 标题 ====================
ax.text(8, 11.5, 'SF-SBLC 网络架构图', ha='center', va='center', fontsize=18, fontweight='bold')
ax.text(8, 11.0, 'Spatial-Frequency Shared and Base Layer Correction', ha='center', va='center', fontsize=12, color='#666')

# ==================== 输入层 ====================
draw_data_block(ax, 0.5, 8.5, 2.5, 1.2, '边界电压 V', '(B, 6, 256)')
ax.text(1.75, 8.0, '多频率测量', ha='center', fontsize=9, color='#888')

# ==================== SharedEncoder ====================
draw_box(ax, 4, 7.5, 3.5, 2.5, colors['encoder'], 'SharedEncoder\n共享编码器', fontsize=11, bold=True)

# 内部结构
ax.text(5.75, 9.3, '输入投影 Linear→GELU', fontsize=8, ha='center', color='#555')
ax.text(5.75, 8.8, '共享1D卷积 × 4层', fontsize=8, ha='center', color='#555')
ax.text(5.75, 8.3, 'BatchNorm + GELU', fontsize=8, ha='center', color='#555')

# 跨频率注意力
attn_box = FancyBboxPatch((4.2, 7.7), 1.8, 0.5, boxstyle="round,pad=0.02",
                          facecolor=colors['attention'], edgecolor='#7B1FA2', linewidth=1)
ax.add_patch(attn_box)
ax.text(5.1, 7.95, '跨频率注意力', fontsize=8, ha='center', va='center')

# 输出标注
draw_data_block(ax, 8, 8.2, 2, 0.8, 'h', '(B, 6, 512)')

# ==================== BLC 模块 ====================
draw_box(ax, 4, 4.5, 3.5, 2.5, colors['blc'], 'BLC\n基础层校正', fontsize=11, bold=True)

ax.text(5.75, 6.3, '全局池化 → 基础层提取', fontsize=8, ha='center', color='#555')
ax.text(5.75, 5.8, '校正预测器', fontsize=8, ha='center', color='#555')
ax.text(5.75, 5.3, '门控机制 (sigmoid)', fontsize=8, ha='center', color='#555')

# BLC输出
ax.text(5.75, 4.7, '残差校正: h + δ·gate', fontsize=8, ha='center', style='italic', color='#888')

# ==================== FrequencyFusion ====================
draw_box(ax, 8.5, 4.5, 3.5, 2.5, colors['fusion'], 'FrequencyFusion\n频率融合解码', fontsize=11, bold=True)

ax.text(10.25, 6.3, '频率自注意力', fontsize=8, ha='center', color='#555')
ax.text(10.25, 5.8, '特征级联 (F×D → D)', fontsize=8, ha='center', color='#555')
ax.text(10.25, 5.3, '融合投影 MLP', fontsize=8, ha='center', color='#555')

# CBAM注意力
cbam_box = FancyBboxPatch((8.7, 4.7), 1.8, 0.5, boxstyle="round,pad=0.02",
                          facecolor=colors['attention'], edgecolor='#7B1FA2', linewidth=1)
ax.add_patch(cbam_box)
ax.text(9.6, 4.95, 'CBAM注意力', fontsize=8, ha='center', va='center')

# ==================== ResNet Backbone ====================
draw_box(ax, 12.5, 4.5, 3, 2.5, colors['backbone'], 'ResNetBackbone\n残差骨干', fontsize=11, bold=True)

ax.text(14, 6.3, '输入投影 Linear', fontsize=8, ha='center', color='#555')
ax.text(14, 5.8, '残差块 × 8层', fontsize=8, ha='center', color='#555')
ax.text(14, 5.3, 'LayerNorm + GELU', fontsize=8, ha='center', color='#555')

# ==================== 中间数据流 ====================
# h_corrected
draw_data_block(ax, 8, 6.0, 2, 0.6, 'h_corrected', '(B, 6, 512)')

# h_pool
draw_data_block(ax, 10.5, 7.2, 1.8, 0.5, 'h_pool', '(B, 512)')

# sigma_fused
draw_data_block(ax, 10.5, 3.5, 1.8, 0.5, 'σ_fused', '(B, 1500)')

# sigma_refined
draw_data_block(ax, 13.5, 3.5, 1.8, 0.5, 'σ_refined', '(B, 1500)')

# ==================== 融合与输出 ====================
# 融合节点
fusion_node = Circle((12, 2.5), 0.3, facecolor='#90A4AE', edgecolor='#37474F', linewidth=1.5)
ax.add_patch(fusion_node)
ax.text(12, 2.5, '+', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

# 全局残差
ax.text(5.75, 3.8, 'base_map', fontsize=9, ha='center', color='#888')
ax.annotate('', xy=(11.7, 2.5), xytext=(5.75, 3.5),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1, ls='--'),
            annotation_clip=False)
ax.text(8.5, 2.8, '全局残差连接', fontsize=8, ha='center', color='#888', style='italic')

# 输出
draw_data_block(ax, 11, 1.2, 2.5, 1.0, 'σ (电导率)', '(B, 1500)')
ax.text(12.25, 0.7, '单元级电导率分布', ha='center', fontsize=9, color='#888')

# ==================== 绘制箭头 ====================
# 输入 → SharedEncoder
draw_arrow(ax, (3, 9.1), (4, 9.1))

# SharedEncoder → h
draw_arrow(ax, (7.5, 8.75), (8, 8.6))

# h → BLC
draw_arrow(ax, (9, 8.2), (9, 7.0))
draw_arrow(ax, (9, 7.0), (6, 7.0))

# h → FrequencyFusion (直接路径)
draw_arrow(ax, (10, 8.2), (10, 7.0))

# BLC → h_corrected
draw_arrow(ax, (6, 4.5), (6, 3.5))
draw_arrow(ax, (6, 3.5), (9, 3.5))
draw_arrow(ax, (9, 3.5), (9, 6.0))

# h_corrected → FrequencyFusion
draw_arrow(ax, (9, 6.0), (10.25, 7.0))

# h_pool → ResNetBackbone
draw_arrow(ax, (11.4, 7.45), (12.5, 6.25))

# FrequencyFusion → sigma_fused
draw_arrow(ax, (10.25, 4.5), (10.5, 4.0))

# ResNetBackbone → sigma_refined
draw_arrow(ax, (14, 4.5), (14, 4.0))
draw_arrow(ax, (14, 4.0), (14, 3.5))

# sigma_fused, sigma_refined → fusion
draw_arrow(ax, (11.4, 3.75), (11.7, 2.7))
draw_arrow(ax, (13.5, 3.75), (12.3, 2.7))

# fusion → output
draw_arrow(ax, (12, 2.2), (12, 2.2))
draw_arrow(ax, (12, 2.2), (12.25, 2.2))

# ==================== 图例 ====================
legend_y = 0.3
ax.add_patch(FancyBboxPatch((0.5, legend_y), 0.8, 0.4, boxstyle="round,pad=0.02",
                            facecolor=colors['encoder'], edgecolor='#333'))
ax.text(1.5, legend_y + 0.2, '编码器', fontsize=9, va='center')

ax.add_patch(FancyBboxPatch((3, legend_y), 0.8, 0.4, boxstyle="round,pad=0.02",
                            facecolor=colors['blc'], edgecolor='#333'))
ax.text(4, legend_y + 0.2, '校正模块', fontsize=9, va='center')

ax.add_patch(FancyBboxPatch((5.5, legend_y), 0.8, 0.4, boxstyle="round,pad=0.02",
                            facecolor=colors['fusion'], edgecolor='#333'))
ax.text(6.5, legend_y + 0.2, '解码器', fontsize=9, va='center')

ax.add_patch(FancyBboxPatch((8, legend_y), 0.8, 0.4, boxstyle="round,pad=0.02",
                            facecolor=colors['backbone'], edgecolor='#333'))
ax.text(9, legend_y + 0.2, '残差骨干', fontsize=9, va='center')

# ==================== 信息流说明 ====================
info_box = FancyBboxPatch((13, 0.2), 2.8, 1.8, boxstyle="round,pad=0.03",
                          facecolor='#FAFAFA', edgecolor='#BDBDBD', linewidth=1)
ax.add_patch(info_box)
ax.text(14.4, 1.8, '信息流', fontsize=10, fontweight='bold', ha='center')
ax.text(13.1, 1.5, '实线: 主路径', fontsize=8, ha='left')
ax.text(13.1, 1.2, '虚线: 残差连接', fontsize=8, ha='left')
ax.text(13.1, 0.9, '+: 特征融合', fontsize=8, ha='left')
ax.text(13.1, 0.6, 'B: Batch size', fontsize=8, ha='left')
ax.text(13.1, 0.35, 'F: 频率数 (6)', fontsize=8, ha='left')

plt.tight_layout()
plt.savefig('/Users/vangogh/455/王楠师姐/EITProject/eit_root_imaging/docs/network_architecture.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/vangogh/455/王楠师姐/EITProject/eit_root_imaging/docs/network_architecture.pdf',
            bbox_inches='tight', facecolor='white')
print("架构图已保存!")
