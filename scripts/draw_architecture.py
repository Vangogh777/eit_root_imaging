#!/usr/bin/env python3
"""
 ConvSpatialEIT v2 Architecture Diagram
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
fig.patch.set_facecolor('#0a0e17')
ax.set_facecolor('#0a0e17')

ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# 颜色
C_INPUT    = '#3b82f6'    # 蓝色
C_FUSION   = '#a855f7'    # 紫色
C_ENCODER  = '#22c55e'    # 绿色
C_SAMPLER  = '#f59e0b'    # 橙色
C_PE       = '#ec4899'    # 粉色
C_GNN      = '#ef4444'    # 红色
C_OUTPUT   = '#14b8a6'    # 青色
C_LABEL    = '#94a3b8'
C_TITLE    = '#e0e8f0'
C_BOX      = '#1e293b'
C_LINE     = (0.23, 0.51, 0.96, 0.3)

def draw_block(ax, x, y, w, h, color, label, sublabel="", shape='rect'):
    """块"""
    if shape == 'rect':
        rect = mpatches.FancyBboxPatch((x, y), w, h, 
            boxstyle=mpatches.BoxStyle("Round", pad=0.08),
            facecolor=color, edgecolor='none', alpha=0.85, 
            linewidth=0)
        ax.add_patch(rect)
    elif shape == 'pill':
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle=mpatches.BoxStyle("Round", pad=0.3),
            facecolor=color, edgecolor=(1,1,1,0.1), 
            alpha=0.85, linewidth=1)
        ax.add_patch(rect)
    
    ax.text(x + w/2, y + h/2 + 0.05, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.3, sublabel, ha='center', va='center',
                fontsize=7, color=(1,1,1,0.6))

def draw_arrow(ax, x1, x2, y, color=(1,1,1,0.3)):
    """箭头"""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

def draw_down_arrow(ax, x, y1, y2, color=(1,1,1,0.2)):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# ============ Title ============
ax.text(8, 9.6, 'ConvSpatialEIT v2 Architecture', 
        ha='center', va='center', fontsize=18, fontweight='bold', color=C_TITLE)
ax.text(8, 9.2, 'Multi-frequency boundary voltages -> FEM element conductivity', 
        ha='center', va='center', fontsize=10, color=C_LABEL)

# ============ Y positions (from bottom to top) ============
# 分层
Y_BASE = 0.5
SPACING = 1.35

y_out   = Y_BASE
y_head  = y_out + SPACING
y_gnn   = y_head + SPACING
y_pe    = y_gnn + SPACING
y_samp  = y_pe + SPACING
y_enc   = y_samp + SPACING
y_fuse  = y_enc + SPACING
y_in    = y_fuse + SPACING

BOX_W = 2.0
BOX_H = 0.6

# Center X
cx = 5.0

# ============ 1. INPUT ============
draw_block(ax, cx - BOX_W/2, y_in, BOX_W, BOX_H, C_INPUT, 
           'Voltages', '(B, 6, 208)', 'pill')
ax.text(cx + BOX_W/2 + 0.4, y_in + BOX_H/2, '6 freq x 208 meas', 
        va='center', fontsize=8, color=C_LABEL)

draw_arrow(ax, cx - BOX_W/2, cx + BOX_W/2, y_in + BOX_H/2, (0.23, 0.51, 0.96, 0.15))

# ============ 2. FreqCrossAttention ============
draw_block(ax, cx - BOX_W/2, y_fuse, BOX_W, BOX_H, C_FUSION,
           'Freq CrossAttn', '6->1 freq, d=64', 'pill')
ax.text(cx + BOX_W/2 + 0.4, y_fuse + BOX_H/2, 'Cross-frequency attention', 
        va='center', fontsize=8, color=C_LABEL)

# ============ 3. ConvEncoder ============
# 扩大编码器宽度来展示内部结构
ENC_W = 3.0
draw_block(ax, cx - ENC_W/2, y_enc, ENC_W, BOX_H, C_ENCODER,
           'ConvEncoder', 'base=96, SE+ResBlock', 'rect')
ax.text(cx + ENC_W/2 + 0.3, y_enc + BOX_H/2, 'stem → stage1 → stage2 → SE → proj', 
        va='center', fontsize=8, color=C_LABEL)

# 内部子模块标注
for i, (label, w_pct) in enumerate([('Stem\n96ch', 0.15), ('2×ResBlock\n192ch', 0.35), 
                                      ('ResBlock+SE\n384ch', 0.35), ('Proj\n832ch', 0.15)]):
    x0 = cx - ENC_W/2 + ENC_W * sum([0.15, 0.35, 0.35, 0.15][:i])
    x1 = cx - ENC_W/2 + ENC_W * sum([0.15, 0.35, 0.35, 0.15][:i+1])
    ax.text((x0+x1)/2, y_enc - 0.35, label, ha='center', va='center',
            fontsize=6, color=(1, 1, 1, 0.4))

# ============ 4. GridSampler + Skip ============
SAM_W = 2.5
draw_block(ax, cx - SAM_W/2, y_samp, SAM_W, BOX_H, C_SAMPLER,
           'GridSampler', 'bilinear 13x16 -> elements', 'rect')
ax.text(cx + SAM_W/2 + 0.35, y_samp + BOX_H/2, '+ skip connections',
        va='center', fontsize=8, color=C_LABEL)

# ============ 5. Position Encoding ============
PE_W = 2.2
draw_block(ax, cx - PE_W/2, y_pe, PE_W, BOX_H, C_PE,
           'Position Enc', 'radial + Fourier', 'pill')
ax.text(cx + PE_W/2 + 0.4, y_pe + BOX_H/2, '35-dim (r, x, y, sin/cos)',
        va='center', fontsize=8, color=C_LABEL)

# Concat 标记
ax.text(cx, y_pe + BOX_H + 0.15, 'Concat', ha='center', va='center',
        fontsize=7, fontweight='bold', color=C_FUSION,
        bbox=dict(boxstyle='round,pad=0.15', facecolor=(0.66, 0.33, 0.97, 0.15), edgecolor='none'))

# ============ 6. GNN ============
GNN_W = 3.5
# 4 层内部子块
n_layers = 4
layer_w = GNN_W / n_layers
for i in range(n_layers):
    lx = cx - GNN_W/2 + i * layer_w
    color = C_GNN if i < n_layers - 1 else '#f97316'  # 最后一层变橙色
    rect = mpatches.FancyBboxPatch((lx + 0.05, y_gnn), layer_w - 0.1, BOX_H,
        boxstyle=mpatches.BoxStyle("Round", pad=0.1),
        facecolor=color, edgecolor='none', alpha=0.7 if i < n_layers-1 else 0.9,
        linewidth=0)
    ax.add_patch(rect)
    label = f'SimpleGNN\n{i+1}' if i < 3 else 'SimpleGNN\n4 (out)'
    ax.text(lx + layer_w/2, y_gnn + BOX_H/2, label, ha='center', va='center',
            fontsize=7, fontweight='bold', color='white')
    # 层间小箭头
    if i < n_layers - 1:
        draw_arrow(ax, lx + layer_w - 0.1, lx + layer_w + 0.1, y_gnn + BOX_H/2, (1, 1, 1, 0.2))

ax.text(cx + GNN_W/2 + 0.35, y_gnn + BOX_H/2, f'4 layers\\n{147260} edges',
        va='center', fontsize=8, color=C_LABEL)

# ============ 7. Output Head ============
draw_block(ax, cx - BOX_W/2, y_head, BOX_W, BOX_H, C_OUTPUT,
           'Output Head', 'MLP: 256→128→1', 'pill')
ax.text(cx + BOX_W/2 + 0.4, y_head + BOX_H/2, 'sigmoid → [σ_min, σ_max]',
        va='center', fontsize=8, color=C_LABEL)

# ============ 8. Output ============
draw_block(ax, cx - BOX_W/2, y_out, BOX_W, BOX_H, '#14b8a6',
           'σ (B, n_elems)', '11466 elements', 'pill')
ax.text(cx + BOX_W/2 + 0.4, y_out + BOX_H/2, 'one value per element',
        va='center', fontsize=8, color=C_LABEL)

# ============ 垂直箭头连接 ============
draw_down_arrow(ax, cx, y_in - 0.1, y_fuse + BOX_H)
draw_down_arrow(ax, cx, y_fuse - 0.1, y_enc + BOX_H)
draw_down_arrow(ax, cx, y_enc - 0.1, y_samp + BOX_H)
draw_down_arrow(ax, cx, y_samp - 0.1, y_pe + BOX_H + 0.3)  # 到 concat

# 从 concat 到 GNN
draw_down_arrow(ax, cx, y_pe + BOX_H + 0.45, y_gnn + BOX_H)

draw_down_arrow(ax, cx, y_gnn - 0.1, y_head + BOX_H)
draw_down_arrow(ax, cx, y_head - 0.1, y_out + BOX_H)

# ============ 右侧参数标注 ============
params_info = [
    (y_in + BOX_H/2,  ''),
    (y_fuse + BOX_H/2, '26K'),
    (y_enc + BOX_H/2,  '4.93M'),
    (y_samp + BOX_H/2, ''),
    (y_pe + BOX_H/2,   ''),
    (y_gnn + BOX_H/2,  '1.11M'),
    (y_head + BOX_H/2, '41K'),
    (y_out + BOX_H/2,  ''),
]
for y_pos, text in params_info:
    if text:
        ax.text(14.5, y_pos, text, ha='center', va='center',
                fontsize=8, color=(1, 1, 1, 0.4),
                bbox=dict(boxstyle='round,pad=0.15', facecolor=(1, 1, 1, 0.04), edgecolor='none'))

# ============ 左侧分组括号和标签 ============
# 在左侧画分组大括号标注
groups = [
    (y_in - 0.1, y_fuse + BOX_H + 0.1, 'Feature\nExtraction', C_FUSION),
    (y_enc - 0.1, y_pe + BOX_H + 0.1, 'Spatial\nMapping', C_SAMPLER),
    (y_gnn - 0.1, y_gnn + BOX_H + 0.1, 'Graph\nReasoning', C_GNN),
    (y_head - 0.1, y_out + BOX_H + 0.1, 'Prediction', C_OUTPUT),
]

for y0, y1, label, color in groups:
    ym = (y0 + y1) / 2
    # 竖线
    ax.plot([1.0, 1.0], [y0, y1], color=color, alpha=0.3, lw=1.5, solid_capstyle='round')
    # 短横线
    ax.plot([0.95, 1.0], [y0, y0], color=color, alpha=0.3, lw=1)
    ax.plot([0.95, 1.0], [y1, y1], color=color, alpha=0.3, lw=1)
    # 标签
    ax.text(0.5, ym, label, ha='center', va='center', fontsize=8,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=(0.06, 0.09, 0.16, 0.8), edgecolor='none'))

# ============ 底部参数汇总 ============
ax.text(8, 0.05, 'Params: 6.1M | Pos enc: 35-dim | Edges: 147,260 | Infer: ~1.5 ms | Train: supervised+unsupervised',
        ha='center', va='center', fontsize=8, color=C_LABEL,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=(0.06, 0.09, 0.16, 0.6), 
                  edgecolor=(0.23, 0.51, 0.96, 0.15), linewidth=1))

plt.tight_layout(pad=0.5)
plt.savefig('results/architecture_v2.png', dpi=200, bbox_inches='tight', facecolor='#0a0e17')
plt.close()
print("✅ Architecture diagram saved: results/architecture_v2.png")
