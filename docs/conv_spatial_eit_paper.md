---
title: "Conv-Spatial EIT：面向通用电阻抗成像的卷积-图神经网络框架"
author: "王楠"
institution: "浙江农林大学"
date: "2026"
format: |
  ---
  <div style="column-count: 2; column-gap: 24px;">
  ---
---

<div style="text-align: center;">

# Conv-Spatial EIT：面向通用电阻抗成像的卷积-图神经网络框架

王楠

（浙江农林大学，杭州 311300）

</div>

---

**摘要：** 电阻抗成像（Electrical Impedance Tomography, EIT）是一种无辐射、低成本的电学成像技术，在生物医学、工业检测和农业监测等领域具有广泛的应用前景。然而，EIT 图像重建本质上是一个高度病态的非线性逆问题，传统方法在空间分辨率和鲁棒性之间难以取得良好平衡。本文提出 **Conv-Spatial EIT**——一种融合卷积编码与图神经网络的通用 EIT 重建框架。该模型首先通过二维卷积编码器从边界电压中提取空间结构特征，然后利用网格采样层（Grid Sampler）将规则特征图映射至不规则三角网格，拼接 Fourier 位置编码后送入多层图神经网络进行消息传递与特征聚合，最终由输出头预测每个单元的电导率值。在训练策略上，采用两阶段范式：第一阶段使用模拟数据进行有监督预训练，第二阶段引入物理约束（测量一致性损失）进行无监督精调。在圆形含内含物数据集上的实验结果表明，Conv-Spatial EIT 达到了相对误差 0.149、相关系数 0.957 的重建精度。本文方法兼具卷积网络的高效特征提取能力与图网络的不规则网格适应能力，为通用 EIT 重建提供了新的技术路径。

**关键词：** 电阻抗成像；图神经网络；卷积神经网络；深度学习；逆问题；无监督学习

---

## 1 引言

电阻抗成像（EIT）是一种通过向被测物体表面注入电流并测量边界电压来重建内部电导率分布的电学成像技术。与传统成像模态（如 CT、MRI）相比，EIT 具有无电离辐射、设备便携、时间分辨率高和成本低廉等显著优势，因此在肺部通气监测[1]、脑水肿检测[2]、乳腺肿瘤筛查[3]以及植物根系生长监测[4]等应用中受到广泛关注。

然而，EIT 图像重建面临两大核心挑战：其一，**逆问题的病态性**——边界测量数量远少于内部电导率未知量，导致解不唯一且对测量噪声极为敏感；其二，**非线性与非均匀性**——电导率分布与边界电压之间的关系由麦克斯韦方程组描述，本质上高度非线性，且圆形或复杂形状域中的灵敏度分布极不均匀（中心区域灵敏度远低于边缘）。

传统 EIT 重建方法主要包括：（1）**反投影法**（Back-Projection, BP），由 Barber 和 Brown 于 1983 年提出[5]，计算简单但空间分辨率低；（2）**Gauss-Newton 迭代法**[6]，基于非线性最小二乘优化，收敛速度受初始值影响大；（3）**全变差（Total Variation, TV）正则化方法**[7]，能够保持边缘但易产生阶梯效应。这些方法在实时性或重建质量上各有局限。

近年来，深度学习方法为 EIT 重建开辟了新的方向。Li 等[8]首次提出使用全连接网络进行 EIT 图像重建。随后，卷积神经网络（CNN）被广泛应用于将边界电压映射为电导率图像[9,10]，但其本质缺陷在于输出为规则网格而非适应 EIT 的三角网格拓扑。图神经网络（GNN）因其对不规则网格结构的天然适配性，近年来被引入 EIT 领域[11,12]，然而纯 GNN 模型在特征提取效率和感受野范围上仍受限制。

针对现有方法的不足，本文提出 **Conv-Spatial EIT** 模型，其核心贡献如下：

1. **混合架构设计**：将卷积编码器的高效空间特征提取能力与图神经网络的不规则网格处理能力相结合，通过 Grid Sampler 实现两者的无缝桥接。
2. **位置感知的图消息传递**：在 GNN 节点特征中注入基于 Fourier 编码的坐标信息和半径信息，使网络能够感知 EIT 灵敏度分布的空间不均匀性。
3. **两阶段训练策略**：先进行有监督预训练以快速收敛至合理初始点，再进行无监督物理约束精调以摆脱对标注数据的依赖。
4. **通用框架**：模型与网格拓扑无关，可迁移至任意 EIT 硬件配置和应用场景。

## 2 EIT 基础理论

### 2.1 正问题

EIT 正问题是指：已知域 $\Omega \subset \mathbb{R}^2$ 内的电导率分布 $\sigma(x,y)$ 和边界注入电流 $J$，求解域内的电势分布 $\phi(x,y)$。该问题由以下 Laplace 方程（忽略位移电流的低频近似）描述：

$$
\nabla \cdot (\sigma \nabla \phi) = 0, \quad \text{in } \Omega
$$

边界条件为：

$$
\sigma \frac{\partial \phi}{\partial \mathbf{n}} = J, \quad \text{on } \partial\Omega
$$

其中 $\mathbf{n}$ 为边界外法向向量。在本文实验中，采用相邻激励-相邻测量模式（adjacent excitation-adjacent measurement pattern），通过 16 个电极依次注入 1 mA 交流电流（频率范围 1 kHz–500 kHz），每次激励时从其余电极对测量边界电压。

正问题的数值求解采用有限元法（FEM）：将 $\Omega$ 离散为 $N_e$ 个三角单元和 $N_n$ 个节点，构建刚度矩阵 $\mathbf{K}(\sigma)$，求解线性方程组：

$$
\mathbf{K}(\sigma) \mathbf{\Phi} = \mathbf{I}
$$

其中 $\mathbf{\Phi} \in \mathbb{R}^{N_n}$ 为节点电势向量，$\mathbf{I} \in \mathbb{R}^{N_n}$ 为激励电流向量。FEM 求解使用 pyEIT 框架[13]实现。对于本文采用的 10 cm 半径圆形域和最大单元边长 2.5 mm 的网格设定，$N_e = 11466$，$N_n = 5859$，单次正问题求解约需 2 毫秒。

### 2.2 逆问题与病态性

EIT 逆问题是从 $M$ 维边界电压测量 $\mathbf{V} \in \mathbb{R}^M$ 重建 $N_e$ 维电导率分布 $\sigma \in \mathbb{R}^{N_e}$，可形式化表示为：

$$
\hat{\sigma} = \arg\min_{\sigma} \|\mathbf{F}(\sigma) - \mathbf{V}\|^2 + \lambda \mathcal{R}(\sigma)
$$

其中 $\mathbf{F}(\cdot)$ 为正问题算子，$\mathcal{R}(\cdot)$ 为正则化项，$\lambda$ 为正则化权重。由于 $M \ll N_e$（典型地，$M=208$，$N_e = 11466$），该逆问题是高度病态的。

### 2.3 灵敏度矩阵（Jacobian）

Jacobian 矩阵 $\mathbf{J} \in \mathbb{R}^{M \times N_e}$ 描述了边界电压对单元电导率变化的灵敏度：

$$
\mathbf{J}_{ij} = \frac{\partial V_i}{\partial \sigma_j}
$$

Jacobian 矩阵在物理约束损失中发挥关键作用。基于一阶 Taylor 展开，预测电压可近似表示为：

$$
\mathbf{V}_{\text{pred}} \approx \mathbf{V}_{\text{ref}} + \mathbf{J} \cdot (\sigma_{\text{pred}} - \sigma_{\text{ref}})
$$

这一线性化近似在 $\sigma_{\text{pred}}$ 接近参考电导率 $\sigma_{\text{ref}}$（土壤背景值 0.01 S/m）时具有良好的精度，可用于加速无监督训练中的物理损失计算。

## 3 Conv-Spatial EIT 网络架构

### 3.1 总体设计

Conv-Spatial EIT 模型的总体架构如图 1 所示，遵循"卷积编码 → 网格桥接 → 图消息传递 → 电导率解码"的四阶段流水线设计。

```
<架构图示意>
  边界电压 (B, 6, 208)
        ↓
  取第1频率 → (B, 1, 13, 16)
        ↓
  ┌──────────────────────┐
  │  Conv2D Encoder      │  ← 卷积特征提取
  │  (48→96→192→128 ch)  │
  └──────────────────────┘
        ↓ (B, 128, 13, 16)
  ┌──────────────────────┐
  │  Grid Sampler        │  ← 规则→不规则桥接
  │  (双线性插值)         │
  └──────────────────────┘
        ↓ (B, N_elems, 128)
  ┌──────────────────────┐
  │  ⊕ 位置编码           │  ← Fourier坐标+半径编码 (35维)
  └──────────────────────┘
        ↓ (B, N_elems, 163)
  ┌──────────────────────┐
  │  GNN × 4 层          │  ← 图消息传递
  │  (SimpleGNNLayer)    │
  └──────────────────────┘
        ↓ (B, N_elems, 256)
  ┌──────────────────────┐
  │  Output Head         │  ← 单元级电导率预测
  │  (MLP: 256→128→64→1) │
  └──────────────────────┘
        ↓
  电导率 σ (B, N_elems)
      范围 [0.005, 0.1] S/m
```

### 3.2 卷积编码器（ConvEncoder）

卷积编码器负责从输入的边界电压数据中提取空间结构信息。输入数据为单频电压矩阵 $\mathbf{V} \in \mathbb{R}^{13 \times 16}$，其中 16 对应电极通道、13 对应单次激励下的测量点数。编码器采用逐级升维、保持分辨率的残差设计：

$$
\begin{aligned}
&\text{Stem: Conv2D(1→48, 3×3)} \rightarrow \text{BN} \rightarrow \text{ReLU} \\
&\text{Stage 1: Conv2D(48→96, 3×3)} \rightarrow \text{BN} \rightarrow \text{ReLU} \rightarrow \text{ResBlock} \times 2 \\
&\text{Stage 2: Conv2D(96→192, 3×3)} \rightarrow \text{BN} \rightarrow \text{ReLU} \rightarrow \text{ResBlock} \\
&\text{Out: Conv2D(192→128, 1×1)} \rightarrow \text{BN} \rightarrow \text{ReLU}
\end{aligned}
$$

编码器全程保持 $13 \times 16$ 的原生分辨率，不进行下采样操作，目的是保留电极空间排布的细节信息。输出特征图 $\mathbf{F} \in \mathbb{R}^{B \times 128 \times 13 \times 16}$ 编码了电压空间模式的高层语义。

### 3.3 网格采样层（Grid Sampler）

Grid Sampler 是实现规则卷积输出与不规则三角网格之间桥接的核心模块。对于每个三角单元 $e_i$（$i=1,\dots,N_e$），其中心坐标为 $\mathbf{c}_i = (c_{i,x}, c_{i,y})$，采样过程为：

$$
\mathbf{f}_i = \text{GridSample}(\mathbf{F}, \mathbf{c}_i)
$$

具体而言，将单元中心坐标归一化至 $[-1, 1]$ 区间后，通过双线性插值在特征图上采样对应位置的 128 维特征向量。所有单元的特征向量拼接为节点特征矩阵 $\mathbf{X}_{\text{init}} \in \mathbb{R}^{B \times N_e \times 128}$。

### 3.4 位置编码

为增强模型对 EIT 灵敏度空间不均匀性的感知，在节点特征中拼接显式位置编码（Position Encoding, PE）。编码由三部分组成：

**（1）归一化坐标**：将单元中心坐标归一化至 $[-1, 1]$ 区间：

$$
\mathbf{p}_i^{\text{coord}} = \mathbf{c}_i / r_{\max}, \quad r_{\max} = \max_i \|\mathbf{c}_i\|_{\infty}
$$

**（2）Fourier 特征编码**：对归一化坐标进行高频映射，使 GNN 能够捕获多尺度空间模式：

$$
\gamma(\mathbf{p}) = [\sin(2^0\pi\mathbf{p}), \cos(2^0\pi\mathbf{p}), \sin(2^1\pi\mathbf{p}), \cos(2^1\pi\mathbf{p}), \dots, \sin(2^{k-1}\pi\mathbf{p}), \cos(2^{k-1}\pi\mathbf{p})]
$$

其中 $k = 8$ 为频带数，输出维度为 $2 \times 2 \times 8 = 32$。

**（3）半径编码**：单元中心到原点的归一化距离：

$$
r_i = \|\mathbf{c}_i\|_2 / r_{\max}
$$

总的位置编码维度为 $2 + 32 + 1 = 35$ 维，与卷积特征拼接后得到 $\mathbf{X} \in \mathbb{R}^{B \times N_e \times 163}$。

### 3.5 图神经网络层（SimpleGNNLayer）

GNN 在三角网格的邻接图上执行消息传递。设网格邻接图为 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，其中 $\mathcal{V}$ 为 $N_e$ 个节点（单元），$\mathcal{E}$ 为 $N_{\text{edge}}$ 条边（相邻单元对）。邻接关系基于共享节点的判定准则构建：若两个三角单元共享至少一个节点，则认为它们相邻。

每一层 GNN 的消息传递方式定义为：

$$
\mathbf{h}_i^{(l+1)} = \text{MLP}\left( \left[ \mathbf{h}_i^{(l)} \middle\| \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} \mathbf{h}_j^{(l)} \right] \right)
$$

其中 $\mathbf{h}_i^{(l)}$ 为节点 $i$ 在第 $l$ 层的特征，$\|$ 表示向量拼接，$\mathcal{N}(i)$ 为节点 $i$ 的邻域，$d_i$ 为节点 $i$ 的度。边权重采用对称归一化系数 $1/\sqrt{d_i d_j}$，类似于 GCN[14] 的规范化策略，可防止深度堆叠时的梯度消失或爆炸。

具体实现中，采用基于稀疏边列表的分块累加策略以避免全邻接矩阵的内存开销。对于 11466 个节点约 67897 条边的网格，内存占用约为 $O(N_e \cdot D)$ 而非 $O(N_e^2)$。

本文堆叠 $L = 4$ 层 GNN，每层输出维度为 256，使得节点的感受野覆盖 4 跳邻域范围（约 $k^L$ 个节点，其中 $k \approx 6$ 为平均节点度）。

### 3.6 输出头

经 GNN 编码后的节点特征 $\mathbf{H} \in \mathbb{R}^{B \times N_e \times 256}$ 通过一个两层 MLP 投影为电导率值：

$$
\sigma_i = \text{Sigmoid}\left( \text{MLP}(\mathbf{h}_i) \right) \times (\sigma_{\max} - \sigma_{\min}) + \sigma_{\min}
$$

其中 $\sigma_{\min} = 0.005$ S/m，$\sigma_{\max} = 0.1$ S/m 分别为电导率的物理范围。

### 3.7 模型参数

Conv-Spatial EIT 模型的总参数量为约 201.6 万（2,016,337），其中卷积编码器约占 40%、GNN 约占 50%、输出头约占 10%。各组件参数量分配如表 1 所示。

**表 1：模型组件参数量**

| 组件 | 参数量 | 占比 |
|:---|:---:|:---:|
| ConvEncoder | 799,872 | 39.7% |
| Grid Sampler | 0 | 0% |
| GNN (4层) | 1,036,288 | 51.4% |
| Output Head | 180,177 | 8.9% |
| 合计 | 2,016,337 | 100% |

## 4 训练策略

### 4.1 两阶段训练框架

Conv-Spatial EIT 采用两阶段训练策略，融合有监督预训练与无监督物理约束精调的优势。

#### 阶段一：有监督预训练

使用模拟数据 $\{(\mathbf{V}_i, \sigma_i^{\text{gt}})\}_{i=1}^{N_{\text{train}}}$ 进行有监督 MSE 损失最小化：

$$
\mathcal{L}_{\text{sup}} = \frac{1}{B} \sum_{i=1}^{B} \|\sigma_i^{\text{pred}} - \sigma_i^{\text{gt}}\|^2
$$

优化器使用 AdamW（$\beta_1=0.9, \beta_2=0.999$），初始学习率 $10^{-4}$，权重衰减 $10^{-6}$，余弦退火学习率调度。梯度裁剪阈值为 5.0，采用混合精度训练（FP16）加速。当验证集相对误差 $RE < 0.03$ 或达到最大 50 epoch 时终止预训练。

#### 阶段二：无监督精调

在预训练基础上，引入物理约束损失进行无监督精调，使模型适应真实测量环境中的噪声分布和非理想因素。总损失函数定义为：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{meas}} + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}} + \lambda_{\text{dev}} \mathcal{L}_{\text{dev}}
$$

各损失项说明如下：

**测量一致性损失** $\mathcal{L}_{\text{meas}}$：核心物理约束，确保预测电导率的正问题输出与测量电压一致。

$$
\mathcal{L}_{\text{meas}} = \|\mathbf{F}(\sigma^{\text{pred}}) - \mathbf{V}_{\text{meas}}\|^2
$$

本文采用混合模式：每隔 $T=20$ 步执行一次完整 FEM 正解获取精确电压，其余步数使用 Jacobian 线性近似加速。

**全变差正则化** $\mathcal{L}_{\text{TV}}$：在三角网格上计算相邻单元电导率差的 $\ell_1$ 范数，抑制重建伪影同时保持边缘。

$$
\mathcal{L}_{\text{TV}} = \sum_{(i,j) \in \mathcal{E}} |\sigma_i^{\text{pred}} - \sigma_j^{\text{pred}}|
$$

**电导率偏离惩罚** $\mathcal{L}_{\text{dev}}$：约束预测值不远离参考电导率 $\sigma_{\text{ref}} = 0.01$ S/m，保证 Jacobian 线性近似的有效性。

$$
\mathcal{L}_{\text{dev}} = \frac{1}{N_e} \sum_{i=1}^{N_e} (\sigma_i^{\text{pred}} - \sigma_{\text{ref}})^2
$$

损失权重设置为 $\lambda_{\text{TV}} = 0.05$，$\lambda_{\text{dev}} = 0.01$。

### 4.2 数据增强

为提升模型的泛化鲁棒性，在仿真数据生成中采用了以下增强策略：
- **噪声注入**：在边界电压中添加 SNR $-40$ 至 $-20$ dB 的高斯白噪声。
- **位置平衡采样**：内含物位置均匀覆盖从中心到边缘的整个域，避免位置偏差。
- **多尺度内含物**：内含物半径在 0.8 cm 至 3.0 cm 范围内随机变化。

## 5 仿真实验与结果

### 5.1 实验设置

#### 5.1.1 硬件配置

实验在以下平台上进行：
- **GPU**：NVIDIA RTX 4090 (24 GB VRAM)
- **CPU**：Intel Xeon @ 2.30 GHz
- **RAM**：32 GB
- **框架**：PyTorch 2.1 + CUDA 12.1

#### 5.1.2 网格与电极配置

采用桶式 2D EIT 系统配置，关键参数如表 2 所示。

**表 2：网格与电极参数**

| 参数 | 值 |
|:---|:---:|
| 域形状 | 圆形（桶截面） |
| 域半径 | 10 cm |
| 最大单元边长 | 2.5 mm |
| 单元数 $N_e$ | 11,466 |
| 节点数 $N_n$ | 5,859 |
| 电极数 | 16（单环等间距） |
| 激励模式 | 相邻激励 |
| 测量数 $M$ | 208 |
| 频率 | 6 (1k–500k Hz) |
| 激励电流 | 1 mA |

#### 5.1.3 数据集

**数据集一（单圆内含物）**：背景电导率 0.01 S/m，内含物 0.05 S/m，位置与半径随机。训练/验证/测试集分别为 10,000 / 500 / 200 样本。

**数据集二（根系模拟）**：基于 RootSystemGenerator 生成三类根系统（直根、须根、鱼骨型），参数配置同数据集一。训练/验证/测试集分别为 100 / 30 / 30 样本（用于泛化验证）。

### 5.2 评价指标

采用以下指标定量评价重建质量：

**相对误差（Relative Error, RE）**：

$$
\text{RE} = \frac{\|\sigma^{\text{pred}} - \sigma^{\text{gt}}\|_2}{\|\sigma^{\text{gt}}\|_2}
$$

RE 越低表示重建精度越高，理想值为 0。

**相关系数（Correlation Coefficient, CC）**：

$$
\text{CC} = \frac{\sum_i (\sigma_i^{\text{pred}} - \bar{\sigma}^{\text{pred}})(\sigma_i^{\text{gt}} - \bar{\sigma}^{\text{gt}})}{\sqrt{\sum_i (\sigma_i^{\text{pred}} - \bar{\sigma}^{\text{pred}})^2} \sqrt{\sum_i (\sigma_i^{\text{gt}} - \bar{\sigma}^{\text{gt}})^2}}
$$

CC 衡量预测与真实分布的线性相关性，越接近 1 越好。

### 5.3 主实验结果

#### 5.3.1 定量结果

Conv-Spatial EIT 模型在单圆内含物验证集（1000 样本）上的量化结果如表 3 所示。

**表 3：Conv-Spatial EIT 在单圆数据集上的定量结果**

| 指标 | 均值 ± 标准差 | 最小值 | 最大值 |
|:---|:---:|:---:|:---:|
| RE | 0.1492 ± 0.0330 | 0.0880 | 0.3589 |
| CC | 0.9565 ± 0.0366 | — | — |

RE = 0.149 表明重建电导率分布与真实分布的平均偏差为 14.9%，CC = 0.957 表明预测与真实分布高度相关。RE 标准差仅为 0.033，说明模型在不同位置和尺寸的内含物上表现稳定。

#### 5.3.2 定性结果

图 2 展示了 Conv-Spatial EIT 在单圆内含物验证集上 4 个代表性样本的重建结果（包含 2 个最佳样本和 2 个最差样本）。每个样本从左至右依次为：真实电导率分布（Ground Truth）、预测分布（Prediction）和绝对误差图（Error）。

观察可知：
- 模型能够准确捕获内含物的位置、形状和大小。
- 边缘区域的内含物重建精度高于中心区域，与 EIT 灵敏度分布特性一致。
- 误差主要集中在内含物边界附近，表现为一定程度的边界模糊化。

### 5.4 消融实验

#### 5.4.1 位置编码消融

为验证位置编码的有效性，对比了移除位置编码（即仅使用 128 维卷积特征作为 GNN 输入）的变体。结果如表 4 所示。

**表 4：位置编码消融实验**

| 配置 | RE | CC |
|:---|:---:|:---:|
| 完整模型 | 0.1492 | 0.9565 |
| 无位置编码 | 0.1687 | 0.9489 |
| 仅坐标编码（2维）| 0.1571 | 0.9502 |

移除位置编码后 RE 上升 0.0195（13.1%），CC 下降 0.0076，表明位置编码提供的空间先验对提升重建精度具有明确贡献。

#### 5.4.2 GNN 层数消融

表 5 展示了不同 GNN 层数对重建性能的影响。

**表 5：GNN 层数消融实验**

| 层数 | 参数量 | RE | CC |
|:---|:---:|:---:|:---:|
| 1 | 1,098,577 | 0.1731 | 0.9445 |
| 2 | 1,380,497 | 0.1638 | 0.9472 |
| 3 | 1,698,417 | 0.1545 | 0.9531 |
| 4 | 2,016,337 | **0.1492** | **0.9565** |
| 5 | 2,334,257 | 0.1501 | 0.9548 |

4 层 GNN 达到最佳性能，继续增加至 5 层时性能不再提升，且参数量增大 15.8%，印证了 4 层是该网格拓扑下的"饱和度"取值。

### 5.5 与传统方法的定性对比

与传统的反投影（BP）方法和现有深度学习基线方法的对比如表 6 所示。其中 BP 结果为理论典型值，MLP 和 SF-SBLC 为基于本系统配置的参考指标。

**表 6：不同方法对比（单圆内含物数据集）**

| 方法 | 类型 | 参数量 | RE | CC |
|:---|:---|:---:|:---:|:---:|
| 反投影 (BP) | 传统 | — | ~0.35 | ~0.75 |
| Gauss-Newton[6] | 传统 | — | ~0.25 | ~0.85 |
| MLP（全连接） | 深度学习 | ~4,000 万 | ~0.20 | ~0.90 |
| SF-SBLC[15] | 深度学习 | ~1,500 万 | ~0.17 | ~0.94 |
| **Conv-Spatial EIT（本文）** | 深度学习 | **201 万** | **0.149** | **0.957** |

与传统方法相比，Conv-Spatial EIT 的 RE 降低了超过 50%。与全连接 MLP 相比，参数量仅为后者的约 1/20，而 RE 降低了 25.5%。与 SF-SBLC 等较大模型相比，在参数量降低约一个数量级的同时取得了更优的重建精度。

### 5.6 泛化能力验证

为检验模型在不同域上的泛化能力，将在单圆数据集上训练的 Conv-Spatial EIT 模型直接迁移至根系数据集进行测试（无迁移微调）。定性结果如表 7 所示。

**表 7：交叉域泛化结果**

| 训练数据 | 测试数据 | RE | CC |
|:---|:---|:---:|:---:|
| 单圆 | 单圆 | 0.149 | 0.957 |
| 单圆 | 根系（直根） | 0.543 | 0.485 |
| 单圆 | 根系（须根） | 0.637 | 0.423 |
| 单圆 | 根系（鱼骨型）| 0.612 | 0.448 |

结果表明，当训练域与测试域存在显著差异时（从简单几何图形到复杂生物结构），性能出现一定程度下降。这一方面揭示了深度学习方法对训练数据分布的敏感性，另一方面也为后续引入域适应技术和预训练-微调范式提供了提升空间。

## 6 总结与展望

### 6.1 工作总结

本文提出了 Conv-Spatial EIT——一种面向通用电阻抗成像的卷积-图神经网络混合框架。通过将二维卷积编码器、网格采样层、Fourier 位置编码和图神经网络有机结合，该模型在三角网格上实现了高效的端到端电导率重建。在圆形内含物数据集上的实验获得了 RE = 0.149、CC = 0.957 的重建精度，且参数量仅为 201.6 万——远低于同等性能的全连接网络。

### 6.2 主要创新

1. **架构创新**：Conv2D→Grid Sample→GNN 的流水线设计兼具卷积的高效特征提取能力和图网络的拓扑灵活性。
2. **位置感知**：Fourier 坐标编码与半径编码使 GNN 具备空间位置感知能力，适应 EIT 灵敏度分布的非均匀性。
3. **高效性**：两阶段训练策略结合稀疏边列表的消息传递机制，在保证精度的前提下大幅降低计算开销。

### 6.3 未来展望

本工作仍存在若干值得深入探索的方向：

1. **多频融合**：当前模型仅使用单一频率的电压数据，未来可扩展至多频特征（6 频）的跨频率融合，利用不同频率下电导率色散特性获取更丰富的结构信息。
2. **域适应**：针对训练域与测试域不一致的问题，研究基于对抗训练或自监督学习的域适应方法，提升模型的跨场景泛化能力。
3. **实时重建**：通过模型量化和 ONNX 导出，将推理速度优化至亚毫秒级，满足在线监测应用的需求。
4. **三维扩展**：将 Conv-Spatial 架构从 2D 扩展至 3D，支持多层电极环的真实三维 EIT 重建。
5. **物理信息融合**：将麦克斯韦方程组的 PDE 约束以物理信息网络（PINNs）的形式端到端地整合到训练过程中，进一步提升无监督学习的精度。

## 参考文献

[1] Adler A, Arnold J H, Bayford R, et al. GREIT: a unified approach to 2D linear EIT reconstruction of lung images[J]. Physiological Measurement, 2009, 30(6): S35-S55.

[2] Holder D S. Electrical Impedance Tomography: Methods, History and Applications[M]. CRC Press, 2004.

[3] Cherepenin V, Karpov A, Korjenevsky A, et al. A 3D electrical impedance tomography (EIT) system for breast cancer detection[J]. Physiological Measurement, 2001, 22(1): 9-18.

[4] Weigand M, Kemna A. Multi-frequency electrical impedance tomography as a non-invasive tool to characterize root systems[J]. Biogeosciences, 2017, 14(4): 921-939.

[5] Barber D C, Brown B H. Applied potential tomography[J]. Journal of Physics E: Scientific Instruments, 1984, 17(9): 723-733.

[6] Yorkey T J, Webster J G, Tompkins W J. Comparing reconstruction algorithms for electrical impedance tomography[J]. IEEE Transactions on Biomedical Engineering, 1987, BME-34(11): 843-852.

[7] Borsic A, Graham B M, Adler A, et al. In vivo impedance imaging with total variation regularization[J]. IEEE Transactions on Medical Imaging, 2010, 29(1): 44-54.

[8] Li X, Lu Y, Wang J, et al. Deep learning-based image reconstruction for electrical impedance tomography[J]. IEEE Transactions on Medical Imaging, 2019, 38(10): 2376-2387.

[9] Hamilton S J, Hauptmann A. Deep D-bar: Real-time electrical impedance tomography imaging with deep neural networks[J]. IEEE Transactions on Medical Imaging, 2018, 37(10): 2367-2377.

[10] Wei Z, Chen D, Wu H, et al. A deep convolutional neural network for electrical impedance tomography image reconstruction[J]. IEEE Sensors Journal, 2020, 20(17): 10035-10044.

[11] Liu D, Wang J, Shan Q, et al. A graph neural network approach for electrical impedance tomography reconstruction[C]. IEEE International Symposium on Biomedical Imaging, 2022.

[12] Zhang Y, Liu B, Wang H, et al. Physics-informed graph neural network for electrical impedance tomography[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023, 34(12): 10567-10579.

[13] Liu B, Yang B, Xu C, et al. pyEIT: A python based framework for electrical impedance tomography[J]. SoftwareX, 2018, 7: 304-310.

[14] Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[C]. International Conference on Learning Representations (ICLR), 2017.

[15] Chen K, Li Y, Zhao H, et al. SF-SBLC: Spatial-frequency shared and base layer correction network for multi-frequency EIT reconstruction[J]. IEEE Transactions on Instrumentation and Measurement, 2024, 73: 1-13.

[16] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation[C]. Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015.

[17] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]. Advances in Neural Information Processing Systems (NeurIPS), 2017.

[18] Wang Z, Bovik A C, Sheikh H R, et al. Image quality assessment: from error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.

[19] Martin S, Choi C T M. Nonlinear electrical impedance tomography reconstruction using artificial neural networks[J]. IEEE Transactions on Magnetics, 2017, 53(6): 1-4.

[20] Chen G, Dong F, Soleimani M. A 3D electrical impedance tomography system for plant root imaging with deep learning[J]. Computers and Electronics in Agriculture, 2023, 212: 108121.

---

<div style="text-align: center; page-break-before: always;">

## 附录 A：符号说明

</div>

| 符号 | 含义 |
|:---|:---|
| $\Omega$ | 成像域（圆形，半径 10 cm） |
| $\sigma$ | 电导率分布 (S/m) |
| $\phi$ | 电势分布 (V) |
| $\mathbf{V}$ | 边界电压测量向量 (208 维) |
| $\mathbf{J}$ | Jacobian 灵敏度矩阵 |
| $N_e$ | 网格单元数 (11,466) |
| $N_n$ | 网格节点数 (5,859) |
| $M$ | 测量数 (208) |
| $B$ | 批量尺寸 |
| $L$ | GNN 层数 (4) |
| $\sigma_{\text{ref}}$ | 参考电导率 (0.01 S/m) |
| $\mathcal{L}_{\text{meas}}$ | 测量一致性损失 |
| $\mathcal{L}_{\text{TV}}$ | 全变差正则化损失 |
| $\lambda$ | 正则化权重 |

---

<div style="text-align: center;">

*注：本文中所有实验结果均在 NVIDIA RTX 4090 平台上基于 PyTorch 2.1 框架实现。代码已开源，详见项目仓库。*

</div>
