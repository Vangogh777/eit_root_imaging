"""
随机植物根结构生成器
====================
生成 2D 横截面中的根系电导率分布。
支持三类根型：直根（taproot）、须根（fibrous）、鱼骨型（herringbone）。
直接光栅化到 pyEIT 网格单元上，无需外部模拟器。
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RootSegment:
    """单条根段"""
    x: float          # 起点 x (m)
    y: float          # 起点 y (m)
    angle: float      # 生长方向 (rad)
    length: float     # 段长 (m)
    radius: float     # 根半径 (m)
    branching_level: int = 0  # 分支级数 (0=主根)

    @property
    def end_point(self) -> Tuple[float, float]:
        return (self.x + self.length * np.cos(self.angle),
                self.y + self.length * np.sin(self.angle))

    @property
    def mid_point(self) -> Tuple[float, float]:
        return (self.x + 0.5 * self.length * np.cos(self.angle),
                self.y + 0.5 * self.length * np.sin(self.angle))


class RootSystemGenerator:
    """
    生成二维根系统电导率分布。

    用法:
        generator = RootSystemGenerator(mesh_nodes, mesh_elements)
        sigma = generator.generate(seed=42)  # 返回 (n_elems,) 电导率
        batch = generator.generate_batch(1000)  # 批量生成
    """

    def __init__(self, mesh_nodes: np.ndarray, mesh_elements: np.ndarray,
                 domain_radius: float = 0.1,
                 conductivity_root: float = 0.05,
                 conductivity_soil: float = 0.01):
        """
        参数:
            mesh_nodes: (n_nodes, 2) pyEIT 网格节点坐标
            mesh_elements: (n_elems, 3) 三角单元节点索引
            domain_radius: 域半径 (m)，用于约束根在桶内
            conductivity_root: 根组织电导率
            conductivity_soil: 土壤电导率（背景）
        """
        self.nodes = mesh_nodes
        self.elements = mesh_elements.astype(int)
        self.n_elems = len(mesh_elements)
        self.domain_radius = domain_radius
        self.sigma_root = conductivity_root
        self.sigma_soil = conductivity_soil

        # 预计算每个三角单元的中心坐标
        self.elem_centers = np.mean(self.nodes[self.elements], axis=1)
        # 预计算单元到原点的距离（用于边界约束）
        self.elem_dist = np.linalg.norm(self.elem_centers, axis=1)

    def _is_inside_domain(self, x: float, y: float) -> bool:
        """检查坐标是否在圆形域内"""
        return np.sqrt(x**2 + y**2) < self.domain_radius * 0.95

    def generate(self, seed: Optional[int] = None,
                 root_type: Optional[str] = None) -> np.ndarray:
        """
        生成一个随机根系统的电导率分布

        参数:
            seed: 随机种子
            root_type: None(随机)/"taproot"/"fibrous"/"herringbone"

        返回:
            sigma: (n_elems,) 每个单元的电导率值
        """
        if seed is not None:
            np.random.seed(seed)

        if root_type is None:
            root_type = np.random.choice(
                ['taproot', 'fibrous', 'herringbone'],
                p=[0.4, 0.3, 0.3]
            )

        # 生成根段列表
        if root_type == 'taproot':
            segments = self._generate_taproot()
        elif root_type == 'fibrous':
            segments = self._generate_fibrous()
        elif root_type == 'herringbone':
            segments = self._generate_herringbone()
        else:
            raise ValueError(f"Unknown root type: {root_type}")

        # 光栅化到网格单元
        sigma = np.full(self.n_elems, self.sigma_soil)
        for seg in segments:
            self._rasterize_segment(sigma, seg)

        return sigma

    def generate_with_label(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成电导率图 + 二值掩码（根位置标注）
        返回: (sigma, mask) — mask[i]=1 表示单元 i 有根
        """
        if seed is not None:
            np.random.seed(seed)

        sigma = np.full(self.n_elems, self.sigma_soil)
        mask = np.zeros(self.n_elems, dtype=np.float32)

        root_type = np.random.choice(['taproot', 'fibrous', 'herringbone'], p=[0.4, 0.3, 0.3])
        gen_fn = getattr(self, f'_generate_{root_type}')
        segments = gen_fn()

        for seg in segments:
            affected = self._rasterize_segment(sigma, seg, return_mask=True)
            mask[affected] = 1.0

        return sigma, mask

    def _rasterize_segment(self, sigma: np.ndarray, seg: RootSegment,
                            return_mask: bool = False) -> Optional[np.ndarray]:
        """将一条根段光栅化到网格单元"""
        sx, sy = seg.x, seg.y
        ex, ey = seg.end_point
        seg_len = np.sqrt((ex - sx)**2 + (ey - sy)**2)
        if seg_len < 1e-8:
            return None

        centers = self.elem_centers
        cx, cy = centers[:, 0], centers[:, 1]

        # 点到线段的最小距离
        dx, dy = ex - sx, ey - sy
        t = ((cx - sx) * dx + (cy - sy) * dy) / (dx**2 + dy**2)
        t = np.clip(t, 0, 1)
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        dist = np.sqrt((cx - proj_x)**2 + (cy - proj_y)**2)

        # 距离小于根半径 + 渐变过渡
        mask = dist < seg.radius
        if np.any(mask):
            # 渐变赋值：中心强，边缘弱
            for idx in np.where(mask)[0]:
                d = dist[idx]
                weight = 1.0 - (d / seg.radius) ** 2
                sigma[idx] = self.sigma_soil + weight * (self.sigma_root - self.sigma_soil)

        if return_mask:
            return mask
        return None

    def _generate_taproot(self) -> List[RootSegment]:
        """直根系统（如萝卜/胡萝卜）"""
        segments = []
        # 主根从上方生长点向下
        x0, y0 = np.random.uniform(-0.02, 0.02), 0.09
        angle = -np.pi / 2  # 向下

        n_main = np.random.randint(3, 6)
        for i in range(n_main):
            length = np.random.uniform(0.015, 0.025)
            radius = 0.010 * (1 - i * 0.15)  # 越往下越细
            seg = RootSegment(x0, y0, angle, length, max(radius, 0.002), 0)
            segments.append(seg)
            x0, y0 = seg.end_point
            angle += np.random.uniform(-0.15, 0.15)

        # 侧根
        for i in range(1, min(4, len(segments))):
            parent = segments[i]
            px, py = parent.mid_point
            n_branches = np.random.randint(1, 4)
            for _ in range(n_branches):
                side = np.random.choice([-1, 1])
                branch_angle = parent.angle + side * np.random.uniform(0.4, 1.2)
                branch_len = np.random.uniform(0.01, 0.03)
                branch_radius = np.random.uniform(0.0015, 0.003)
                seg = RootSegment(px, py, branch_angle, branch_len, branch_radius, 1)
                segments.append(seg)

        return segments

    def _generate_fibrous(self) -> List[RootSegment]:
        """须根系统（如小麦/玉米）"""
        segments = []
        base_x, base_y = np.random.uniform(-0.02, 0.02), 0.09
        n_roots = np.random.randint(6, 14)

        for _ in range(n_roots):
            angle = np.random.uniform(-np.pi * 0.85, -np.pi * 0.15)
            length = np.random.uniform(0.02, 0.07)
            radius = np.random.uniform(0.002, 0.005)
            seg = RootSegment(base_x, base_y, angle, length, radius, 0)
            segments.append(seg)

            # 少数根再分支
            if np.random.rand() < 0.3:
                px, py = seg.end_point
                for _ in range(np.random.randint(1, 3)):
                    ba = angle + np.random.uniform(-0.6, 0.6)
                    bl = np.random.uniform(0.005, 0.015)
                    br = radius * np.random.uniform(0.3, 0.6)
                    segments.append(RootSegment(px, py, ba, bl, br, 1))

        return segments

    def _generate_herringbone(self) -> List[RootSegment]:
        """鱼骨型根系统"""
        segments = []
        x0, y0 = np.random.uniform(-0.01, 0.01), 0.09
        angle = -np.pi / 2
        n_nodes = np.random.randint(4, 8)

        for i in range(n_nodes):
            length = np.random.uniform(0.012, 0.02)
            radius = 0.007 * (1 - i * 0.12)
            seg = RootSegment(x0, y0, angle, length, max(radius, 0.002), i)
            segments.append(seg)
            x0, y0 = seg.end_point

            # 每隔一节发一对侧根
            if i >= 1 and i % 2 == 1:
                for side in [-1, 1]:
                    ba = angle + side * np.random.uniform(0.5, 1.0)
                    bl = np.random.uniform(0.01, 0.025)
                    br = np.random.uniform(0.0015, 0.003)
                    segments.append(RootSegment(x0, y0, ba, bl, br, i + 1))

            angle += np.random.uniform(-0.1, 0.1)

        return segments

    def generate_batch(self, n_samples: int, seed_start: int = 0,
                       return_masks: bool = False) -> np.ndarray:
        """批量生成"""
        if return_masks:
            sigmas, masks = [], []
            for i in range(n_samples):
                s, m = self.generate_with_label(seed=seed_start + i)
                sigmas.append(s)
                masks.append(m)
            return np.stack(sigmas, axis=0), np.stack(masks, axis=0)
        else:
            sigmas = []
            for i in range(n_samples):
                sigmas.append(self.generate(seed=seed_start + i))
            return np.stack(sigmas, axis=0)

    def generate_with_variations(self, n_samples: int) -> List[np.ndarray]:
        """
        生成多样化的根结构（带有渐变对比度和偏移）
        用于增强训练数据的多样性
        """
        results = []
        for i in range(n_samples):
            seed = i * 7 + 13
            sigma = self.generate(seed=seed)

            # 随机调整对比度
            contrast = np.random.uniform(3.0, 8.0)
            root_mask = sigma > self.sigma_soil * 1.1
            sigma[root_mask] = self.sigma_soil * contrast

            # 随机添加土壤不均匀性（小范围电导率波动）
            noise = np.random.randn(self.n_elems) * 0.001
            sigma += noise
            sigma = np.clip(sigma, 0.001, None)

            results.append(sigma)
        return results


if __name__ == "__main__":
    # 快速测试
    from pyeit.mesh import create

    mesh = create(n_el=16, fd=0.1, h0=0.005)
    nodes = mesh['node']
    elems = mesh['element']

    gen = RootSystemGenerator(nodes, elems)
    sigma = gen.generate(seed=42)
    print(f"电导率范围: [{sigma.min():.4f}, {sigma.max():.4f}]")
    print(f"根单元数: {(sigma > 0.011).sum()} / {len(sigma)}")
    print("[测试通过] RootSystemGenerator 工作正常")
