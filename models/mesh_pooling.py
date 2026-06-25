"""
Mesh Pooling / Unpooling for hierarchical graph U-Net.
========================================================
基于 Farthest Point Sampling (FPS) + k-NN 的图粗化策略。
不需要 Graclus 等外部依赖, 纯 PyTorch 实现。
"""
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


def fps_downsample(centers: np.ndarray, n_target: int) -> np.ndarray:
    """
    Farthest Point Sampling: 从 n 个点中均匀选 n_target 个锚点。

    返回: anchor indices (n_target,)
    """
    n = centers.shape[0]
    if n_target >= n:
        return np.arange(n, dtype=np.int64)

    idx = np.zeros(n_target, dtype=np.int64)
    idx[0] = np.random.randint(0, n)

    # 距离缓存, 动态更新
    dist = np.full(n, np.inf, dtype=np.float32)
    for i in range(1, n_target):
        # 更新到最近锚点的距离
        new_dist = ((centers - centers[idx[i - 1]]) ** 2).sum(axis=1)
        dist = np.minimum(dist, new_dist)
        idx[i] = np.argmax(dist)

    return idx


def build_hierarchy(centers: np.ndarray, elements: np.ndarray,
                    n_levels: int = 3, k_neighbors: int = 8):
    """
    预计算多尺度图层次结构。

    参数:
        centers: (n_elems, 2) 元素中心坐标
        elements: (n_elems, 3) 三角元素连接
        n_levels: 粗化层数 (不含 level 0)
        k_neighbors: 每层 k-NN 邻居数

    返回:
        hierarchy: list of dict, 每一层:
            - nodes: 节点数
            - edges: (2, n_edges) 边索引
            - edge_weight: (n_edges,) 归一化权重
            - edge_feat: (n_edges, 4) 边特征
            - centers: (nodes, 2) 该层中心坐标
            - cluster: (child_nodes,) 父节点索引 (pooling map)
    """
    hierarchy = []
    n_elems = centers.shape[0]

    # ---- Level 0: 原始 FEM 图 ----
    edge_idx, edge_weight, edge_feat = _build_element_graph(centers, elements)
    hierarchy.append({
        'nodes': n_elems,
        'edges': torch.from_numpy(edge_idx).long(),
        'edge_weight': torch.from_numpy(edge_weight).float(),
        'edge_feat': torch.from_numpy(edge_feat).float(),
        'centers': centers.astype(np.float32),
        'cluster': torch.arange(n_elems, dtype=torch.long),  # identity
    })

    # ---- Level 1..K: 逐层粗化 ----
    cur_centers = centers.astype(np.float32)
    cur_n = n_elems

    for level in range(n_levels):
        target_n = max(cur_n // 4, 8)

        anchors = fps_downsample(cur_centers, target_n)
        anchor_centers = cur_centers[anchors]

        # k-NN 图: 每个锚点连到 k 个最近锚点
        from scipy.spatial import KDTree
        tree = KDTree(anchor_centers)
        _, knn_idx = tree.query(anchor_centers, k=min(k_neighbors + 1, target_n))

        # 构建边: 双向 + self-loop
        level_edges = []
        for i in range(target_n):
            for j in knn_idx[i]:
                if i != j:
                    level_edges.append((i, j))
            level_edges.append((i, i))  # self-loop

        edge_arr = np.array(level_edges, dtype=np.int64).T
        if edge_arr.size == 0:
            edge_arr = np.zeros((2, 1), dtype=np.int64)

        # 边权重: degree-based normalization
        deg = np.ones(target_n, dtype=np.float32)
        for e in level_edges:
            if e[0] != e[1]:
                deg[e[0]] += 1.0
        deg = np.sqrt(deg) + 1e-8
        ew = np.ones(edge_arr.shape[1], dtype=np.float32)
        ew /= deg[edge_arr[0]] * deg[edge_arr[1]]

        # 边特征: 距离 / 共享节点 / 半径比 / 余弦相似性 (简化)
        ef = np.zeros((edge_arr.shape[1], 4), dtype=np.float32)
        max_dist = np.max(np.abs(anchor_centers)) * 2 + 1e-8
        for e_idx, (i, j) in enumerate(zip(edge_arr[0], edge_arr[1])):
            ci, cj = anchor_centers[i], anchor_centers[j]
            d = np.linalg.norm(ci - cj)
            ef[e_idx, 0] = d / (d + 0.002)
            ef[e_idx, 2] = min(np.linalg.norm(ci), np.linalg.norm(cj)) / \
                           (max(np.linalg.norm(ci), np.linalg.norm(cj)) + 1e-8)
            if d > 1e-8:
                dot = (ci * cj).sum() / ((np.linalg.norm(ci) + 1e-8) * (np.linalg.norm(cj) + 1e-8))
            else:
                dot = 1.0
            ef[e_idx, 3] = (dot + 1.0) / 2.0
        ef[:, 1] = 0.0  # 非 FEM 图没有共享节点

        # cluster map: 每个父级节点对应哪些子节点
        # 这里用 k-NN 分配: 每个子节点分配给最近的锚点
        _, cluster = tree.query(cur_centers)

        hierarchy.append({
            'nodes': target_n,
            'edges': torch.from_numpy(edge_arr).long(),
            'edge_weight': torch.from_numpy(ew).float(),
            'edge_feat': torch.from_numpy(ef).float(),
            'centers': anchor_centers,
            'cluster': torch.from_numpy(cluster.astype(np.int64)).long(),
        })

        cur_centers = anchor_centers
        cur_n = target_n

    return hierarchy


def _build_element_graph(centers: np.ndarray, elements: np.ndarray):
    """从 FEM 三角元素构建邻接图"""
    n_elems = elements.shape[0]
    node_to_elems = defaultdict(list)
    for i, tri in enumerate(elements):
        for node in tri:
            node_to_elems[int(node)].append(i)

    undirected = set()
    for elems in node_to_elems.values():
        for a in range(len(elems)):
            for b in range(a + 1, len(elems)):
                i, j = int(elems[a]), int(elems[b])
                if i != j:
                    undirected.add((min(i, j), max(i, j)))

    directed = []
    for i, j in sorted(undirected):
        directed.append((i, j))
        directed.append((j, i))
    directed.extend((i, i) for i in range(n_elems))

    edge_idx = np.array(directed, dtype=np.int64).T
    deg = np.ones(n_elems, dtype=np.float32)
    for i, j in undirected:
        deg[i] += 1.0; deg[j] += 1.0
    deg = np.sqrt(deg) + 1e-8
    edge_weight = np.ones(edge_idx.shape[1], dtype=np.float32)
    edge_weight /= deg[edge_idx[0]] * deg[edge_idx[1]]

    edge_feat = np.zeros((edge_idx.shape[1], 4), dtype=np.float32)
    elem_sets = [set(map(int, tri)) for tri in elements]
    max_r = np.max(np.abs(centers)) * 2 + 1e-8
    for e, (i, j) in enumerate(zip(edge_idx[0], edge_idx[1])):
        ci, cj = centers[i], centers[j]
        d = np.linalg.norm(ci - cj)
        edge_feat[e, 0] = d / (d + 0.002)
        edge_feat[e, 1] = len(elem_sets[i] & elem_sets[j]) / 3.0
        ri, rj = np.linalg.norm(ci) / max_r, np.linalg.norm(cj) / max_r
        edge_feat[e, 2] = min(ri, rj) / (max(ri, rj) + 1e-8)
        if d > 1e-8:
            dot = (ci * cj).sum() / ((np.linalg.norm(ci) + 1e-8) * (np.linalg.norm(cj) + 1e-8))
        else:
            dot = 1.0
        edge_feat[e, 3] = (dot + 1.0) / 2.0

    return edge_idx, edge_weight, edge_feat


class GraphPool(nn.Module):
    """图池化: max aggregation within clusters"""
    def forward(self, x: torch.Tensor, cluster: torch.Tensor, n_parent: int) -> torch.Tensor:
        """
        x: (B, n_child, dim)
        cluster: (n_child,) parent index for each child
        n_parent: number of parent nodes
        """
        B, _, D = x.shape
        device = x.device
        out = x.new_full((B, n_parent, D), -1e9)
        cluster_expand = cluster.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        out.scatter_reduce_(1, cluster_expand, x, reduce='amax', include_self=False)
        return out


class GraphUnpool(nn.Module):
    """图反池化: broadcast parent features to children"""
    def forward(self, x: torch.Tensor, cluster: torch.Tensor, n_child: int) -> torch.Tensor:
        """
        x: (B, n_parent, dim)
        cluster: (n_child,) parent for each child
        n_child: number of child nodes
        """
        return x[:, cluster, :]
