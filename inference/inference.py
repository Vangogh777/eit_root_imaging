"""
推理脚本
========
加载训练好的模型，对新测量数据进行推理。
支持 ONNX 导出和批量推理。
"""

import os
import sys
import yaml
import numpy as np
import torch
from typing import Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sf_sblc import SFSBLC


class EITInference:
    """
    EIT 推理引擎

    用法:
        engine = EITInference("checkpoints/model_final.pt", "config/train_config.yaml")
        sigma = engine(voltages)  # 单样本
        sigmas = engine.batch(voltages_batch)  # 批量
    """

    def __init__(self, checkpoint_path: str, config_path: str = "config/train_config.yaml",
                 device: Optional[str] = None):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model(checkpoint_path)
        self.model.eval()

    def _load_model(self, checkpoint_path: str):
        """加载模型权重"""
        model_cfg = self.cfg['model']
        self.model = SFSBLC(
            input_dim=model_cfg['input_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            n_frequencies=model_cfg['n_frequencies'],
            n_elems=model_cfg.get('n_elems', 1500),
            n_res_blocks=model_cfg.get('n_res_blocks', 8),
            dropout=model_cfg.get('dropout', 0.1),
            use_attention=model_cfg.get('use_attention', True),
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"模型加载完成: {checkpoint_path}")

    def preprocess(self, voltages: np.ndarray) -> torch.Tensor:
        """预处理：确保维度正确"""
        if voltages.ndim == 2:
            voltages = voltages[np.newaxis, :, :]  # (1, F, M)
        tensor = torch.from_numpy(voltages).float().to(self.device)
        return tensor

    def postprocess(self, sigma: torch.Tensor) -> np.ndarray:
        """后处理"""
        return sigma.detach().cpu().numpy()

    @torch.no_grad()
    def __call__(self, voltages: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """单样本推理"""
        if isinstance(voltages, np.ndarray):
            voltages = self.preprocess(voltages)
        elif voltages.dim() == 2:
            voltages = voltages.unsqueeze(0)

        out = self.model(voltages)
        sigma = self.postprocess(out['sigma'])

        if sigma.shape[0] == 1:
            sigma = sigma.squeeze(0)
        return sigma

    @torch.no_grad()
    def batch(self, voltages: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """批量推理"""
        if isinstance(voltages, np.ndarray):
            voltages = self.preprocess(voltages)
        out = self.model(voltages)
        return self.postprocess(out['sigma'])

    @torch.no_grad()
    def predict_with_details(self, voltages: torch.Tensor) -> dict:
        """推理 + 返回所有中间结果"""
        if voltages.dim() == 2:
            voltages = voltages.unsqueeze(0)
        out = self.model(voltages)
        return {
            'sigma': out['sigma'].cpu().numpy(),
            'base_map': out['base_map'].cpu().numpy() if out['base_map'] is not None else None,
            'freq_weights': out['freq_weights'].cpu().numpy() if out['freq_weights'] is not None else None,
            'blc_gates': out['blc_gates'].cpu().numpy() if out['blc_gates'] is not None else None,
        }
