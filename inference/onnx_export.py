"""
ONNX 模型导出
==============
将训练好的 PyTorch 模型导出为 ONNX 格式，用于部署。
支持:
  - 静态/动态 batch
  - 与官方 onnxruntime 验证
"""

import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sf_sblc import SFSBLC


def export_to_onnx(checkpoint_path: str, config_path: str = "config/train_config.yaml",
                   output_path: str = "inference/model.onnx",
                   dynamic_batch: bool = True, verbose: bool = True):
    """导出 ONNX 模型"""

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cpu')

    # 加载模型
    model_cfg = cfg['model']
    model = SFSBLC(
        input_dim=model_cfg['input_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        n_frequencies=model_cfg['n_frequencies'],
        n_elems=model_cfg.get('n_elems', 1500),
        n_res_blocks=model_cfg.get('n_res_blocks', 8),
        dropout=model_cfg.get('dropout', 0.1),
        use_attention=model_cfg.get('use_attention', True),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval()

    # 生成示例输入
    n_freq = model_cfg['n_frequencies']
    n_meas = model_cfg['input_dim']
    dummy_input = torch.randn(1, n_freq, n_meas)

    # 动态轴配置
    dynamic_axes = {
        'voltages': {0: 'batch_size'},
        'sigma': {0: 'batch_size'},
    } if dynamic_batch else None

    # 导出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['voltages'],
        output_names=['sigma'],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        verbose=verbose,
    )

    print(f"ONNX 模型已导出: {output_path}")

    # 验证
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过 ✓")
    except ImportError:
        print("[WARN] onnx 库未安装，跳过验证")

    # 使用 onnxruntime 测试
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        result = session.run([output_name], {input_name: dummy_input.numpy()})
        print(f"ONNX Runtime 推理测试通过: output shape {result[0].shape} ✓")
    except ImportError:
        print("[WARN] onnxruntime 未安装，跳过 runtime 测试")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    parser.add_argument("--output", type=str, default="inference/model.onnx")
    parser.add_argument("--static", action="store_true")
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        dynamic_batch=not args.static,
    )
