# convert_to_onnx.py

import torch.onnx
import torch

def convert_to_onnx(model_path, onnx_model_path, input_size=(1, 3, 640, 640)):
    """
    将 PyTorch 模型转换为 ONNX 格式。
    
    Args:
        model_path (str): PyTorch 模型路径（量化后的模型）。
        onnx_model_path (str): 输出的 ONNX 模型路径。
        input_size (tuple): 模型输入尺寸。
    """
    # 加载量化后的 PyTorch 模型
    model = torch.load(model_path)
    model.eval()
    
    # 创建假输入
    dummy_input = torch.randn(*input_size)
    
    # 导出模型为 ONNX 格式
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_model_path, 
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"模型已转换为 ONNX 格式并保存为 '{onnx_model_path}'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将模型转换为 ONNX 格式")
    parser.add_argument('--model_path', type=str, default='quantized_model.pth', help='量化后的模型路径 (.pth)')
    parser.add_argument('--onnx_model_path', type=str, default='model.onnx', help='输出的 ONNX 模型路径 (.onnx)')
    parser.add_argument('--input_size', type=int, nargs=4, default=[1, 3, 640, 640], help='模型输入尺寸 (例如: 1 3 640 640)')

    args = parser.parse_args()
    convert_to_onnx(args.model_path, args.onnx_model_path, tuple(args.input_size))