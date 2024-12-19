# prune_quantize.py

import torch
import torch.quantization as quantization

def prune_and_quantize(model_path, quantized_model_path):
    """
    加载模型，进行剪枝和量化，并保存量化后的模型。
    
    Args:
        model_path (str): 原始模型路径（.pt 文件）。
        quantized_model_path (str): 量化后模型保存路径。
    """
    # 加载训练好的模型
    # 假设模型是通过 torch.save(model) 保存的整个模型
    model = torch.load(model_path)
    model.eval()
    
    # 动态量化模型
    model_q = quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.Conv2d},  # 需要量化的层类型
        dtype=torch.qint8
    )
    
    # 保存量化后的模型
    torch.save(model_q, quantized_model_path)
    print(f"模型已量化并保存为 '{quantized_model_path}'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型剪枝与量化")
    parser.add_argument('--model_path', type=str, default='model.pt', help='原始模型路径 (.pt)')
    parser.add_argument('--quantized_model_path', type=str, default='quantized_model.pth', help='量化后模型保存路径 (.pth)')

    args = parser.parse_args()
    prune_and_quantize(args.model_path, args.quantized_model_path)