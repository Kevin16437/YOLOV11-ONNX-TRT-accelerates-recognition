from ultralytics import YOLO
import os

def export_to_onnx(model_path, onnx_path):
    """
    将 YOLOv8 模型导出为 ONNX 格式。

    Args:
        model_path: YOLOv8 模型路径 (best.pt)
        onnx_path: 输出 ONNX 模型路径 (best.onnx)
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        # 加载 YOLO 模型
        model = YOLO(model_path)
        
        # 导出为 ONNX
        model.export(format='onnx', 
                    opset=12,  # ONNX opset version
                    simplify=True)  # 简化 ONNX 模型
        
        print(f"Model successfully exported to ONNX")
    except Exception as e:
        print(f"Error during export: {e}")

if __name__ == "__main__":
    model_path = "best.pt"
    onnx_path = "best.onnx"  # YOLO 会自动在同一目录下生成这个文件
    export_to_onnx(model_path, onnx_path)

# 2import torch
# import onnx
# import os

# def export_to_onnx(model_path, onnx_path, input_shape):
#     """
#     将 PyTorch 模型导出为 ONNX 格式。

#     Args:
#         model_path: PyTorch 模型路径 (best.pt)
#         onnx_path: 输出 ONNX 模型路径 (best.onnx)
#         input_shape: 模型输入形状，例如 (1, 3, 640, 640)
#     """
#     # 检查模型文件是否存在
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found at {model_path}")

#     # 加载 PyTorch 模型
#     model = torch.load(model_path, map_location='cpu')  # 将模型加载到 CPU 上

#     # 确保模型处于评估模式
#     model.eval()

#     # 创建一个虚拟输入
#     dummy_input = torch.randn(input_shape, device='cpu')

#     # 导出模型到 ONNX
#     try:
#         torch.onnx.export(
#             model,
#             dummy_input,
#             onnx_path,
#             export_params=True,
#             opset_version=12,  # 建议使用较新的 opset 版本
#             do_constant_folding=True,
#             input_names=['input'],
#             output_names=['output1','output2'], # 修改这里, 你需要根据你的模型修改
#             dynamic_axes={'input': {0: 'batch_size'},
#                           'output1': {0: 'batch_size'},
#                           'output2': {0: 'batch_size'}} # 修改这里, 你需要根据你的模型修改
#         )

#         print(f"Model successfully exported to ONNX at {onnx_path}")
#     except Exception as e:
#       print(f"Error when exporting, please check output_names and dynamic_axes settings: {e}")

# if __name__ == "__main__":
#     model_path = "best.pt"
#     onnx_path = "best.onnx"
#     input_shape = (1, 3, 640, 640)  # 假设你的模型输入形状是 640x640
#     export_to_onnx(model_path, onnx_path, input_shape)



# #1 import torch
# # import onnx
# # import os

# # def export_to_onnx(model_path, onnx_path, input_shape):
# #     """
# #     将 PyTorch 模型导出为 ONNX 格式。

# #     Args:
# #         model_path: PyTorch 模型路径 (best.pt)
# #         onnx_path: 输出 ONNX 模型路径 (best.onnx)
# #         input_shape: 模型输入形状，例如 (1, 3, 640, 640)
# #     """
# #     # 检查模型文件是否存在
# #     if not os.path.exists(model_path):
# #         raise FileNotFoundError(f"Model file not found at {model_path}")

# #     # 加载 PyTorch 模型
# #     model = torch.load(model_path, map_location='cpu')  # 将模型加载到 CPU 上

# #     # 确保模型处于评估模式
# #     model.eval()

# #     # 创建一个虚拟输入
# #     dummy_input = torch.randn(input_shape, device='cpu')

# #     # 导出模型到 ONNX
# #     torch.onnx.export(
# #         model,
# #         dummy_input,
# #         onnx_path,
# #         export_params=True,
# #         opset_version=12,  # 建议使用较新的 opset 版本
# #         do_constant_folding=True,
# #         input_names=['input'],
# #         output_names=['output'],
# #         dynamic_axes={'input': {0: 'batch_size'},
# #                       'output': {0: 'batch_size'}}
# #     )

# #     print(f"Model successfully exported to ONNX at {onnx_path}")

# # if __name__ == "__main__":
# #     model_path = "best.pt"
# #     onnx_path = "best.onnx"
# #     input_shape = (1, 3, 640, 640)  # 假设你的模型输入形状是 640x640
# #     export_to_onnx(model_path, onnx_path, input_shape)