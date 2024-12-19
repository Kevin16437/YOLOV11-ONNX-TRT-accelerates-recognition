# onnx_inference.py

import onnxruntime
import numpy as np
import cv2
import time
import argparse

def preprocess_image(image_path, input_size=(640, 640)):
    """
    读取并预处理图像。
    
    Args:
        image_path (str): 图像路径。
        input_size (tuple): 模型输入尺寸。
    
    Returns:
        np.ndarray: 预处理后的图像数据。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图像 '{image_path}' 未找到")
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))  # (C, H, W)
    image = np.expand_dims(image, axis=0).astype(np.float32)  # (1, C, H, W)
    image /= 255.0  # 归一化
    return image

def infer_onnx(onnx_model_path, image_path):
    """
    使用 ONNX Runtime 执行推理并记录时间。
    
    Args:
        onnx_model_path (str): ONNX 模型路径。
        image_path (str): 输入图像路径。
    
    Returns:
        float: 推理时间。
        list: 推理输出。
    """
    # 加载 ONNX 模型
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 预处理图像
    input_data = preprocess_image(image_path)
    
    # 执行推理
    start_time = time.time()
    onnx_output = session.run([output_name], {input_name: input_data})
    end_time = time.time()
    
    inference_time = end_time - start_time
    return inference_time, onnx_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Runtime 推理")
    parser.add_argument('--onnx_model_path', type=str, default='model.onnx', help='ONNX 模型路径 (.onnx)')
    parser.add_argument('--image_path', type=str, default='input_image.jpg', help='输入图像路径 (.jpg/.png)')
    args = parser.parse_args()
    
    inference_time, output = infer_onnx(args.onnx_model_path, args.image_path)
    print(f"ONNX Runtime 推理时间: {inference_time:.6f} 秒")