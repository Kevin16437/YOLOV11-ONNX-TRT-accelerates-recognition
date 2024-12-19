# generate_py_files.ps1

$files = @{
    "prune_quantize.py" = @"
import torch
import torch.quantization as quantization

def prune_and_quantize(model_path, quantized_model_path):
    \"\"\"
    加载模型，进行剪枝和量化，并保存量化后的模型。
    
    Args:
        model_path (str): 原始模型路径 (.pt)。
        quantized_model_path (str): 量化后模型保存路径 (.pth)。
    \"\"\"
    # 加载训练好的模型
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
"@

    "convert_to_onnx.py" = @"
import torch.onnx
import torch

def convert_to_onnx(model_path, onnx_model_path, input_size=(1, 3, 640, 640)):
    \"\"\"
    将 PyTorch 模型转换为 ONNX 格式。
    
    Args:
        model_path (str): PyTorch 模型路径（量化后的模型）。
        onnx_model_path (str): 输出的 ONNX 模型路径。
        input_size (tuple): 模型输入尺寸。
    \"\"\"
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
"@

    "tensorrt_inference.py" = @"
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import time
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def infer(engine, input_data):
    # 创建执行上下文
    context = engine.create_execution_context()
    
    # 获取输入和输出绑定
    input_binding_idx = engine.get_binding_index('input')
    output_binding_idx = engine.get_binding_index('output')
    
    # 分配 GPU 内存
    d_input = cuda.mem_alloc(input_data.nbytes)
    output_shape = engine.get_binding_shape(output_binding_idx)
    output_size = trt.volume(output_shape) * input_data.dtype.itemsize
    d_output = cuda.mem_alloc(output_size)
    
    # 将输入数据拷贝到 GPU
    cuda.memcpy_htod(d_input, input_data)
    
    # 准备输出缓冲区
    output_data = np.empty(output_shape, dtype=np.float32)
    
    # 执行推理
    bindings = [int(d_input), int(d_output)]
    
    start_time = time.time()
    context.execute_v2(bindings=bindings)
    end_time = time.time()
    
    # 将输出数据从 GPU 拷贝回主机
    cuda.memcpy_dtoh(output_data, d_output)
    
    # 计算推理时间
    inference_time = end_time - start_time
    return inference_time, output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT 引擎推理")
    parser.add_argument('--engine_path', type=str, default='model.trt', help='TensorRT 引擎路径 (.trt)')
    args = parser.parse_args()
    
    # 加载 TensorRT 引擎
    trt_runtime = trt.Runtime(TRT_LOGGER)
    engine = load_engine(trt_runtime, args.engine_path)
    print(f"已加载 TensorRT 引擎 '{args.engine_path}'")
    
    # 准备输入数据
    input_shape = engine.get_binding_shape(0)  # 假设第一个绑定是输入
    input_size = np.prod(input_shape)
    input_data = np.random.random(input_shape).astype(np.float32)
    
    # 执行推理
    inference_time, output = infer(engine, input_data)
    print(f"TensorRT 推理时间: {inference_time:.6f} 秒")
"@

    "onnx_inference.py" = @"
import onnxruntime
import numpy as np
import cv2
import time
import argparse

def preprocess_image(image_path, input_size=(640, 640)):
    \"\"\"
    读取并预处理图像。
    
    Args:
        image_path (str): 图像路径。
        input_size (tuple): 模型输入尺寸。
    
    Returns:
        np.ndarray: 预处理后的图像数据。
    \"\"\"
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
    \"\"\"
    使用 ONNX Runtime 执行推理并记录时间。
    
    Args:
        onnx_model_path (str): ONNX 模型路径。
        image_path (str): 输入图像路径。
    
    Returns:
        float: 推理时间。
        list: 推理输出。
    \"\"\"
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
"@

    "compare_inference_time.py" = @"
import subprocess
import argparse

def run_prune_quantize(model_path, quantized_model_path):
    cmd = ['python', 'prune_quantize.py', '--model_path', model_path, '--quantized_model_path', quantized_model_path]
    subprocess.run(cmd, check=True)

def run_convert_to_onnx(quantized_model_path, onnx_model_path, input_size):
    cmd = [
        'python', 'convert_to_onnx.py', 
        '--model_path', quantized_model_path, 
        '--onnx_model_path', onnx_model_path, 
        '--input_size'
    ] + [str(i) for i in input_size]
    subprocess.run(cmd, check=True)

def run_convert_to_tensorrt(onnx_model_path):
    cmd = ['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'convert_to_tensorrt.ps1', '-OnnxModelPath', onnx_model_path]
    subprocess.run(cmd, check=True)

def run_tensorrt_inference(engine_path):
    cmd = ['python', 'tensorrt_inference.py', '--engine_path', engine_path]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    trt_time = None
    for line in result.stdout.split('\n'):
        if "TensorRT 推理时间" in line:
            trt_time = float(line.strip().split(":")[1].split()[0])
            print(f"TensorRT 推理时间: {trt_time:.6f} 秒")
            break
    return trt_time

def run_onnx_inference(onnx_model_path, image_path):
    cmd = ['python', 'onnx_inference.py', '--onnx_model_path', onnx_model_path, '--image_path', image_path]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    onnx_time = None
    for line in result.stdout.split('\n'):
        if "ONNX Runtime 推理时间" in line:
            onnx_time = float(line.strip().split(":")[1].split()[0])
            print(f"ONNX Runtime 推理时间: {onnx_time:.6f} 秒")
            break
    return onnx_time

def main():
    parser = argparse.ArgumentParser(description="比较 TensorRT 和 ONNX Runtime 推理时间")
    parser.add_argument('--model_path', type=str, default='model.pt', help='原始模型路径 (.pt)')
    parser.add_argument('--quantized_model_path', type=str, default='quantized_model.pth', help='量化后模型路径 (.pth)')
    parser.add_argument('--onnx_model_path', type=str, default='model.onnx', help='ONNX 模型路径 (.onnx)')
    parser.add_argument('--image_path', type=str, default='input_image.jpg', help='输入图像路径 (.jpg/.png)')
    parser.add_argument('--input_size', type=int, nargs=4, default=[1, 3, 640, 640], help='模型输入尺寸 (例如: 1 3 640 640)')
    args = parser.parse_args()
    
    # 步骤1: 模型剪枝与量化
    print("步骤1: 模型剪枝与量化")
    run_prune_quantize(args.model_path, args.quantized_model_path)
    
    # 步骤2: 转换为 ONNX
    print("步骤2: 转换模型为 ONNX 格式")
    run_convert_to_onnx(args.quantized_model_path, args.onnx_model_path, args.input_size)
    
    # 步骤3: 转换为 TensorRT 引擎
    print("步骤3: 使用 TensorRT 转换 ONNX 为 TensorRT 引擎")
    run_convert_to_tensorrt(args.onnx_model_path)
    
    # 步骤4: TensorRT 推理
    print("步骤4: TensorRT 引擎推理")
    trt_time = run_tensorrt_inference('model.trt')
    
    # 步骤5: ONNX Runtime 推理
    print("步骤5: ONNX Runtime 推理")
    onnx_time = run_onnx_inference(args.onnx_model_path, args.image_path)
    
    # 步骤6: 比较推理时间
    if trt_time and onnx_time:
        speedup = onnx_time / trt_time
        print(f"TensorRT 加速比: {speedup:.2f}x")
    else:
        print("无法计算加速比，请检查推理时间是否正确记录。")

if __name__ == "__main__":
    main()
"@

    "convert_to_tensorrt.ps1" = @"
# convert_to_tensorrt.ps1

Param(
    [Parameter(Mandatory=$true)]
    [string]$OnnxModelPath
)

$TrtEngine = "model.trt"

# 执行 TensorRT 转换
trtexec --onnx=$OnnxModelPath --saveEngine=$TrtEngine `
        --minShapes=input:1x3x640x640 `
        --optShapes=input:1x3x640x640 `
        --maxShapes=input:1x3x640x640 `
        --workspace=4096 --fp16

Write-Host "TensorRT 引擎已保存为 '$TrtEngine'"
"@
}

foreach ($file in $files.Keys) {
    Set-Content -Path $file -Value $files[$file]
    Write-Host "已生成文件: $file"
}

Write-Host "所有 Python 文件已成功生成。"