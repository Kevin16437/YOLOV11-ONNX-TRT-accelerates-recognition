import subprocess
import argparse
import sys

def run_command(cmd):
    """通用函数执行命令并捕获输出"""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        sys.exit(1)

def run_prune_quantize(model_path, quantized_model_path):
    cmd = ['python', 'prune_quantize.py', '--model_path', model_path, '--quantized_model_path', quantized_model_path]
    run_command(cmd)

def run_convert_to_onnx(quantized_model_path, onnx_model_path, input_size):
    cmd = [
        'python', 'convert_to_onnx.py', 
        '--model_path', quantized_model_path, 
        '--onnx_model_path', onnx_model_path, 
        '--input_size'
    ] + [str(i) for i in input_size]
    run_command(cmd)

def run_convert_to_tensorrt(onnx_model_path):
    cmd = ['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'convert_to_tensorrt.ps1', '-OnnxModelPath', onnx_model_path]
    run_command(cmd)

def extract_inference_time(output, keyword):
    """从命令输出中提取推理时间"""
    time = None
    for line in output.split('\n'):
        if keyword in line:
            time = float(line.strip().split(":")[1].split()[0])
            print(f"{keyword} 推理时间: {time:.6f} 秒")
            break
    return time

def run_tensorrt_inference(engine_path):
    cmd = ['python', 'tensorrt_inference.py', '--engine_path', engine_path]
    result = run_command(cmd)
    return extract_inference_time(result, "TensorRT")

def run_onnx_inference(onnx_model_path, image_path):
    cmd = ['python', 'onnx_inference.py', '--onnx_model_path', onnx_model_path, '--image_path', image_path]
    result = run_command(cmd)
    return extract_inference_time(result, "ONNX Runtime")

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

# # compare_inference_time.py

# import subprocess
# import argparse

# def run_prune_quantize(model_path, quantized_model_path):
#     cmd = ['python', 'prune_quantize.py', '--model_path', model_path, '--quantized_model_path', quantized_model_path]
#     subprocess.run(cmd, check=True)

# def run_convert_to_onnx(quantized_model_path, onnx_model_path, input_size):
#     cmd = [
#         'python', 'convert_to_onnx.py', 
#         '--model_path', quantized_model_path, 
#         '--onnx_model_path', onnx_model_path, 
#         '--input_size'
#     ] + [str(i) for i in input_size]
#     subprocess.run(cmd, check=True)

# def run_convert_to_tensorrt(onnx_model_path):
#     cmd = ['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'convert_to_tensorrt.ps1', '-OnnxModelPath', onnx_model_path]
#     subprocess.run(cmd, check=True)

# def run_tensorrt_inference(engine_path):
#     cmd = ['python', 'tensorrt_inference.py', '--engine_path', engine_path]
#     result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#     trt_time = None
#     for line in result.stdout.split('\n'):
#         if "TensorRT 推理时间" in line:
#             trt_time = float(line.strip().split(":")[1].split()[0])
#             print(f"TensorRT 推理时间: {trt_time:.6f} 秒")
#             break
#     return trt_time

# def run_onnx_inference(onnx_model_path, image_path):
#     cmd = ['python', 'onnx_inference.py', '--onnx_model_path', onnx_model_path, '--image_path', image_path]
#     result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#     onnx_time = None
#     for line in result.stdout.split('\n'):
#         if "ONNX Runtime 推理时间" in line:
#             onnx_time = float(line.strip().split(":")[1].split()[0])
#             print(f"ONNX Runtime 推理时间: {onnx_time:.6f} 秒")
#             break
#     return onnx_time

# def main():
#     parser = argparse.ArgumentParser(description="比较 TensorRT 和 ONNX Runtime 推理时间")
#     parser.add_argument('--model_path', type=str, default='model.pt', help='原始模型路径 (.pt)')
#     parser.add_argument('--quantized_model_path', type=str, default='quantized_model.pth', help='量化后模型路径 (.pth)')
#     parser.add_argument('--onnx_model_path', type=str, default='model.onnx', help='ONNX 模型路径 (.onnx)')
#     parser.add_argument('--image_path', type=str, default='input_image.jpg', help='输入图像路径 (.jpg/.png)')
#     parser.add_argument('--input_size', type=int, nargs=4, default=[1, 3, 640, 640], help='模型输入尺寸 (例如: 1 3 640 640)')
#     args = parser.parse_args()
    
#     # 步骤1: 模型剪枝与量化
#     print("步骤1: 模型剪枝与量化")
#     run_prune_quantize(args.model_path, args.quantized_model_path)
    
#     # 步骤2: 转换为 ONNX
#     print("步骤2: 转换模型为 ONNX 格式")
#     run_convert_to_onnx(args.quantized_model_path, args.onnx_model_path, args.input_size)
    
#     # 步骤3: 转换为 TensorRT 引擎
#     print("步骤3: 使用 TensorRT 转换 ONNX 为 TensorRT 引擎")
#     run_convert_to_tensorrt(args.onnx_model_path)
    
#     # 步骤4: TensorRT 推理
#     print("步骤4: TensorRT 引擎推理")
#     trt_time = run_tensorrt_inference('model.trt')
    
#     # 步骤5: ONNX Runtime 推理
#     print("步骤5: ONNX Runtime 推理")
#     onnx_time = run_onnx_inference(args.onnx_model_path, args.image_path)
    
#     # 步骤6: 比较推理时间
#     if trt_time and onnx_time:
#         speedup = onnx_time / trt_time
#         print(f"TensorRT 加速比: {speedup:.2f}x")
#     else:
#         print("无法计算加速比，请检查推理时间是否正确记录。")

# if __name__ == "__main__":
#     main()