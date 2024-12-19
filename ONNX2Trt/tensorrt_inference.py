# tensorrt_inference.py

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