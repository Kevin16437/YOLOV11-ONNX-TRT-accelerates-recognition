import tensorrt as trt
import numpy as np

def print_engine_info(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    
    try:
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            # 反序列化引擎
            engine = runtime.deserialize_cuda_engine(f.read())
            
            # 创建执行上下文
            context = engine.create_execution_context()
            
            # 打印引擎基本信息
            print("引擎信息:")
            
            # 使用不同的方法获取绑定数量
            try:
                # 方法1：尝试获取绑定数量
                binding_count = engine.num_io_tensors
                print(f"张量数量: {binding_count}")
            except Exception as e:
                print(f"获取张量数量失败: {e}")
            
            # 遍历并打印每个张量信息
            for i in range(engine.num_io_tensors):
                try:
                    # 获取张量名称
                    tensor_name = engine.get_tensor_name(i)
                    
                    # 获取张量形状
                    try:
                        tensor_shape = engine.get_tensor_shape(tensor_name)
                    except Exception as shape_error:
                        tensor_shape = "无法获取形状"
                    
                    # 获取张量数据类型
                    try:
                        tensor_dtype = engine.get_tensor_dtype(tensor_name)
                    except Exception as dtype_error:
                        tensor_dtype = "无法获取数据类型"
                    
                    # 判断是输入还是输出
                    is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
                    
                    print(f"\n张量 {i}:")
                    print(f"  名称: {tensor_name}")
                    print(f"  是输入: {is_input}")
                    print(f"  形状: {tensor_shape}")
                    print(f"  数据类型: {tensor_dtype}")
                    
                except Exception as tensor_error:
                    print(f"处理张量 {i} 时发生错误: {tensor_error}")
    
    except Exception as e:
        print(f"加载引擎文件时发生错误: {e}")

# 使用你的引擎文件路径
print_engine_info('best.engine')