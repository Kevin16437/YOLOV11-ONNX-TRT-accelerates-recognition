import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import statistics

class YOLOTRTInference:
    def __init__(self, engine_path):
        """
        初始化TensorRT推理引擎
        
        :param engine_path: .engine文件的路径
        """
        # 创建TensorRT运行时和推理引擎
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出信息
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        # 分配GPU内存
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        
        # 为每个张量分配内存
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            
            # 获取张量形状和数据类型
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = self.engine.get_tensor_dtype(tensor_name)
            
            # 计算张量大小
            size = trt.volume(shape)
            np_dtype = trt.nptype(dtype)
            
            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, np_dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(cuda_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(host_mem)
            else:
                self.outputs.append(host_mem)

    def preprocess_image(self, image_path, input_shape=(640, 640)):
        """
        预处理图像
        
        :param image_path: 图像路径
        :param input_shape: 模型期望的输入形状
        :return: 预处理后的图像数组
        """
        # 读取图像
        img = cv2.imread(image_path)
        
        # 调整大小
        img = cv2.resize(img, input_shape)
        
        # 标准化 - 归一化到[0,1]
        img = img.astype(np.float32) / 255.0
        
        # 调整维度顺序 (HWC -> CHW)
        img = img.transpose((2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # 加上batch维度, (1, 3, 640, 640)
        
        return img

    def inference(self, input_data):
        """
        执行推理
        
        :param input_data: 预处理后的输入数据
        :return: 推理结果
        """
        try:
            # 检查输入数据的形状，确保是四维的
            print(f"Input data shape: {input_data.shape}")
            if input_data.shape != (1, 3, 640, 640):
                raise ValueError(f"Input data shape is incorrect. Expected (1, 3, 640, 640), but got {input_data.shape}")

            # 将输入数据复制到GPU
            np.copyto(self.inputs[0], input_data.ravel())
            print("Input data copied to host memory.")
            
            # 检查绑定的内存地址
            print(f"Bindings: {self.bindings}")
            print(f"Input shape: {self.inputs[0].shape}, dtype: {self.inputs[0].dtype}")
            
            # 将数据从主机内存复制到设备内存
            cuda.memcpy_htod_async(self.bindings[0], self.inputs[0], self.stream)
            print("Input data copied to device memory.")
            
            # 执行推理
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            print("Inference executed.")
            
            # 将输出从GPU复制回CPU
            cuda.memcpy_dtoh_async(self.outputs[0], self.bindings[1], self.stream)
            
            # 同步流
            self.stream.synchronize()
            print("Output copied to host memory.")
            
            return self.outputs[0]
        except Exception as e:
            print(f"Error during inference: {e}")
            raise

    def postprocess_output(self, output, confidence_threshold=0.5):
        """
        后处理推理输出
        
        :param output: 模型输出
        :param confidence_threshold: 置信度阈值
        :return: 检测结果
        """
        # 默认YOLOv8输出 (1, 5, 8400)
        # 前2行可能是框的坐标，第3行是类别置信度
        output = output.reshape(5, -1)
        
        # 提取置信度和类别
        confidences = output[4, :]
        
        # 筛选高于阈值的检测
        valid_detections = np.where(confidences > confidence_threshold)[0]
        
        return {
            'raw_output': output,
            'valid_detections': valid_detections,
            'num_detections': len(valid_detections)
        }

def main():
    # TensorRT引擎路径
    engine_path = 'best.engine'
    
    # 测试图像路径
    image_path = 'input_image.jpg'  # 请替换为你的测试图像
    
    # 创建TensorRT推理实例
    trt_inference = YOLOTRTInference(engine_path)
    
    # 预处理图像
    input_data = trt_inference.preprocess_image(image_path)
    
    # 进行多次推理以获得稳定的性能测量
    num_runs = 10
    inference_times = []
    
    print(f"开始执行 {num_runs} 次推理测试...")
    
    for _ in range(num_runs):
        # 测量推理时间
        start_time = time.time()
        
        # 执行推理
        output = trt_inference.inference(input_data)
        
        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        inference_times.append(inference_time)
        
        # 后处理输出
        detections = trt_inference.postprocess_output(output)
        
        # 可选：打印每次推理的时间
        print(f"推理时间: {inference_time:.2f} ms, 检测数: {detections['num_detections']}")
    
    # 计算平均推理时间和标准差
    avg_inference_time = statistics.mean(inference_times)
    std_dev_inference_time = statistics.stdev(inference_times)
    
    print("\n推理性能总结:")
    print(f"平均推理时间: {avg_inference_time:.2f} ms")
    print(f"推理时间标准差: {std_dev_inference_time:.2f} ms")

if __name__ == "__main__":
    main()
