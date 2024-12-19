import os
os.add_dll_directory("D:/TensorRT-10.7.0.23/lib")
import time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime


class TensorRTConverter:
    def __init__(self, logger_severity=trt.Logger.WARNING):
        self.logger = trt.Logger(logger_severity)
        
    def build_engine(self, onnx_path, engine_path, fp16_mode=False, max_workspace_size=1<<30):
        """
        Build TensorRT engine from ONNX model
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            fp16_mode: Enable FP16 precision
            max_workspace_size: Maximum workspace size (default 1GB)
        """
        if os.path.exists(engine_path):
            print(f"Loading existing engine from {engine_path}")
            return self.load_engine(engine_path)
            
        print(f"Building new TensorRT engine from {onnx_path}")
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

        if fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
            
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"ONNX parse error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
                
        # Check and mark output if needed
        last_layer = network.get_layer(network.num_layers - 1)
        if not last_layer.get_output(0).shape:
            network.mark_output(last_layer.get_output(0))
            
        engine = builder.build_serialized_network(network, config)
        with open(engine_path, 'wb') as f:
            f.write(engine)
            
        return engine
        
    def load_engine(self, engine_path):
        """Load TensorRT engine from file"""
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())

class TensorRTInference:
    def __init__(self, engine):
        self.engine = engine
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Get input and output tensor names
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        self.buffers = self._allocate_buffers()
        
    def _allocate_buffers(self):
        """Allocate device buffers for input and output"""
        buffers = {}
        for name in self.input_names + self.output_names:
            # Get tensor shape and dtype
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            
            # Calculate total size of the tensor
            size = np.prod(shape)
            
            # Convert TensorRT type to numpy type
            numpy_dtype = trt.nptype(dtype)
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, numpy_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            buffers[name] = {
                'host': host_mem,
                'device': device_mem,
                'shape': shape
            }
        
        return buffers
        
    def infer(self, input_data, batch_size=1):
        """
        Run inference
        
        Args:
            input_data: Input data as numpy array
            batch_size: Batch size for inference
        """
        # Copy input data to input buffer
        np.copyto(self.buffers[self.input_names[0]]['host'], input_data.ravel())
        
        # Transfer input data to GPU
        cuda.memcpy_htod(self.buffers[self.input_names[0]]['device'], 
                         self.buffers[self.input_names[0]]['host'])
        
        # Prepare bindings for inference
        bindings = [int(self.buffers[name]['device']) for name in self.input_names + self.output_names]
        
        # Run inference
        self.context.execute_v2(bindings=bindings)
        
        # Transfer results from GPU to host
        outputs = []
        for name in self.output_names:
            output = np.empty(self.buffers[name]['shape'], 
                              dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            cuda.memcpy_dtoh(output, self.buffers[name]['device'])
            outputs.append(output)
        
        return outputs

class ImageProcessor:
    @staticmethod
    def preprocess(image_path, input_shape):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to input image
            input_shape: Model input shape (batch, channels, height, width)
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_shape[3], input_shape[2]))
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
        
    @staticmethod
    def postprocess(output, original_shape):
        """
        Postprocess model output
        
        Args:
            output: Model output
            original_shape: Original image shape
        """
        # Customize this method based on your model's output format
        output = output.reshape(-1, 85)  # Adjust according to your model
        return output

def main():
    # Configuration
    onnx_path = "best.onnx"
    engine_path = "best.engine"
    image_path = "input_image.jpg"
    input_shape = (1, 3, 640, 640)
    
    # Create TensorRT engine
    converter = TensorRTConverter()
    engine = converter.build_engine(onnx_path, engine_path, fp16_mode=True)
    
    # Create inference wrapper
    trt_inference = TensorRTInference(engine)
    
    # Process image and run inference
    processor = ImageProcessor()
    input_data = processor.preprocess(image_path, input_shape)
    
    # TensorRT inference
    start_time = time.time()
    trt_outputs = trt_inference.infer(input_data)
    trt_time = (time.time() - start_time) * 1000
    print(f"TensorRT inference time: {trt_time:.2f}ms")
    
    # ONNX Runtime inference for comparison
    session = onnxruntime.InferenceSession(onnx_path)
    start_time = time.time()
    onnx_outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    onnx_time = (time.time() - start_time) * 1000
    print(f"ONNX Runtime inference time: {onnx_time:.2f}ms")
    
    # Process outputs
    original_image = cv2.imread(image_path)
    trt_processed = processor.postprocess(trt_outputs[0], original_image.shape)
    onnx_processed = processor.postprocess(onnx_outputs[0], original_image.shape)
    
    print("TensorRT output shape:", trt_processed.shape)
    print("ONNX Runtime output shape:", onnx_processed.shape)
    
if __name__ == "__main__":
    main()