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