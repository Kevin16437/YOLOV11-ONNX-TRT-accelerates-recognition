from ultralytics import YOLO
import cv2
import numpy as np

def predict_image(model_path, image_path):
    # 加载训练好的模型
    model = YOLO(model_path)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像文件")
        return
        
    # 调整图像大小
    max_size = 900  # 设置最大显示尺寸
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        # 计算缩放比例
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # 进行预测
    results = model(image)
    
    # 处理预测结果
    highest_confidence = 0  # 初始化最高置信度
    best_box = None  # 用于存储最高置信度的边界框
    
    for result in results:
        boxes = result.boxes
        
        # 遍历所有检测到的目标
        for box in boxes:
            # 获取置信度
            conf = float(box.conf)
            
            # 仅考虑置信度高于当前最高置信度的目标
            if conf > highest_confidence:
                highest_confidence = conf
                best_box = box
    
    # 如果找到最高置信度的目标，绘制它的边界框
    if best_box is not None:
        # 获取边界框坐标
        x1, y1, x2, y2 = best_box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 获取类别
        cls = int(best_box.cls)
        class_name = model.names[cls]
        
        # 在图像上绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 在边框旁边显示类别和置信度
        label = f'{class_name} {highest_confidence:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_x = x1
        label_y = y1 + 30
        
        # 如果标签超出了图像边界，调整位置
        if label_x + label_size[0] > image.shape[1]:
            label_x = image.shape[1] - label_size[0] - 10
        if label_y < 0:
            label_y = y1 + 20
        
        cv2.putText(image, label, (label_x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 显示结果
    cv2.namedWindow('预测结果', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('预测结果', 800, 600)  # 设置固定窗口大小
    cv2.imshow('预测结果', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果图像
    cv2.imwrite('prediction_result.jpg', image)

if __name__ == "__main__":
    # 设置模型路径和图像路径
    model_path = "F:/Kevin_Study/大三上/智能制造技术/YOLOV11_homework/ultralytics-main1217/ultralytics-main/runs/detect/train5/weights/best.pt"  # 替换为你的模型路径
    image_path = "F:/Kevin_Study/大三上/智能制造技术/YOLOV11_homework/tower_dataset/images/train/89.jpg" # 替换为你的图像路径
    
    # 执行预测
    predict_image(model_path, image_path)
