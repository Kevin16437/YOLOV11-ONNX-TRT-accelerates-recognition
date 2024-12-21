from ultralytics import YOLO
import cv2
import os
import numpy as np

def predict_from_camera(model_path):
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在！")
        return
    
    try:
        # 加载训练好的模型
        model = YOLO(model_path)
        print(f"成功加载模型：{model_path}")
    except Exception as e:
        print(f"加载模型时出错：{str(e)}")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法访问摄像头")
        return
    
    while True:
        # 读取每一帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break
        
        # 进行预测
        results = model(frame)
        
        for result in results:
            boxes = result.boxes
            
            # 遍历所有检测到的目标
            for box in boxes:
                # 获取置信度
                conf = float(box.conf)
                
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 获取类别
                cls = int(box.cls)
                class_name = model.names[cls]
                
                # 根据置信度决定框的颜色
                # 置信度 >= 0.7 使用绿色，否则使用红色
                color = (0, 255, 0) if conf >= 0.7 else (0, 0, 255)
                
                # 在图像上绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 在边框旁边显示类别和置信度
                label = f'{class_name} {conf:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_x = x1
                label_y = y1 - 10  # 将标签放在框的上方
                
                # 如果标签超出了图像边界，调整位置
                if label_x + label_size[0] > frame.shape[1]:
                    label_x = frame.shape[1] - label_size[0] - 10
                if label_y < 0:
                    label_y = y1 + 20
                
                # 添加半透明的背景以使文字更容易读取
                cv2.rectangle(frame, 
                            (label_x, label_y - 20), 
                            (label_x + label_size[0], label_y + label_size[1] - 20), 
                            color, 
                            cv2.FILLED)
                
                # 添加文字标签
                cv2.putText(frame, 
                           label, 
                           (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, 
                           (255, 255, 255),  # 白色文字
                           2)

        # 显示每一帧的结果
        cv2.imshow('实时目标检测', frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 指定完整的模型路径
    model_path = os.path.join(os.path.dirname(__file__), 'yolov8x.pt')
    print(f"尝试加载模型：{model_path}")
    predict_from_camera(model_path)