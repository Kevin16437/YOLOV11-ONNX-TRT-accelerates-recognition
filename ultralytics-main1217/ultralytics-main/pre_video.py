from ultralytics import YOLO
import cv2
import numpy as np

def predict_from_camera(model_path):
    # 加载训练好的模型
    model = YOLO('best.pt')
    
    # 打开摄像头 (0为默认摄像头，其他值为外接摄像头)
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
        
        # 获取预测结果
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 在边框旁边显示类别和置信度
            label = f'{class_name} {highest_confidence:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_x = x1
            label_y = y1 + 30
            
            # 如果标签超出了图像边界，调整位置
            if label_x + label_size[0] > frame.shape[1]:
                label_x = frame.shape[1] - label_size[0] - 10
            if label_y < 0:
                label_y = y1 + 20
            
            cv2.putText(frame, label, (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示每一帧的结果
        cv2.imshow('实时目标检测', frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设置模型路径
    # model = YOLO('yolov8n.pt') 
    # model_path = "F:/Kevin_Study/大三上/智能制造技术/YOLOV11_homework/ultralytics-main1217/ultralytics-main/runs/detect/train5/weights/best.pt"  # 替换为你的模型路径
    
    # 执行实时预测
    predict_from_camera('best.pt')
