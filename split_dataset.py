import os
import random
import shutil

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8, val_ratio=0.2):
    """
    划分数据集为训练集、验证集和测试集。

    Args:
      image_dir: 图像文件夹路径。
      label_dir: 标签文件夹路径 (这里指labelme的json文件所在文件夹)。
      output_dir: 输出文件夹路径。
      train_ratio: 训练集比例。
      val_ratio: 验证集比例。
    """
    image_files = os.listdir(image_dir)
    random.shuffle(image_files)

    num_images = len(image_files)
    train_num = int(num_images * train_ratio)
    val_num = int(num_images * val_ratio)
    train_files = image_files[:train_num]
    val_files = image_files[train_num:train_num + val_num]
    test_files = image_files[train_num + val_num:]

    train_output_dir = os.path.join(output_dir, "train")
    val_output_dir = os.path.join(output_dir, "val")
    test_output_dir = os.path.join(output_dir, "test")

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    def copy_files(files, output_dir):
        for file in files:
            image_path = os.path.join(image_dir, file)
            label_path = os.path.join(label_dir, file.replace(".jpg", ".json")) # 假设图片格式为jpg
            if os.path.exists(label_path):
                shutil.copy(image_path, os.path.join(output_dir,file))
                shutil.copy(label_path, os.path.join(output_dir,file.replace(".jpg", ".json")))
            else:
                print(f"Warning: label file for {file} not found.")

    copy_files(train_files, train_output_dir)
    copy_files(val_files, val_output_dir)
    copy_files(test_files, test_output_dir)

    print("Dataset split complete.")
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print(f"Testing set: {len(test_files)} images")
    
image_dir = "F:/Kevin_Study/大三上/智能制造技术/YOLOV11_homework/tower"  
label_dir = "F:/Kevin_Study/大三上/智能制造技术/YOLOV11_homework/json"
output_dir = "F:/Kevin_Study/大三上/智能制造技术/YOLOV11_homework/tower_dataset" # 替换为你的输出文件夹路径
split_dataset(image_dir, label_dir, output_dir)
