import json
import os
import shutil

def convert_labelme_to_yolo(labelme_dir, output_dir):
    """
    将labelme生成的json文件转换为yolo格式的txt文件。

    Args:
      labelme_dir: labelme的json文件所在文件夹。
      output_dir: 输出的yolo txt文件所在文件夹。
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(labelme_dir):
        if not filename.endswith(".json"):
            continue
        json_file_path = os.path.join(labelme_dir, filename)
        txt_file_path = os.path.join(output_dir, filename.replace(".json", ".txt"))
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            image_width = data['imageWidth']
            image_height = data['imageHeight']

            with open(txt_file_path, 'w') as txt_file:
                for shape in data['shapes']:
                    label = shape['label']
                    # 假设你的labelme标注中，上海东方明珠的标签为 "oriental_pearl"
                    if label == "oriental_pearl":
                        points = shape['points']
                        x_min = min(points[0][0], points[1][0])
                        x_max = max(points[0][0], points[1][0])
                        y_min = min(points[0][1], points[1][1])
                        y_max = max(points[0][1], points[1][1])

                        center_x = (x_min + x_max) / 2 / image_width
                        center_y = (y_min + y_max) / 2 / image_height
                        width = (x_max - x_min) / image_width
                        height = (y_max - y_min) / image_height
                        # 写入 YOLO 格式的标签
                        txt_file.write(f"0 {center_x} {center_y} {width} {height}\n")

    print(f"Labelme to YOLO conversion complete. Saved to {output_dir}")

# 对训练集、验证集、测试集分别进行转换
labelme_dir = 'tower_dataset/train'  # 替换为你的训练集json文件所在文件夹
output_dir = 'tower_dataset/labels/train'  # 替换为你的训练集txt文件输出文件夹
convert_labelme_to_yolo(labelme_dir, output_dir)

labelme_dir = 'tower_dataset/val'  # 替换为你的验证集json文件所在文件夹
output_dir = 'tower_dataset/labels/val'  # 替换为你的验证集txt文件输出文件夹
convert_labelme_to_yolo(labelme_dir, output_dir)

labelme_dir = 'tower_dataset/test'  # 替换为你的测试集json文件所在文件夹
output_dir = 'tower_dataset/labels/test'  # 替换为你的测试集txt文件输出文件夹
convert_labelme_to_yolo(labelme_dir, output_dir)


# 整理图像文件到相应文件夹
os.makedirs("tower_dataset/images/train", exist_ok=True)
os.makedirs("tower_dataset/images/val", exist_ok=True)
os.makedirs("tower_dataset/images/test", exist_ok=True)

for file in os.listdir("tower_dataset/train"):
  if file.endswith(".jpg"):
    shutil.move(f"tower_dataset/train/{file}", "tower_dataset/images/train")

for file in os.listdir("tower_dataset/val"):
  if file.endswith(".jpg"):
    shutil.move(f"tower_dataset/val/{file}", "tower_dataset/images/val")

for file in os.listdir("tower_dataset/test"):
  if file.endswith(".jpg"):
    shutil.move(f"tower_dataset/test/{file}", "tower_dataset/images/test")
