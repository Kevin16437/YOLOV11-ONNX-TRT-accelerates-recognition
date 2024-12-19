import json
import os

def convert_labelme_to_yolo(json_file, output_dir):
    """
    将 LabelMe JSON 标注文件转换为 YOLO 格式的 TXT 标注文件。
    可以同时处理 'rectangle' 和 'polygon' 两种 shape_type。

    Args:
        json_file (str): LabelMe JSON 标注文件的路径。
        output_dir (str): 输出 YOLO TXT 标注文件的目录。
    """
    print(f"Processing file: {json_file}")  # 添加调试信息
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading json file {json_file} or invalid JSON format: {e}")
        return

    if 'shapes' not in data or 'imagePath' not in data or 'imageWidth' not in data or 'imageHeight' not in data:
        print(f"Error: invalid labelme json format. Expected 'shapes', 'imagePath', 'imageWidth' and 'imageHeight' keys in {json_file}")
        return

    image_path = data['imagePath']
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(output_dir, f"{image_name}.txt")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for shape in data['shapes']:
            if 'label' not in shape or not isinstance(shape,dict):
               print(f"Warning: skip annotation without label info or invalid shape format: {shape}")
               continue
            label = shape['label']
             # Assuming single class, you may need to adjust if you have multiple classes
            class_id = 0  # Default class_id (you might need to define a class mapping)

            if shape['shape_type'] == 'rectangle':
                 if  'points' not in shape or not isinstance(shape['points'],list) or len(shape['points']) !=2:
                      print(f"Warning: skip rectangle annotation without points info or invalid points format: {shape}")
                      continue
                 points = shape['points']
                 min_x = min(points[0][0], points[1][0])
                 min_y = min(points[0][1], points[1][1])
                 max_x = max(points[0][0], points[1][0])
                 max_y = max(points[0][1], points[1][1])
                 bbox_width = max_x - min_x
                 bbox_height = max_y - min_y

            elif shape['shape_type'] == 'polygon':
                  if 'points' not in shape or not isinstance(shape['points'],list):
                     print(f"Warning: skip polygon annotation without points info or invalid points format: {shape}")
                     continue
                  points = shape['points']
                  if not points:
                     print(f"Warning: skip polygon annotation with no points in {shape}")
                     continue
                  min_x = float('inf')
                  min_y = float('inf')
                  max_x = float('-inf')
                  max_y = float('-inf')
                  for x, y in points:
                     min_x = min(min_x, x)
                     min_y = min(min_y, y)
                     max_x = max(max_x, x)
                     max_y = max(max_y, y)
                  bbox_width = max_x - min_x
                  bbox_height = max_y - min_y
            else:
                  print(f"Warning: skip annotation with unsupported shape type: {shape}")
                  continue
            x_center = (min_x + bbox_width / 2) / image_width
            y_center = (min_y + bbox_height / 2) / image_height
            width = bbox_width / image_width
            height = bbox_height / image_height
            outfile.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")



def main():
    current_dir = os.getcwd()
    print(f"Current directory:{current_dir}") # 添加调试信息
    output_dir = os.path.join(current_dir, 'labels')
    os.makedirs(output_dir, exist_ok=True)
    print(os.listdir(current_dir))  # 打印当前目录下的文件
    
    for filename in os.listdir(current_dir):
        if filename.lower().endswith('.json'):
            json_path = os.path.join(current_dir, filename)
            convert_labelme_to_yolo(json_path, output_dir)

    print("Conversion completed. TXT labels are in 'labels' folder.")

if __name__ == "__main__":
    main()