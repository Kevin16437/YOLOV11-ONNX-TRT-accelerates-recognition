import json
import os

def convert_json_to_yolo(json_file, output_dir):
    """
    将 JSON 标注文件转换为 YOLO 格式的 TXT 标注文件。

    Args:
        json_file (str): JSON 标注文件的路径。
        output_dir (str): 输出 TXT 标注文件的目录。
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
             data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading json file {json_file} or invalid JSON format: {e}")
        return


    if 'images' not in data or 'annotations' not in data or not isinstance(data,dict):
        print(f"Error: invalid json format. Expected 'images' and 'annotations' keys in {json_file}")
        return


    images_dict = {img['id']: img for img in data['images']}

    # Create a dictionary to map annotation ids to a list of annotations
    annotation_dict = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotation_dict:
            annotation_dict[image_id] = []
        annotation_dict[image_id].append(annotation)


    for image_id, image_annotations in annotation_dict.items():
            image = images_dict.get(image_id)
            if not image:
                 print(f"Warning: could not find the image info for {image_id}")
                 continue
            image_width = image['width']
            image_height = image['height']
            image_name = os.path.splitext(image['file_name'])[0] # use same name but .txt format
            output_file = os.path.join(output_dir, f"{image_name}.txt")
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for annotation in image_annotations:
                    # Assuming the box is in the form [x, y, width, height]
                   if "bbox" not in annotation or "category_id" not in annotation:
                        print(f"Warning: skip annotation without bbox or category_id info")
                        continue;
                   bbox = annotation['bbox']
                   class_id = annotation['category_id']
                   x_center = (bbox[0] + bbox[2] / 2) / image_width
                   y_center = (bbox[1] + bbox[3] / 2) / image_height
                   width = bbox[2] / image_width
                   height = bbox[3] / image_height
                   outfile.write(f"{class_id -1 if class_id > 0 else class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n") #class_id start from 0


def main():
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, 'labels')  # 输出到当前目录下的 labels 文件夹
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(current_dir):
        if filename.lower().endswith('.json'):
            json_path = os.path.join(current_dir, filename)
            convert_json_to_yolo(json_path, output_dir)

    print("Conversion completed. TXT labels are in 'labels' folder.")


if __name__ == "__main__":
    main()