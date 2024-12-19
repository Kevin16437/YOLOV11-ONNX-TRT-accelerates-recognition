import os
from PIL import Image

def convert_and_rename_images(directory):
    """
    将指定目录下所有图片文件转换为 JPG 格式，并按顺序重命名。

    Args:
      directory: 要处理的目录的路径。
    """
    count = 1
    for filename in os.listdir(directory):
      if any(filename.lower().endswith(ext) for ext in ['.png', '.jpeg', '.jpg', '.bmp', '.gif', '.tiff']):
          file_path = os.path.join(directory, filename)
          jpg_path = os.path.join(directory, str(count) + ".jpg")

          try:
            img = Image.open(file_path)
            img = img.convert('RGB')
            img.save(jpg_path, "JPEG")
            print(f"Converted & Renamed: {filename} -> {count}.jpg")
            count += 1
            if file_path != jpg_path: # 防止覆盖新创建的jpg文件
              os.remove(file_path) # 删除原文件
          except Exception as e:
               print(f"Error processing {filename}: {e}")
    print("Conversion and Renaming complete!")


if __name__ == "__main__":
    target_directory = r"F:\Kevin_Study\大三上\智能制造技术\YOLOV11_homework\tower"  # 替换为你的目标目录
    convert_and_rename_images(target_directory)