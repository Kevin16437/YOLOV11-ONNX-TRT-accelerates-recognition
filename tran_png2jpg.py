import os
from PIL import Image

def convert_png_to_jpg(directory):
    """
    将指定目录下的所有 PNG 文件转换为 JPG 文件。

    Args:
      directory: 要处理的目录的路径。
    """

    for filename in os.listdir(directory):
        if filename.lower().endswith(".png"):
            png_path = os.path.join(directory, filename)
            jpg_path = os.path.join(directory, os.path.splitext(filename)[0] + ".jpg")

            try:
                img = Image.open(png_path)
                img = img.convert('RGB')  # 将图像转换为 RGB 模式，以便保存为 JPG
                img.save(jpg_path, "JPEG")
                print(f"Converted: {filename} -> {os.path.basename(jpg_path)}")
                os.remove(png_path)  # 删除原始 PNG 文件（可选）
            except Exception as e:
                 print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    target_directory = r"F:\F Download\photo (2)\photo"  # 替换为你的目标目录
    convert_png_to_jpg(target_directory)
    print("Conversion complete!")
