from PIL import Image
import os


def resize_image(input_path, output_path, max_size=1000):
    """按比例缩小图片，最长边不超过 max_size"""
    with Image.open(input_path) as img:
        # 获取原始尺寸
        width, height = img.size

        # 计算新尺寸（保持长宽比）
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        # 调整大小（使用高质量抗锯齿）
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # 保存（保持 PNG 无损）
        resized_img.save(output_path, "PNG", optimize=True)


# 示例：处理单个文件
input_path = "ren.jpg"
output_path = "ren/output_resized.jpg"
resize_image(input_path, output_path)