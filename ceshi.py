import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def patch(image_data,H,W,px,py,s):

    # Step 2: 直接处理图像数据
    # 由于批次维度为 1，直接取出图像数据
    image_data = image_data[0]  # 获取形状为 (C, H, W) 的图像数据

    # Step 3: 转换为 (H, W, C) 形式，以便使用 PIL
    image_data = np.transpose(image_data, (1, 2, 0))  # 转换为 (H, W, C)

    # 确保图像数据的数值范围在 [0, 255] 之间
    if image_data.max() <= 1:
        image_data = (image_data * 255).astype(np.uint8)
    else:
        image_data = image_data.astype(np.uint8)

    # Step 4: 将 numpy 数组转换为 PIL 图像对象
    image_pil = Image.fromarray(image_data)

    # Step 5: 调整图像大小
    image_resized = image_pil.resize((W, H))  # 调整到 976x818 的大小

    # Step 6: 转换回 numpy 数组，并恢复为 (C, H, W) 形状
    resized_image_data = np.array(image_resized)
    plt.imshow(resized_image_data)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    # 恢复为 (C, H, W) 形状
    resized_image_data = np.transpose(resized_image_data, (2, 0, 1))

    # 查看调整后的图像形状
    print("调整后的图像数据形状:", resized_image_data.shape)

    # Step 7: 缩放图像到 30x30
    scaled_image = Image.fromarray(np.transpose(resized_image_data, (1, 2, 0)).astype(np.uint8))  # 转换回 (H, W, C)
    scaled_image = scaled_image.resize((121, 121))  # 缩放到 30x30

    # 转换为 numpy 数组
    scaled_image_data = np.array(scaled_image)

    # 恢复为 (C, H, W) 形状
    scaled_image_data = np.transpose(scaled_image_data, (2, 0, 1))

    # Step 8: 创建一个全零图像
    output_image = np.zeros_like(resized_image_data)

    # Step 9: 计算目标位置 (x=506, y=114) 的中心
    center_x, center_y = px, py
    w, h = s, s

    # 计算图像放置的位置，确保不超出图像的边界
    x_start = int(max(center_x - w // 2, 0))
    y_start = int(max(center_y - h // 2, 0))
    x_end = int(min(center_x + w // 2, resized_image_data.shape[2]))
    y_end = int(min(center_y + h // 2, resized_image_data.shape[1]))

    # 将缩放后的图像放入全零图像的指定位置
    output_image[:, y_start:y_end, x_start:x_end] = scaled_image_data[:, 0:(y_end - y_start), 0:(x_end - x_start)]

    # Step 10: 可视化结果
    output_image_pil = Image.fromarray(np.transpose(output_image, (1, 2, 0)).astype(np.uint8))
    plt.imshow(output_image_pil)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    return output_image
if __name__=='__main__':
    image_data = np.load('patch1000.npy')
    H=976
    W=818
    px=114.70
    py=493.29
    s=121
    mask=patch(image_data,H,W,px,py,s)
    print(mask)

