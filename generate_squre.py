import numpy as np
from PIL import Image
import torch
import os
import cv2
import matplotlib.pyplot as plt
def patch(image_data,H,W,py,px,s):

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
    # 恢复为 (C, H, W) 形状
    resized_image_data = np.transpose(resized_image_data, (2, 0, 1))
    # Step 7: 缩放图像到 s*s
    scaled_image = Image.fromarray(np.transpose(resized_image_data, (1, 2, 0)).astype(np.uint8))  # 转换回 (H, W, C)
    scaled_image = scaled_image.resize((s, s))  # 缩放到 30x30
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
    # Step 8: 创建掩码，只在补丁区域内赋值为 1
    square_mask = np.zeros_like(resized_image_data)
    square_mask[:, y_start:y_end, x_start:x_end] = 255
    # 将缩放后的图像放入全零图像的指定位置
    output_image[:, y_start:y_end, x_start:x_end] = scaled_image_data[:, 0:(y_end - y_start), 0:(x_end - x_start)]
    # 将通道维度（3）移动到最后，形成 (976, 818, 3) 的形状
    output_image_rgb = np.transpose(output_image, (1, 2, 0))
    output_image_rgb = output_image_rgb / 255.0
    #显示图像
    # plt.imshow(output_image_rgb)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show(block=True)
    return output_image,square_mask #到这里都没有问题，就是贴到图片上的时候出现了问题
def apply_mask_to_image(base_image, output_image, square_mask):
    """
    将 mask 叠加到原图上。
    :param base_image: 原始图片，形状为 (3, H, W) 的 numpy 数组
    :param mask: mask 图片，形状为 (3, H, W) 的 numpy 数组
    :return: 叠加后的图片，形状为 (3, H, W)
    """
    # 确保 base_image 和 mask 的形状一致
    assert base_image.shape == output_image.shape, "Base image and mask must have the same shape!"
    # 归一化后叠加
    base_image=base_image/255.0
    output_image=output_image/255.0
    square_mask=square_mask/255.0
    #将补丁通过掩码贴到图片上
    combined_image = base_image * (1 - square_mask) + square_mask * output_image
    # 将通道维度（3）移动到最后，形成 (976, 818, 3) 的形状
    output_image_rgb = np.transpose(combined_image, (1, 2, 0))
    # 保存图片的路径
    # save_path = "workspace/cross_modal_patch_attack/yinshenyi-result/square"
    # os.makedirs(save_path, exist_ok=True)  # 确保目录存在
    # save_file = os.path.join(save_path, file_name)
    # # 由于 output_image_rgb 归一化了，需要转换为 uint8 格式
    # save_img = (output_image_rgb * 255).astype(np.uint8)
    # # 使用 OpenCV 保存图片
    # cv2.imwrite(save_file, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
    # print(f"图片已保存至: {save_file}")
    # 显示图像
    # plt.imshow(output_image_rgb)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show(block=True)
    return combined_image
def generate_square_patch(image_path,H,W,px,py,s):
    # 加载隐身衣的
    # image_data = np.load('patch1000.npy')
    #加载yolo对抗的
    # image = cv2.imread('object_score.png')
    # image = cv2.imread('empatch-y2.png')
    image = cv2.imread('DOEPatch-yyf.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = np.transpose(image, (2, 0, 1))
    image_data = np.expand_dims(image_array, axis=0)
    output_image,square_mask = patch(image_data, H, W, px, py, s)
    # 加载原始图片
    base_image_pil = Image.open(image_path).convert("RGB")  # 打开图片并确保为 RGB 格式
    base_image = np.array(base_image_pil)  # 转换为 numpy 数组，形状为 (H, W, 3)
    # 转换为 (3, H, W) 格式
    base_image = np.transpose(base_image, (2, 0, 1))
    # 确保 mask 和 base_image 形状一致
    if output_image.shape != base_image.shape:
        raise ValueError(f"Mask shape {output_image.shape} does not match base image shape {base_image.shape}!")
    # 调用函数将 mask 叠加到原图上
    combined_image = apply_mask_to_image(base_image,output_image,square_mask) #base_image原始图片，output_image就是在指定位置贴上补丁，其余位置的像素为0
    # 转换为 (H, W, C) 以使用 PIL 或 matplotlib 可视化
    combined_image_visual = np.transpose(combined_image, (1, 2, 0))#这里是归一化的，如果你直接转unit8的话小于1的值全变为0了
    # 可视化结果（需要将上面的combined_image_visual反归一化，然后转为unit8类型的，不然plt会报错）
    # plt.imshow(combined_image_visual)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show(block=True)
    return combined_image_visual
def calculate_iou(box1, box2):
    box1 = torch.tensor(box1)
    box2 = torch.tensor(box2)
    box1 = box1.unsqueeze(0)
    N = box1.size(0)
    K = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    ious = inter / union_area

    return ious

def find_max_iou(box, boxes):
    """
    找到与 box 的 IoU 最大的边界框及其索引。

    参数:
    box -- [x1, y1, x2, y2]，单个边界框
    boxes -- [[x1, y1, x2, y2], ...]，多个边界框

    返回:
    max_iou -- 最大的 IoU 值
    max_index -- 最大 IoU 对应的索引
    """
    iou_values = calculate_iou(box, boxes)
    max_iou, max_index = torch.max(iou_values, dim=-1)
    return max_iou.item(), max_index.item()
# if __name__=='__main__':
#     H=976
#     W=818
#     px=114.70
#     py=493.29
#     s=121
#     square_patch=generate_square_patch(H,W,px,py,s)
#     print(3)




