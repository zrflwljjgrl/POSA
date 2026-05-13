import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# 加载目标图像
original_image = Image.open('inria/Train/pos/crop001001.png').convert("RGB")
original_image_tensor = transforms.ToTensor()(original_image).unsqueeze(0)  # 形状: [1, C, H, W]

# 初始化补丁和掩码
patch_size = (50, 50)
patch = torch.randn(1, 3, patch_size[0], patch_size[1])  # 随机生成补丁

# 定义补丁放置的位置
x, y = 100, 100  # 补丁左上角坐标

# 获取原始图像的高度和宽度
_, _, H, W = original_image_tensor.shape

# 检查补丁是否超出图像边界
if x + patch_size[0] > H or y + patch_size[1] > W:
    raise ValueError("补丁位置加上补丁尺寸超出图像边界，请调整位置或补丁大小。")

# 创建一个与原始图像相同大小的补丁图层
full_patch = torch.zeros_like(original_image_tensor) #全0的tensor变量
full_patch[:, :, x:x+patch_size[0], y:y+patch_size[1]] = patch

# 创建掩码，指定补丁应用的位置
mask = torch.zeros_like(original_image_tensor)
mask[:, :, x:x+patch_size[0], y:y+patch_size[1]] = 1

# 将补丁应用到图像上
patched_image_tensor = mask * full_patch + (1 - mask) * original_image_tensor

# 将张量转换回PIL图像
patched_image = patched_image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
patched_image = np.clip(patched_image * 255, 0, 255).astype(np.uint8)
patched_pil_image = Image.fromarray(patched_image)

# 保存结果图像
patched_pil_image.save('patched_image.png')
print("补丁已成功应用并保存为 'patched_image.png'")