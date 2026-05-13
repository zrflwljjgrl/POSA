import os
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import math

from pandas._libs.parsers import k

from attack_utils.spline_inria import spline_multi_mask, get_multi_mask
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

from pytorchyolov3.models import load_model
from pytorchyolov3.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from yolov3.detect_infrared import detect_infrared
from yolov3.detect_visible import detect_visible
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
from generate_squre import find_max_iou
from generate_squre_change import generate_square_patch
 #改图片就改这里，下面要改方形补丁的获取以及变形补丁的保存路径
tmp_dir_vis = '../workspace/cross_modal_patch_attack/yolov3_image/yinshenyi/tmp_dir_visible'
mask_dir = '../workspace/cross_modal_patch_attack/yolov3_image/yinshenyi/mask'
final_dir = '../workspace/cross_modal_patch_attack/yolov3_image/yinshenyi/final'
# 创建目录（如果不存在）
os.makedirs(tmp_dir_vis, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

trans = transforms.Compose([
    transforms.ToTensor(),
])
content_inf = 0
content_vis = 1
classes = (
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'whistle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush', 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'whistle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tvmonitor', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)


def ifcross(p1, p2, p, px, py):
    d1 = (p1[0] - px) * (p[1] - py) - (p1[1] - py) * (p[0] - px)
    d2 = (p2[0] - px) * (p[1] - py) - (p2[1] - py) * (p[0] - px)
    if d1 * d2 < 0:
        return False
    else:
        return True


def compute_dis(p, px, py):
    dis = pow(pow(p[0] - px, 2) + pow(p[1] - py, 2), 0.5)
    return dis


# 为了找到贴了补丁的目标在正常图片中的位置求iou
def patch_final(mask, patch_img):  # 输入掩码，最后输出掩码的最小矩形框的补丁
    mask = mask.astype(np.uint8)
    # 去掉 batch 维度，将补丁形状从 (1, c, h, w) 转换为 (c, h, w)
    patch_img = (patch_img * 255).astype(np.uint8)  # 反归一化并转换为 uint8
    patch_img = patch_img[0]  # 现在形状为 (c, h, w)

    # 2. 将补丁从 (c, h, w) 转换为 (h, w, c)，以便 OpenCV 处理
    patch_img = np.transpose(patch_img, (1, 2, 0))  # 形状变为 (h, w, c)

    # 3. 获取掩码的轮廓（找到掩码的边界）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]  # 取最大的轮廓

    # 4. 获取掩码轮廓的最小外接矩形
    rect = cv2.minAreaRect(contour)  # 获取旋转矩形
    box = cv2.boxPoints(rect)  # 获取矩形的四个顶点
    box = np.int0(box)  # 转换为整数坐标

    # 5. 计算透视变换矩阵
    # 定义补丁的四个顶点（对应补丁的四个角）
    h, w = patch_img.shape[:2]
    patch_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])  # 补丁的四个角

    # 定义目标顶点（掩码的最小外接矩形的四个顶点）
    mask_corners = np.float32(box)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(patch_corners, mask_corners)

    # 6. 对补丁进行透视变换
    warped_patch = cv2.warpPerspective(patch_img, M, (mask.shape[1], mask.shape[0]))
    warped_patch_new = warped_patch.transpose(2, 0, 1)  # 交换维度

    #可视化图像
    # plt.imshow(warped_patch)
    # plt.show(block=True)
    return warped_patch_new


# 计算把补丁加入进来
def patch(image_data, H, W, circle, s, region):
    # 向上取整
    s1 = math.ceil(region[1] - region[0])
    s2 = math.ceil(region[3] - region[2])
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
    # plt.imshow(resized_image_data)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show(block=True)
    # 恢复为 (C, H, W) 形状
    resized_image_data = np.transpose(resized_image_data, (2, 0, 1))

    # 查看调整后的图像形状
    # print("调整后的图像数据形状:", resized_image_data.shape)

    # Step 7: 缩放图像到 最大限制范围那么大
    scaled_image = Image.fromarray(np.transpose(resized_image_data, (1, 2, 0)).astype(np.uint8))  # 转换回 (H, W, C)
    scaled_image = scaled_image.resize((s2, s1))  # 缩放到最大限制范围

    # 转换为 numpy 数组
    scaled_image_data = np.array(scaled_image)

    # 恢复为 (C, H, W) 形状
    scaled_image_data = np.transpose(scaled_image_data, (2, 0, 1))

    # Step 8: 创建一个全零图像
    output_image = np.zeros_like(resized_image_data)

    # Step 9: 计算目标位置 (x=506, y=114) 的中心
    center_x, center_y = circle[1], circle[0]
    w, h = s2, s1

    # 计算图像放置的位置，确保不超出图像的边界
    x_start = int(max(center_x - w // 2, 0))
    y_start = int(max(center_y - h // 2, 0))
    x_end = int(min(center_x + w // 2, resized_image_data.shape[2]))
    y_end = int(min(center_y + h // 2, resized_image_data.shape[1]))

    # 将缩放后的图像放入全零图像的指定位置
    output_image[:, y_start:y_end, x_start:x_end] = scaled_image_data[:, 0:(y_end - y_start), 0:(x_end - x_start)]

    # Step 10: 可视化结果
    output_image_pil = Image.fromarray(np.transpose(output_image, (1, 2, 0)).astype(np.uint8))
    # plt.imshow(output_image_pil)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show(block=True)
    return output_image


def out_transform(out):
    bsize, _, h, w = out.size()  # 获取批次大小，高和宽
    out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * 5, 5 + 80)
    xy_pred = torch.sigmoid(out[:, :, 0:2])  # 提取出xy中心坐标
    conf_pred = torch.sigmoid(out[:, :, 4:5])  # 提取出置信度
    hw_pred = torch.exp(out[:, :, 2:4])  # 提取出宽和高
    class_score = out[:, :, 5:]  # 提取出分类得分
    class_pred = F.softmax(class_score, dim=-1)  # dim是指在class_score哪个维度上进行softmax
    delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)  # 连接xy中心点和宽高两个数据
    return delta_pred, conf_pred, class_pred


# 扰动掩码由spline_multi_mask函数生成
# GrieFunc是核心函数，计算扰动对红外和可见光模型的攻击效果，包括对抗成功率 r_attack 和距离成功的差距 dis_to_success
def GrieFunc(vardim, x, visible_ori, threat_visible_model, prob_ori_visible, img_name, step_number, h, w, circle, s,
             box, ori_squre_confidence, region):
    state = []
    p1 = []
    length = int(len(x))
    # 两个区域的坐标分别加入到p1和p2中去，这里就只需要一个p1，把坐标添加进去即可，把p2删除了
    for i in range(length):
        if i % 2 == 0:
            p1.append([x[i], x[i + 1]])
    state.append(p1)  # 添加一个p1即可
    # mask = np.ones((h, w), dtype=np.int8)  # 检查变形后是否有效果
    # for m, n in p1:
    #     mask[int(m)][int(n)] = 0
    mask = spline_multi_mask(state, h, w)  # 这里的h,w是原始图像经过裁剪前的h和w，这一步是将P1点通过论文方法连接起来生成掩码
    len_x = len(mask)
    len_y = len(mask[0])

    # obtain the mask
    mask = trans(mask)  # transfrom转为张量
    mask = mask[0].cpu().detach().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask_path = mask_dir + '/{}.png'.format(step_number)
    # 检查并创建目标文件夹（如果不存在）
    # os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    mask.save(mask_path)  # 这里保存的是掩码的边界框，内部还未填充，一个纯白的内部，轮廓是黑的
    mask = cv2.imread(mask_path)  #
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)  # 二值化图像，其中thresh是阈值，意味着小于该阈值的像素值会被设置为maxval 255
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # 返回掩码的轮廓，coutours轮廓点的列表，hierarchy轮廓之间的层次信息
    cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)  # 轮廓绘制在掩码图像上， thickness=-1表示填充轮廓内部，(0, 0, 0)黑色
    cv2.imwrite(mask_dir + '/{}.png'.format(step_number), mask)  # 这里保存了一个纯黑的补丁
    # 下面备注了映射到可见光和红外图像上，删除红外图像的即可
    # cast mask upon infrared images 掩码应用到红外图像上，红外攻击是让红外图像对应掩码的部分的张量置为0，看起来就是全黑
    fig = mask_dir + '/{}.png'.format(step_number)
    mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask) / 255
    mask = mask.astype(np.int8)
    mask = mask ^ (mask & 1 == mask)

    # 上面是对生成的补丁轮廓进行保存，下面将掩码和补丁进行融合处理补丁
    # 加载隐身衣代码(这里加载的图片是归一化的，所以后面加载其他图片也要归一化)
    image_data = np.load('../patch1000.npy')
    # 加载yolo补丁代码
    # image = cv2.imread('object_score.png')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#这里加不加看情况
    # image = image/255.0
    # image=image.astype(np.float32)
    # image_array = np.transpose(image, (2, 0, 1))
    # image_data = np.expand_dims(image_array, axis=0)
    # 第一个版本生成补丁的方法
    # Patch =patch(image_data,h,w,circle,s,region) #这里是把训练的补丁定义出来
    # if Patch.max() > 1:
    #     # 归一化操作：除以 255 将值映射到 [0, 1] 范围
    #     Patch = Patch / 255.0
    Patch = patch_final(mask, image_data)
    if Patch.max() > 1:
        # 归一化操作：除以 255 将值映射到 [0, 1] 范围
        Patch = Patch / 255.0
    x_adv = visible_ori * (1 - mask) + mask * Patch  # 哈德曼乘积用*，矩阵乘法用*，补丁利用掩码加到图像上面去
    adv_final = x_adv[0].cpu().detach().numpy()  # 只是去掉批次维度
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    adv_sample = Image.fromarray(adv_x_255)
    save_path = tmp_dir_vis + '/{}.png'.format(step_number)
    adv_sample.save(save_path)

    # r_attack，下面是重要的，用条件去筛选图案形状的过程
    with open(tmp_dir_vis + '/{}.png'.format(step_number), 'rb') as fig:
        sample = Image.open(fig)
        visible_input = trans(sample)
        visible_ori = torch.stack([visible_input])  # N C H W
        visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear',
                                    align_corners=False)  # 采用双线性插值将不同大小图片上/下采样到统一大小
        nms_thres = 0.4
        conf_thres = 0.001
        with torch.no_grad():
            detections = threat_visible_model(visible_det)
            detections = non_max_suppression(detections, h, w, conf_thres, nms_thres)
        detections = detections[0]
        detections = detections[detections[:, 5] == 0]
        boxes = detections[:, :4].tolist()
        confidences = detections[:, 4:5]
        # 输入数据，预测结果，H，W原始图片的高和宽，置信度0.6，nms阈值0.4
        # 输出数据：将这四个值弄成一个一行的tensor变量，坐标，置信度，类别概率，类别索引,这里评估的时候还过滤了其他类别，只保留person类别
        # 为了可视化图片
        # det_boxes = reslut[:, :5].cpu().numpy()
        # det_classes = reslut[:, -1].long().cpu().numpy()
        # im2show = draw_detection_boxes(sample, det_boxes, det_classes, class_names=classes)
        # plt.figure()
        # plt.imshow(im2show)
        # plt.show()
        # 提取出bbox列表和置信度列表
        max_iou, max_index = find_max_iou(box, boxes)
        # print(f"最大 IoU 值: {max_iou}")
        # print(f"最大 IoU 对应的索引: {max_index}")
        max_confidences = confidences[max_index]  # 这里优化什么的不如方形补丁的话最后就贴方形补丁
    if (max_confidences.item() < ori_squre_confidence.item()):
        return max_confidences
    else:
        return ori_squre_confidence  # 返回以后顺便把self.ori_squre_confidence的值改了，改为当前最优的

def ori_squre(img_path, H, W, px, py, r1, file_name):
    square_patch = generate_square_patch(img_path, H, W, px, py, r1, file_name)
    return square_patch
class DEIndividual:
    '''
    individual of differential evolution algorithm
    '''

    def __init__(self, vardim, points, region):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.points = points
        self.fitness = 0
        self.region = region

    # 根据初始值建立初始种群
    def generate(self, h, w, rg):
        '''
        generate a random chromsome for differential evolution algorithm
        '''
        len = self.vardim
        # chrom首先设置为全零的数组，然后在下面根据points初始值的位置，按照正负3的差距生成初始种群，个人理解也就是初始的圆按照一定程度进行扭曲得到新的补丁形状
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.points[i] + np.random.randint(-rg, rg)
            if i % 2 == 0 and self.chrom[i] >= h:
                self.chrom[i] = h - 1
            if i % 2 != 0 and self.chrom[i] >= w:
                self.chrom[i] = w - 1

        # print(self.chrom)

    def calculateFitness(self, visible_ori, threat_visible_model, prob_ori_visible, img_name, step_number, h, w, circle,
                         s, box, ori_squre_confidence, region):
        '''
        calculate the fitness of the chromsome
        '''
        '''
        计算得到适应度，距离，个体与红外图像中的相关距离，个体与可见光图像中的相关距离
        '''
        # vardim指的是锚点xy坐标的数量24个，chrom指的是初始化锚点经过随机变化得到的锚点坐标
        self.fitness = GrieFunc(
            self.vardim, self.chrom, visible_ori, threat_visible_model, prob_ori_visible, img_name, step_number, h, w,
            circle, s, box, ori_squre_confidence, region)


class DifferentialEvolutionAlgorithm:
    '''
    The class for differential evolution algorithm
    '''

    def __init__(self, sizepop, vardim, points, eq_points, circle, region, visible_ori, darknet_model, \
                 prob_ori_visible, img_name, MAXGEN, params, h, w, s, box, ori_squre_confidence, rg, r2,r1):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        param: algorithm required parameters, it is a list which is consisting of [crossover rate CR, scaling factor F]
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.points = points
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params
        self.circle = circle
        self.eq_points = eq_points
        self.region = region
        self.visible_ori = visible_ori
        self.threat_visible_model = darknet_model
        self.prob_ori_visible = prob_ori_visible
        self.img_name = img_name
        self.step_number = 0
        self.h = h
        self.w = w
        self.s = s
        self.box = box
        self.ori_squre_confidence = ori_squre_confidence
        self.rg = rg  # 锚点变形的范围
        self.r2 = r2  # 内圆变形的范围
        self.r1 = r1

    def initialize(self):
        '''
        initialize the population 初始种群，种群数量由sizepop决定
        '''
        for i in range(0, self.sizepop):
            ind = DEIndividual(self.vardim, self.points, self.region)
            ind.generate(self.h, self.w, self.rg)
            self.population.append(ind)

    def evaluate(self, x, ori_squre_confidence):
        '''
        evaluation of the population 目标置信度
        '''
        x.calculateFitness(self.visible_ori, self.threat_visible_model,
                           self.prob_ori_visible, self.img_name, self.step_number, self.h, self.w, self.circle, self.s,
                           self.box, ori_squre_confidence, x.region)
    def solve(self):
        '''
        evolution process of differential evolution algorithm
        '''
        '''
          建立初始种群，通过self.initialize()生成不一样形状的初始坐标
        '''
        self.step_number = 0
        self.t = 0  # 第几代种群
        self.initialize()
        # p1=[] #检查初始形状是否构建完善
        # length=int(self.vardim)
        # for i in range(length):
        #     if i % 2 == 0:
        #         p1.append([self.points[i], self.points[i + 1]])
        # mask = np.ones((self.h, self.w), dtype=np.int8)  # 初始化mask和原始图片的大小一致
        # for m, n in p1:
        #     mask[int(m)][int(n)] = 0
        '''
            下面的for循环是在计算生成的种群的适应度，个人理解是找到最优的形状，然后进行补丁形状的确定。
        '''
        for i in range(0, self.sizepop):
            self.evaluate(self.population[i], self.ori_squre_confidence)
            self.step_number += 1  # 这里的fitness就是论文里面的J(S)
            self.fitness[i] = self.population[i].fitness.item()  # fitness指的是置信度，保存每个初始化里面的置信度
        # 这里所谓的bestIndex指的是是的置信度最小的形状的索引，越小攻击效果越好
        best = np.min(self.fitness)
        bestIndex = np.argmin(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])  # 把最优形状的信息深度拷贝过去，包括锚点，置信度等信息
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = self.best.fitness
        self.trace[self.t, 1] = self.avefitness
        print("Generation %d: optimal Uobj value is: %f; average Uobj  value is %f;ori_square Uobj value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1], self.ori_squre_confidence))
        x = self.best.chrom
        state = []
        p1 = []
        length = int(len(x))  # 前一半是第一个补丁，后一半是第二个补丁
        for i in range(length):
            if i % 2 == 0:
                p1.append([x[i], x[i + 1]])
        state.append(p1)
        mask = spline_multi_mask(state, self.h, self.w)
        # obtain the mask
        mask = trans(mask)
        mask = mask[0].cpu().detach().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask)
        optimical_dir = '../workspace/result'
        # 这里的img_name需要处理一下路径，只需要最后的图片名称，调试一下就知道了
        file_name = os.path.basename(self.img_name)
        mask_path = optimical_dir + '/{}_{}.png'.format(file_name, self.t)
        mask.save(mask_path)  # 保存的是初始种群中最优适应度下的掩码
        mask = cv2.imread(mask_path)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)  # 填充
        k = 0  # 记录变异过程中最优解有几次没有更新，减少迭代次数
        # 下面是进行变异交叉验证操作
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            for i in range(0, self.sizepop):  # 种群的每一个元素进行变异交叉验证后再次计算fitness与变异前对比，如果大于则替换对应的元素，否则保留原来的
                vi = self.mutationOperation(i)  # 这里是论文里面Mutation的部分，也就是变异
                ui = self.crossoverOperation(i, vi,
                                             self.r2)  # 这里是论文里面Crossover的部分，其实就是判断是否符合论文在界内的要求，不符合就去掉变异的坐标，保留其变异前的坐标
                xi_next = self.selectionOperation(i, ui)  # 这里的ui就是确定好的经过变异和交叉验证得到的新的补丁掩码
                self.population[i] = xi_next
            for i in range(0, self.sizepop):
                self.fitness[i] = self.population[i].fitness
            best = np.min(self.fitness)
            bestIndex = np.argmin(self.fitness)
            # 原来这里是大于，但是我们是逆向的因此是越小越好
            if best < self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
                k = 0  # 如果有更新的话就清零从头记数
            else:  # 记录k的不变次数
                k = k + 1
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            print(
                "Generation %d: optimal function value is: %f; average function value is %f;ori_square Uobj value is %f" % (
                    self.t, self.trace[self.t, 0], self.trace[self.t, 1], self.ori_squre_confidence))
            # 检查是否已经达到了最大迭代数,或者连续5次最低置信度不变认为达到要求了
            if self.t == self.MAXGEN - 1 or k == 5 or self.best.fitness < 0.6:
                #如果置信度和方形置信度一致，那么采用方形补丁
                if (self.best.fitness == self.ori_squre_confidence):
                    ori_squre(self.img_name, self.h, self.w, self.circle[0], self.circle[1], self.r1, file_name)
                #优化的好就用优化后的形状
                else:
                    x = self.best.chrom  # 获取最优适应度下的锚点坐标
                    state = []
                    p1 = []
                    length = int(len(x))
                    # 两个区域的坐标分别加入到p1和p2中去，这里就只需要一个p1，把坐标添加进去即可，把p2删除了
                    for i in range(length):
                        if i % 2 == 0:
                            p1.append([x[i], x[i + 1]])
                    state.append(p1)  # 添加一个p1即可
                    mask = spline_multi_mask(state, self.h, self.w)  # 掩码获取
                    # obtain the mask
                    mask = trans(mask)
                    mask = mask[0].cpu().detach().numpy()
                    mask = (mask * 255).astype(np.uint8)
                    mask = Image.fromarray(mask)
                    final_name = os.path.basename(self.img_name)
                    mask_path = mask_dir + '/{}.png'.format(final_name)  # 与之前不同这里{}里面填写的是图片的名称
                    mask.save(mask_path)
                    mask = cv2.imread(mask_path)
                    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)  # 二值化处理
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)
                    cv2.imwrite(mask_dir + '/{}.png'.format(final_name), mask)

                    # cast mask upon infrared images
                    fig = mask_dir + '/{}.png'.format(final_name)
                    mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
                    mask = np.array(mask) / 255
                    mask = mask.astype(np.int8)
                    mask = mask ^ (mask & 1 == mask)  # 对mask进行按位与操作

                    # 上面是对生成的补丁轮廓进行保存，下面将掩码和补丁进行融合处理补丁
                    # 加载隐身衣服补丁
                    image_data = np.load('../patch1000.npy')
                    # 加载yolo补丁代码
                    # image = cv2.imread('object_score.png')
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # image = image / 255.0
                    # image = image.astype(np.float32)
                    # image_array = np.transpose(image, (2, 0, 1))
                    # image_data = np.expand_dims(image_array, axis=0)
                    Patch = patch_final(mask, image_data)
                    if Patch.max() > 1:
                        # 归一化操作：除以 255 将值映射到 [0, 1] 范围
                        Patch = Patch / 255.0
                    x_adv = self.visible_ori * (1 - mask) + mask * Patch  # 哈德曼乘积用*，矩阵乘法用*，补丁利用掩码加到图像上面去
                    adv_final = x_adv[0].cpu().detach().numpy()  # 只是去掉批次维度
                    adv_final = (adv_final * 255).astype(np.uint8)
                    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
                    save_path = final_dir + '/{}'.format(final_name)
                    plt.imsave(save_path, adv_x_255)
                    #.save保存方式
                    # adv_x_255 = np.transpose(adv_final, (1, 2, 0))
                    # adv_sample = Image.fromarray(adv_x_255)
                    # adv_sample.save(save_path, quality=99)
                break
                    # 可以不对其进行检测
                    # with open(final_dir + '/{}'.format(final_name), 'rb') as fig:
                    #     sample = Image.open(fig)
                    #     visible_input = trans(sample)
                    #     visible_ori = torch.stack([visible_input])  # N C H W
                    #     visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear',
                    #                                 align_corners=False)  # 采用双线性插值将不同大小图片上/下采样到统一大小
                    #     out = self.threat_visible_model(visible_det)
                    #     out = out_transform(out)  # 是为了转换成yolov2论文中的格式，去过滤筛选出最正确的预测框，这一步应该是参照哪个里面的代码，不是隐身衣就是攻击yolov2的那个
                    #     out = [item[0].data for item in out]  # 用3维度的list分别存储预测框的值，置信度，和类别得分
                    #     # 输入数据，预测结果，H，W原始图片的高和宽，置信度0.6，nms阈值0.4
                    #     # 输出数据：将这四个值弄成一个一行的tensor变量，坐标，置信度，类别概率，类别索引,这里评估的时候还过滤了其他类别，只保留person类别
                    #     reslut = yolo_eval(out, self.h, self.w, conf_threshold=0.1, nms_threshold=0.2)
                    #     # 为了可视化图片
                    #     # det_boxes = reslut[:, :5].cpu().numpy()
                    #     # det_classes = reslut[:, -1].long().cpu().numpy()
                    #     # im2show = draw_detection_boxes(sample, det_boxes, det_classes, class_names=classes)
                    #     # plt.figure()
                    #     # plt.imshow(im2show)
                    #     # plt.show()
                    #     # 提取出bbox列表和置信度列表
                    #     boxes = reslut[:, :4].tolist()  # 提取每行前四个值，并转换为 Python 列表
                    #     confidences = reslut[:, 4:5]  # 提取每行的第 5 个值，保持为张量，形状为 (4, 1)
                    #     max_iou, max_index = find_max_iou(self.box, boxes)
                    #     # print(f"最大 IoU 值: {max_iou}")
                    #     # print(f"最大 IoU 对应的索引: {max_index}")
                    #     max_confidences = confidences[max_index]

        print("Optimal function value is: %f; " % \
              self.trace[self.t, 0])  # self.trace保存的是最优适应度和平均适应度，也就是平衡红外光和可见光下最佳的
        print('Optimal solution is:')
        print(self.best.chrom)
        # self.printResult()

    def selectionOperation(self, i, ui):
        '''
        selection operation for differential evolution algorithm
        '''
        xi_next = copy.deepcopy(self.population[i])  # 深度拷贝，在进行修改的时候不影响原个体
        xi_next.chrom = ui
        self.evaluate(xi_next, self.ori_squre_confidence)
        self.step_number += 1
        if xi_next.fitness < self.population[i].fitness:
            # print("change")
            return xi_next
        else:
            # print("no change")
            return self.population[i]

    def crossoverOperation(self, i, vi, r2):  # 实现交叉操作
        '''
        crossover operation for differential evolution algorithm
        '''
        # 下面代码是针对两个补丁来的，需要改成针对一个补丁
        px_1, py_1 = self.circle[0], self.circle[1]  # 掩码的圆心坐标
        y_low, y_high, x_left, x_right = self.region[0], self.region[1], self.region[2], self.region[3]  # 补丁变形范围限制
        k = np.random.randint(0, self.vardim - 1)
        ui = np.zeros(self.vardim)
        for j in range(0, int(self.vardim)):  # 第一个补丁
            if j % 2 == 0:
                dis = compute_dis([vi[j], vi[j + 1]], px_1, py_1)  # 点离圆心的距离
                pick = random.random()
                # ifcross中五个参数，第一个和第二个是通过线段等分的圆的边界点，而vi[j], vi[j+1],是变异的点，这样计算是通过论文中向量的乘积大于0还是小于0的方式来判断是否会存在交叉的现象，
                if (pick < self.params[0] or j == k) and (
                        ifcross(self.eq_points[j // 2], self.eq_points[j // 2 + 1], [vi[j], vi[j + 1]], px_1,
                                py_1) == False \
                        and dis > r2 and vi[j] > y_low and vi[j] < y_high and vi[j + 1] > x_left and vi[
                            j + 1] < x_right):
                    # dis>8是不在内圆，后面是为了限制在区域内
                    ui[j] = vi[j]
                    ui[j + 1] = vi[j + 1]
                else:  # 如果不满足条件就保留原来的个体
                    ui[j] = self.population[i].chrom[j]
                    ui[j + 1] = self.population[i].chrom[j + 1]
        return ui

    def mutationOperation(self, i):
        '''
        mutation operation for differential evolution algorithm
        '''
        a = np.random.randint(0, self.sizepop - 1)
        while a == i:
            a = np.random.randint(0, self.sizepop - 1)
        b = np.random.randint(0, self.sizepop - 1)
        while b == i or b == a:
            b = np.random.randint(0, self.sizepop - 1)
        c = np.random.randint(0, self.sizepop - 1)
        while c == i or c == b or c == a:
            c = np.random.randint(0, self.sizepop - 1)  # 上述 a，b，c就是找了三个自身各不相同也不同于i的另外三个形状的点
        vi = self.population[c].chrom + self.params[1] * \
             (self.population[a].chrom - self.population[b].chrom)  # 这里是差分里面固定的变法，
        # vi的计算方式是：c个体的染色体加上F乘以（a个体的染色体减去b个体的染色体）。这里的F通常是一个缩放因子，用于控制变异的强度。这个计算步骤生成了一个新的变异向量vi
        # 在这里self.params就是缩放因子
        return vi

    def printResult(self):
        '''
        plot the yinshenyi-result of the differential evolution algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Differential Evolution Algorithm for function optimization")
        plt.legend()
        plt.show()
