import os
import shutil
import numpy as np
import cv2
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from torchvision import transforms
from yolov3.detect_infrared import load_infrared_model, detect_infrared
from yolov3.detect_visible import load_visible_model, detect_visible
from itertools import chain
from adv_yolo.darknet import Darknet
from adv_yolo.load_data import *
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib
from itertools import chain
from adv_yolo.load_data import *
from yolo_eval import yolo_eval
from generate_squre_ceshi import generate_square_patch, find_max_iou
from pytorchyolov3.models import load_model
from pytorchyolov3.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from PIL import Image
import torch.nn.functional as F
matplotlib.use('TkAgg')
content = 1
trans = transforms.Compose([
    transforms.ToTensor(),
])
# 新方法，导入yolov2的模型，原始是yolov3
model_path = './adv_yolo/yolov3.cfg'
weights_path = './weights/yolov3.weights'
yolov3=load_model(model_path, weights_path)
yolov3=yolov3.eval()
# 利用掩码把补丁弄上去的方法
patch_applier = PatchApplier().cuda()
# 数据增强
patch_transformer = PatchTransformer().cuda()
# 这里还缺最大置信度得分和分类概率等损失，由于需要self.config后面想一下如何改
square_dir = 'workspace/cross_modal_patch_attack/yolov3_image/empatch-y2/square'  #改，如果没有自动创建
# 创建目录（如果不存在）
os.makedirs(square_dir, exist_ok=True)

def limit_region(bboxes):
    x_lefts = []
    x_rights = []
    y_heads = []
    y_legs = []

    for bbox in bboxes:
        x_left = bbox[0] + (bbox[2] - bbox[0]) / 4
        x_right = bbox[2] - (bbox[2] - bbox[0]) / 4
        y_low = bbox[1]
        y_high = bbox[3]
        y_head = y_low + (y_high - y_low) / 4
        # y_leg = y_low + (y_high - y_low) / 2  #原来的代码
        y_leg = y_high - (y_high - y_low) / 4  # 我把区域限制改大了点儿

        x_lefts.append(x_left)
        x_rights.append(x_right)
        y_heads.append(y_head)
        y_legs.append(y_leg)

    return x_lefts, x_rights, y_heads, y_legs


def get_state(img_path, bbox, H, W, s):
    # """
    # 根据多个预测框生成一个圆形区域（补丁）。
    #
    # 参数：
    # img_path: 图像路径（未使用，但保留参数接口）
    # bboxes: 二维列表，每个元素为 [x1, y1, x2, y2]，代表预测框左上角和右下角坐标。
    #
    # 返回：
    # px_list: 每个预测框中心的 x 坐标列表。
    # py_list: 每个预测框中心的 y 坐标列表。
    # eq_points: 每个预测框对应的圆上的等距点坐标列表。
    # state: 每个预测框对应的圆上的点到下一个点的中点坐标列表。
    # """
    a = 12  # 圆周上的点数
    e = s  # 半径

    # 获取预测框中心
    x1, y1, x2, y2 = bbox
    px = (y1 + y2) / 2 - 0.15 * (bbox[3]-bbox[1])   # 用纵坐标表示横坐标，以匹配原始图片的坐标系
    py = (x1 + x2) / 2

    # 生成圆周上的等距点
    eq_points = []
    for n in range(1, a + 1):
        xx = px + round(e * np.cos(2 * np.pi * (n - 1) / a), 2)
        yy = py + round(e * np.sin(2 * np.pi * (n - 1) / a), 2)
        eq_points.append([xx, yy])
    eq_points.append([px + round(e * np.cos(0), 2), py + round(e * np.sin(0), 2)])
    # 主要是某些边界上的目标可能在添加的时候会超出边界导致越界问题
    eq_points = [[min(x, H), min(y, W)] for x, y in eq_points]
    # 生成中点
    state = []
    for i in range(len(eq_points) - 1):
        pre_x = eq_points[i][0]
        pre_y = eq_points[i][1]
        x = eq_points[i + 1][0]
        y = eq_points[i + 1][1]
        state.append([int(round((pre_x + x) / 2, 2)), int(round((pre_y + y) / 2, 2))])
    return px, py, eq_points, state


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


def ori_squre(img_path, H, W, px, py, r1,file_name):
    square_patch = generate_square_patch(img_path, H, W, px, py, r1,file_name)
    return square_patch


if __name__ == "__main__":
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
    # 读取行人数据集
    dataset_dir = '/home/a/LJJ/Cross-modal_Patch_Attack-main/inria'  # 替换为你的数据集路径
    train_dir = os.path.join(dataset_dir, 'Test')
    pos_lst_path = os.path.join(train_dir, 'pos.lst')
    # 读取 pos.lst 文件中的图片路径
    with open(pos_lst_path, 'r') as f:
        pos_image_paths = f.readlines()
    # 去除每一行末尾的换行符，并生成完整路径
    pos_image_paths = [os.path.join(dataset_dir, path.strip()) for path in pos_image_paths]
    # 循环读取每一张图片
    for img_path in pos_image_paths:
        #     continue
        visible_sample = Image.open(img_path)
        # 显示图片
        # visible_sample.show()
        visible_input = trans(visible_sample)  # to tensor
        # 增加维度，其实每次送入的只有一个图片
        visible_ori = torch.stack([visible_input])  # N C H W
        visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear',
                                    align_corners=False)  # 采用双线性插值将不同大小图片上/下采样到统一大小
        # 原始图像大小前，图像的宽和高（因为yolo要求模型的输入是416*416大小的）
        H, W = visible_sample.size[1], visible_sample.size[0]
        nms_thres = 0.4
        conf_thres = 0.6
        with torch.no_grad():
            detections = yolov3(visible_det)
            detections = non_max_suppression(detections, H, W, conf_thres, nms_thres)
        detections = detections[0]
        detections = detections[detections[:, 5] == 0]
        boxes = detections[:, :4].tolist()
        confidences = detections[:, 4:5]
        # 可视化
        # det_boxes = reslut[:, :5].cpu().numpy()
        # det_classes = reslut[:, -1].long().cpu().numpy()
        # im2show = draw_detection_boxes(visible_sample, det_boxes, det_classes, class_names=classes)
        # plt.figure()
        # plt.imshow(im2show)
        # plt.show()
        # 提取出bbox列表和置信度列表
        file_name = os.path.basename(img_path)
        if len(detections) == 0:
            final_zero = os.path.join(square_dir, file_name)
            visible_sample.save(final_zero)
            continue
        # 改 这是限制补丁在检测框里面的区域，也就是限制补丁的大小，区域坐标，这里需要改的是 x_left等坐标应该是一个列表了
        x_left, x_right, y_head, y_leg = limit_region(boxes)
        # 改 生成的是两个补丁区域，和论文中一致，是生成的两个补丁，eq_points是在圆的边缘均分的几个点，而points是eq_points中两点之间的中点的集合
        # infrared_img是原始图片的路径
        L = confidences.size(0)
        # 这是后面循环给多个人添加补丁的路径
        square_path = os.path.join(square_dir, file_name)
        for i in range(L):
            # 由于我要循环的给图片添加补丁，因此除了第一个图片以外，第二次迭代循环我需要加载保存的已经添加部分补丁的图片
            # 因此这里的visible_ori需要重新加载为新的图片，然后看看img_path需要变不
            print("第{}轮".format(i))
            if i == 0:
                visible_img = visible_ori
            else:
                img_path = square_path
                visible_s = Image.open(img_path)
                visible_i = trans(visible_s)  # to tensor
                visible_img = torch.stack([visible_i])  # N C H W
            # if i!=7:
            #     continue
            # else:
            #     visible_img = visible_ori
            # 这里的box是当前目标的框，补丁大小为目标大小的20%
            box = boxes[i]
            target_size = math.sqrt((((box[2] - box[0]) * 0.2) ** 2) + (((box[3] - box[1]) * 0.2) ** 2))
            r1 = target_size
            s = r1 * r1
            r = math.sqrt(s / math.pi)
            px, py, eq_points, state = get_state(img_path, box, H, W, r)
            prob_ori_visible_index = confidences[i]
            points_index = list(chain.from_iterable(state))
            # 计算初始方形补丁的大小,并且将补丁贴到指定位置,得到最初的方形补丁的置信度
            # 返回初始方形补丁的置信度的值，与后面补丁变形的方法进行比较
            r1=int(r1)
            ori_squre(img_path, H, W, px, py, r1,file_name)



