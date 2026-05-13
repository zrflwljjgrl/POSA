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

# 使用 DPR 版本的差分进化实现
from duibi.yolov2_dpr.DE_inria_v2_dpr import DifferentialEvolutionAlgorithm

from adv_yolo.darknet import Darknet
from adv_yolo.load_data import *
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
from generate_squre import generate_square_patch, find_max_iou

matplotlib.use('TkAgg')
content = 1
trans = transforms.Compose([
    transforms.ToTensor(),
])

darknet_model = Darknet("../../adv_yolo/yolo.cfg")
darknet_model.load_weights("../../weights/yolo.weights")
darknet_model = darknet_model.eval()

# darknet_model = Darknet("./adv_yolo/yolo.cfg")
# darknet_model.load_weights("./weights/yolo.weights")
# darknet_model = darknet_model.eval()
patch_applier = PatchApplier().cuda()
patch_transformer = PatchTransformer().cuda()


def limit_region(px, py, r, k, bbox):
    if k > 2 * r:
        z = (k - 2 * r)
        if z > 10:
            z = z / 4
        x_L = px - k / 2
        x_R = px + k / 2
        y_H = py - k / 2
        y_L = py + k / 2
    else:
        z = (2 * r - k)
        if z > 10:
            z = z / 4
        x_L = px - k / 2 - z
        x_R = px + k / 2 + z
        y_H = py - k / 2 - z
        y_L = py + k / 2 + z
    return x_L, x_R, y_H, y_L, z


def get_state(img_path, bbox, H, W, s):
    """
    与原 yolov2 实现一致：根据一个预测框生成圆形补丁的控制点（eq_points、state）。
    """
    a = 12
    e = s
    x1, y1, x2, y2 = bbox
    px = (y1 + y2) / 2 - 0.15 * (bbox[3] - bbox[1])
    py = (x1 + x2) / 2
    eq_points = []
    for n in range(1, a + 1):
        xx = px + round(e * np.cos(2 * np.pi * (n - 1) / a), 2)
        yy = py + round(e * np.sin(2 * np.pi * (n - 1) / a), 2)
        eq_points.append([xx, yy])
    eq_points.append([px + round(e * np.cos(0), 2), py + round(e * np.sin(0), 2)])
    eq_points = [[min(x, H), min(y, W)] for x, y in eq_points]
    state = []
    for i in range(len(eq_points) - 1):
        pre_x = eq_points[i][0]
        pre_y = eq_points[i][1]
        x = eq_points[i + 1][0]
        y = eq_points[i + 1][1]
        state.append([int(round((pre_x + x) / 2, 2)), int(round((pre_y + y) / 2, 2))])
    return px, py, eq_points, state


def out_transform(out):
    bsize, _, h, w = out.size()
    out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * 5, 5 + 80)
    xy_pred = torch.sigmoid(out[:, :, 0:2])
    conf_pred = torch.sigmoid(out[:, :, 4:5])
    hw_pred = torch.exp(out[:, :, 2:4])
    class_score = out[:, :, 5:]
    class_pred = F.softmax(class_score, dim=-1)
    delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)
    return delta_pred, conf_pred, class_pred


def ori_squre(img_path, H, W, px, py, r1, box):
    square_patch = generate_square_patch(img_path, H, W, px, py, r1)
    square_patch = np.array(square_patch, dtype=np.float32)
    square_patch_show = square_patch
    square_patch = trans(square_patch)
    square_patch_ori = torch.stack([square_patch])
    square_patch_det = F.interpolate(square_patch_ori, (416, 416), mode='bilinear',
                                     align_corners=False)
    square_patch_out = darknet_model(square_patch_det)
    square_patch_out = out_transform(square_patch_out)
    square_patch_out = [item[0].data for item in square_patch_out]
    reslut = yolo_eval(square_patch_out, H, W, conf_threshold=0.001, nms_threshold=0.4)
    square_patch_out_det_boxes = reslut[:, :5].cpu().numpy()
    square_patch_out_det_classes = reslut[:, -1].long().cpu().numpy()
    square_patch_show = (square_patch_show * 255).astype(np.uint8)
    square_patch_show = Image.fromarray(square_patch_show)
    square_patch_im2show = draw_detection_boxes(square_patch_show, square_patch_out_det_boxes,
                                                square_patch_out_det_classes, class_names=classes)
    boxes = reslut[:, :4].tolist()
    confidences = reslut[:, 4:5]
    max_iou, max_index = find_max_iou(box, boxes)
    return confidences[max_index]


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
    dataset_dir = '../../inria'
    final_dir = '../workspace/cross_modal_patch_attack/yolov2_image_dpr/TCEGA/final'
    change_dir = '../workspace/cross_modal_patch_attack/yolov2_image_dpr/TCEGA/final'
    train_dir = os.path.join(dataset_dir, 'Test')
    pos_lst_path = os.path.join(train_dir, 'pos.lst')
    with open(pos_lst_path, 'r') as f:
        pos_image_paths = f.readlines()
    pos_image_paths = [os.path.join(dataset_dir, path.strip()) for path in pos_image_paths]
    u = 0
    for img_path in pos_image_paths:
        u = u + 1
        print("image_name:", img_path)
        visible_sample = Image.open(img_path)
        visible_input = trans(visible_sample)
        visible_ori = torch.stack([visible_input])
        visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear',
                                    align_corners=False)
        H, W = visible_sample.size[1], visible_sample.size[0]
        out = darknet_model(visible_det)
        out = out_transform(out)
        out = [item[0].data for item in out]
        reslut = yolo_eval(out, H, W, conf_threshold=0.6, nms_threshold=0.4)
        file_name = os.path.basename(img_path)
        if len(reslut) == 0:
            final_zero = os.path.join(final_dir, file_name)
            os.makedirs(os.path.dirname(final_zero), exist_ok=True)
            visible_sample.save(final_zero)
            continue
        boxes = reslut[:, :4].tolist()
        confidences = reslut[:, 4:5]
        L = confidences.size(0)
        final_path = os.path.join(final_dir, file_name)
        for i in range(L):
            print("第{}轮".format(i))
            if i == 0:
                visible_img = visible_ori
            else:
                img_path = final_path
                visible_s = Image.open(img_path).convert('RGB')
                visible_i = trans(visible_s)
                visible_img = torch.stack([visible_i])
            box = boxes[i]
            target_size = math.sqrt((((box[2] - box[0]) * 0.2) ** 2) + (((box[3] - box[1]) * 0.2) ** 2))
            r1 = target_size
            s = r1 * r1
            r = math.sqrt(s / math.pi)
            k = min(box[2] - box[0], box[3] - box[1])
            px, py, eq_points, state = get_state(img_path, box, H, W, r)
            prob_ori_visible_index = confidences[i]

            # ----- DPR 版本：用射线长度而不是锚点坐标 -----
            # 射线条数 K，可以调大一些以增加形状自由度（例如 32 或 48）
            K = 32
            # 以当前圆半径 r 作为初始半径，所有射线初始长度相同
            initial_radii = [r] * K

            r1 = int(r1)
            r2 = r / 2
            ori_squre_confidence = ori_squre(img_path, H, W, px, py, r1, box)
            print(ori_squre_confidence)
            x_left, x_right, y_head, y_leg, rg = limit_region(px, py, r, k, box)
            rg = int(rg)
            if rg <= 0:
                rg = 6
            # 注意：现在 vardim = K，points 为初始半径列表 initial_radii
            dea = DifferentialEvolutionAlgorithm(15, K, initial_radii, None, [px, py],
                                                 [y_head, y_leg, x_left, x_right],
                                                 visible_img, darknet_model,
                                                 prob_ori_visible_index, img_path, 20, [1, 0.6], H, W, r, box,
                                                 ori_squre_confidence, rg, r2, r1)
            dea.solve()


