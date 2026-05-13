import glob
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
from DE_inria_v2_wuli import DifferentialEvolutionAlgorithm
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
# 新方法，导入yolov2的模型，原始是yolov3
darknet_model = Darknet("../adv_yolo/yolo.cfg")
darknet_model.load_weights("../weights/yolo.weights")
darknet_model = darknet_model.eval()
# 利用掩码把补丁弄上去的方法
patch_applier = PatchApplier().cuda()
# 数据增强
patch_transformer = PatchTransformer().cuda()
# 这里还缺最大置信度得分和分类概率等损失，由于需要self.config后面想一下如何改
# 设置可见光和红外光下的数据集
# infrared_dir = './workspace/cross_modal_patch_attack/dataset/attack_infrared'
# visible_dir = './workspace/cross_modal_patch_attack/dataset/attack_visible'
# final_dir = './workspace/cross_modal_patch_attack/result/final'


def limit_region(px,py,r,k,bbox):
    #原来的限制方式
    # x_left = bbox[0] + (bbox[2] - bbox[0]) / 4
    # x_right = bbox[2] - (bbox[2] - bbox[0]) / 4
    # y_low = bbox[1]
    # y_high = bbox[3]
    # y_head = y_low + (y_high - y_low) / 4
    # # y_leg = y_low + (y_high - y_low) / 2  #原来的代码
    # y_leg = y_high - (y_high - y_low) / 4  # 我把区域限制改大了点儿
    #限制代码的限制方式
    if k> 2*r:
        z = (k-2*r)
        if z>10:
            z=z/4
        x_L = px - k / 2
        x_R = px + k / 2
        y_H = py - k / 2
        y_L = py + k / 2
    else:
        z = (2 * r-k)
        if z>10:
            z=z/4
        x_L = px - k / 2 -z
        x_R = px + k / 2 +z
        y_H = py - k / 2 -z
        y_L = py + k / 2 +z
    return x_L, x_R, y_H, y_L,z


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
    # a = 12  # 圆周上的点数
    # e = s  # 半径
    # px=[]
    # py=[]
    # eq_points = []
    # state = []
    # for bbox in bboxes:
    #     # 获取预测框中心
    #     x1, y1, x2, y2 = bbox
    #     px_1 = (y1 + y2) / 2 #这里用纵坐标去表示横坐标才能对应上原始图片的坐标补丁上移动一下，贴在人物的胸口
    #     py_1 = (x1 + x2) / 2
    #     px.append(px_1)
    #     py.append(py_1)
    #
    #     # 生成圆周上的等距点
    #     eq_points_1 = []
    #     points_1 = []
    #     for n in range(1, a + 1):
    #         xx = px_1 + round(e * np.cos(2 * np.pi * (n - 1) / a), 2)
    #         yy = py_1 + round(e * np.sin(2 * np.pi * (n - 1) / a), 2)
    #         eq_points_1.append([xx, yy])
    #     eq_points_1.append([px_1 + round(e * np.cos(0), 2), py_1 + round(e * np.sin(0), 2)])
    #     mask = np.ones((H, W), dtype=np.int8)  # 初始化mask和原始图片的大小一致
    #     for m, n in eq_points_1:
    #         mask[int(m)][int(n)] = 0
    #     # 生成中点
    #     for i in range(len(eq_points_1) - 1):
    #         pre_x = eq_points_1[i][0]
    #         pre_y = eq_points_1[i][1]
    #         x = eq_points_1[i + 1][0]
    #         y = eq_points_1[i + 1][1]
    #         points_1.append([int(round((pre_x + x) / 2, 2)), int(round((pre_y + y) / 2, 2))])
    #     mask = np.ones((H, W), dtype=np.int8)  # 初始化mask和原始图片的大小一致
    #     for m, n in points_1:
    #         mask[int(m)][int(n)] = 0
    #     eq_points.append(eq_points_1)
    #     state.append(points_1)
    #
    # return px, py, eq_points, state
    """
        根据一个预测框生成一个圆形区域（补丁）。

        参数：
        img_path: 图像路径（未使用，但保留参数接口）
        bbox: 一维列表，元素为 [x1, y1, x2, y2]，代表预测框左上角和右下角坐标。

        返回：
        px: 预测框中心的 x 坐标。
        py: 预测框中心的 y 坐标。
        eq_points: 圆上的等距点坐标列表。
        state: 圆上的点到下一个点的中点坐标列表。
        """
    a = 12  # 圆周上的点数
    e = s  # 半径

    # 获取预测框中心
    x1, y1, x2, y2 = bbox
    #yolo的中心坐标
    # px = (y1 + y2) / 2 - 0.05 * 255  # 用纵坐标表示横坐标，以匹配原始图片的坐标系
    # py = (x1 + x2) / 2
    #隐身衣
    px = (y1 + y2) / 2 - 0.15 * (bbox[3]-bbox[1])  # 用纵坐标表示横坐标，以匹配原始图片的坐标系
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
    # 初始化 mask,这里超出索引的话加以限制
    # mask = np.ones((H, W), dtype=np.int8)
    # for m, n in eq_points:
    #     mask[int(m)][int(n)] = 0

    # 生成中点
    state = []
    for i in range(len(eq_points) - 1):
        pre_x = eq_points[i][0]
        pre_y = eq_points[i][1]
        x = eq_points[i + 1][0]
        y = eq_points[i + 1][1]
        state.append([int(round((pre_x + x) / 2, 2)), int(round((pre_y + y) / 2, 2))])

    # for m, n in state:
    #     mask[int(m)][int(n)] = 0

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


def ori_squre(img_path, H, W, px, py, r1, box):
    square_patch = generate_square_patch(img_path, H, W, px, py, r1)
    square_patch = np.array(square_patch, dtype=np.float32)
    # 把square_patch保存下来便于后面的可视化
    square_patch_show = square_patch
    square_patch = trans(square_patch)
    square_patch_ori = torch.stack([square_patch])
    square_patch_det = F.interpolate(square_patch_ori, (416, 416), mode='bilinear',
                                     align_corners=False)  # 采用双线性插值将不同大小图片上/下采样到统一大小
    square_patch_out = darknet_model(square_patch_det)  # 图像经过模型预测
    square_patch_out = out_transform(square_patch_out)  # 是为了转换成yolov2论文中的格式，去过滤筛选出最正确的预测框
    square_patch_out = [item[0].data for item in square_patch_out]
    # 输入数据，预测结果，H，W原始图片的高和宽，置信度0.6，nms阈值0.4
    # 输出数据：将这四个值弄成一个一行的tensor变量，坐标，置信度，类别概率，类别索引
    reslut = yolo_eval(square_patch_out, H, W, conf_threshold=0.001, nms_threshold=0.4)
    square_patch_out_det_boxes = reslut[:, :5].cpu().numpy()
    square_patch_out_det_classes = reslut[:, -1].long().cpu().numpy()
    # 先把square_patch_show 反归一化转换成unit8类型，再对其进行可视化部分
    square_patch_show = (square_patch_show * 255).astype(np.uint8)  # 转换为 0-255 的 uint8
    square_patch_show = Image.fromarray(square_patch_show)
    square_patch_im2show = draw_detection_boxes(square_patch_show, square_patch_out_det_boxes,
                                                square_patch_out_det_classes, class_names=classes)
    # 提取出bbox列表和置信度列表
    boxes = reslut[:, :4].tolist()  # 提取每行前四个值，并转换为 Python 列表
    confidences = reslut[:, 4:5]  # 提取每行的第 5 个值，保持为张量，形状为 (4, 1)
    max_iou, max_index = find_max_iou(box, boxes)
    # print(f"最大 IoU 值: {max_iou}")
    # print(f"最大 IoU 对应的索引: {max_index}")
    # plt.figure()
    # plt.imshow(square_patch_im2show)
    # plt.show(block=True)
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
    # 读取行人数据集
    final_dir = '../workspace/cross_modal_patch_attack/wuli/object_score/final'  #不同的补丁保存不同的路径（改）
    change_dir = '../workspace/cross_modal_patch_attack/wuli/object_score/final'
    dataset_dir = 'ren'  # 替换为你的数据集路径
    # 获取文件夹中所有图片的路径（支持常见图片格式）
    pos_image_paths = glob.glob(os.path.join(dataset_dir, '*.[jJ][pP][gG]')) + \
                      glob.glob(os.path.join(dataset_dir, '*.[pP][nN][gG]')) + \
                      glob.glob(os.path.join(dataset_dir, '*.[jJ][pP][eE][gG]')) + \
                      glob.glob(os.path.join(dataset_dir, '*.[bB][mM][pP]')) + \
                      glob.glob(os.path.join(dataset_dir, '*.[tT][iI][fF][fF]'))
    for img_path in pos_image_paths:
        # # 读取图片转为张量
        # ceshi_sample = Image.open("workspace/cross_modal_patch_attack/yinshenyi-result/tmp_dir_visible/0.png")
        # # 显示图片
        # # visible_sample.show()
        # ceshi_input = trans(ceshi_sample)  # to tensor
        # # 增加维度，其实每次送入的只有一个图片
        # ceshi_ori = torch.stack([ceshi_input])  # N C H W
        # ceshi_det = F.interpolate(ceshi_ori, (416, 416), mode='bilinear',
        #                             align_corners=False)  # 采用双线性插值将不同大小图片上/下采样到统一大小
        # # 原始图像大小前，图像的宽和高（因为yolo要求模型的输入是416*416大小的）
        # H, W = ceshi_sample.size[1], ceshi_sample.size[0]
        # out = darknet_model(ceshi_det)
        # out = out_transform(out)  # 是为了转换成yolov2论文中的格式，去过滤筛选出最正确的预测框
        # out = [item[0].data for item in out]
        # # 输入数据，预测结果，H，W原始图片的高和宽，置信度0.6，nms阈值0.4
        # # 输出数据：将这四个值弄成一个一行的tensor变量，坐标，置信度，类别概率，类别索引
        # reslut = yolo_eval(out, H, W, conf_threshold=0.01, nms_threshold=0.4)
        # det_boxes = reslut[:, :5].cpu().numpy()
        # det_classes = reslut[:, -1].long().cpu().numpy()
        # im2show = draw_detection_boxes(ceshi_sample, det_boxes, det_classes, class_names=classes)
        # plt.figure()
        # plt.imshow(im2show)
        # plt.show()
        # 读取图片转为张量
        print("image_name:", img_path)
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
        out = darknet_model(visible_det)
        out = out_transform(out)  # 是为了转换成yolov2论文中的格式，去过滤筛选出最正确的预测框
        out = [item[0].data for item in out]
        # 输入数据，预测结果，H，W原始图片的高和宽，置信度0.6，nms阈值0.4
        # 输出数据：将这四个值弄成一个一行的tensor变量，坐标，置信度，类别概率，类别索引
        reslut = yolo_eval(out, H, W, conf_threshold=0.6, nms_threshold=0.4)
        # 可视化
        # det_boxes = reslut[:, :5].cpu().numpy()
        # det_classes = reslut[:, -1].long().cpu().numpy()
        # im2show = draw_detection_boxes(visible_sample, det_boxes, det_classes, class_names=classes)
        # plt.figure()
        # plt.imshow(im2show)
        # plt.show()
        # 提取出bbox列表和置信度列表
        #如果reslut是空的，那么就保存原始图片跳到下一轮即可
        file_name = os.path.basename(img_path)
        if len(reslut)==0:
            final_zero=os.path.join(final_dir, file_name)
            visible_sample.save(final_zero)
            continue
        boxes = reslut[:, :4].tolist()  # 提取每行前四个值，并转换为 Python 列表，这里的box是相对于原始图片的box
        confidences = reslut[:, 4:5]  # 提取每行的第 5 个值，保持为张量，形状为 (4, 1)
        # 改 生成的是两个补丁区域，和论文中一致，是生成的两个补丁，eq_points是在圆的边缘均分的几个点，而points是eq_points中两点之间的中点的集合
        # infrared_img是原始图片的路径
        L = confidences.size(0)
        # 这是后面循环给多个人添加补丁的路径
        final_path = os.path.join(final_dir, file_name)
        for i in range(L):
            # 由于我要循环的给图片添加补丁，因此除了第一个图片以外，第二次迭代循环我需要加载保存的已经添加部分补丁的图片
            # 因此这里的visible_ori需要重新加载为新的图片，然后看看img_path需要变不
            print("第{}轮".format(i))
            if i == 0:
                visible_img = visible_ori
            else:
                img_path = final_path
                visible_s = Image.open(img_path).convert('RGB')
                visible_i = trans(visible_s)  # to tensor
                visible_img = torch.stack([visible_i])  # N C H W
            # if i!=21:
            #     continue
            # else:
            #     visible_img = visible_ori
            # 这里的box是当前目标的框，补丁大小为目标大小的20%
            box = boxes[i]
            #这里r和r1改为w，h中最小值的1/2,k暂时保存w，h中最小的那个
            #这里是按照隐身衣和yolo补丁中确定大小的方式来确定的，324是指原始补丁的大小，按照比例来缩放的
            target_size = math.sqrt((((box[2]-box[0])*0.2) ** 2) + (((box[3]-box[1])*0.2) ** 2))
            r1=target_size
            s=r1*r1
            r=math.sqrt(s/math.pi)
            k=min(box[2]-box[0],box[3]-box[1])
            px, py, eq_points, state = get_state(img_path, box, H, W, r)
            prob_ori_visible_index = confidences[i]
            points_index = list(chain.from_iterable(state))
            # 计算初始方形补丁的大小,并且将补丁贴到指定位置,得到最初的方形补丁的置信度
            # 返回初始方形补丁的置信度的值，与后面补丁变形的方法进行比较
            r1=int(r1)
            r2=r/2 #限制内圆的范围
            ori_squre_confidence = ori_squre(img_path, H, W, px, py, r1, box)
            print(ori_squre_confidence)
            # eq_points_index=eq_points[i]
            # px_idex=px[i]
            # py_idex = py[i]
            # 这是限制补丁在一定范围内变化,以中心点延申出去作为补丁区域的限制，最大范围就是补丁初始化时候随机变化的最大范围
            x_left, x_right, y_head, y_leg,rg = limit_region(px, py, r,k,box)
            rg=int(rg)
            if rg<=0:
                rg=6
            dea = DifferentialEvolutionAlgorithm(15, 24, points_index, eq_points, [px, py],
                                                 [y_head, y_leg, x_left, x_right],
                                                 visible_img, darknet_model,
                                                 prob_ori_visible_index, img_path, 20, [1, 0.6], H, W, r, box,
                                                 ori_squre_confidence,rg,r2,r1,i)
            dea.solve()  # 图片中的x，y与张量中可视化的坐标轴相反，因此我的补丁打印到图片中存在位置偏差
# for img_path in os.listdir(infrared_dir):
#     infrared_img = infrared_dir + '/' + img_path  # 读取每一张图片
#     visible_img = visible_dir + '/' + img_path
#     infrared_sample = Image.open(infrared_img)  # 图片的形式打开
#     visible_sample = Image.open(visible_img)
#     infrared_input = trans(infrared_sample)  # transform转为张量
#     visible_input = trans(visible_sample)  # to tensor
#     # 增加维度，相当于增加批次维度，将多个大小相同的图像合并到一起
#     infrared_ori = torch.stack([infrared_input])  # N C H W
#     visible_ori = torch.stack([visible_input])  # N C H W
#     # 上采样改变图片的大小
#     infrared_det = F.interpolate(infrared_ori, (416, 416), mode='bilinear',
#                                  align_corners=False)  # 采用双线性插值将不同大小图片上/下采样到统一大小
#     visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear',
#                                 align_corners=False)  # 采用双线性插值将不同大小图片上/下采样到统一大小
#     H, W = infrared_sample.size[1], infrared_sample.size[0]  # 上采样前红外图像的高和宽
#     #这里的bbox要确认一下是中心坐标加宽高，还是左上角坐标加右下角坐标，确定了是左上角坐标加右下角坐标
#     bbox, prob_infrared = detect_infrared(threat_infrared_model, infrared_det)
#     # 由于图像进行上采样改变过尺寸，因此得到的bbox需要按照比例进行改变，以便适应原始图片的大小
#     bbox[0], bbox[1], bbox[2], bbox[3] = int(bbox[0] * W / 416), int(bbox[1] * H / 416), int(
#         bbox[2] * W / 416), int(bbox[3] * H / 416)
#     print(img_path)
#     # 这里为什么不获取可见光的图片预测的bbox是因为是同一张图片，只是一个是可见光，一个是红外图像
#     _, prob_visible = detect_visible(threat_visible_model, visible_det)
#     print('Origin infared score: {}\nOrigin visible score: {}'.format(prob_infrared, prob_visible))
#     x_left, x_right, y_head, y_leg = limit_region(bbox)  # 这是限制补丁在检测框里面的区域，也就是限制补丁的大小
#     print(limit_region(bbox))
#     prob_ori_infrared = prob_infrared
#     prob_ori_visible = prob_visible
#     # get_state根据给定bbox，计算两个补丁区域的中心点坐标并为每个补丁区域生成一组等距分布的点和中间点，作为对抗补丁的状态初始化，就是论文里面补丁区域点的初始分布
#     # 生成的是两个补丁区域，和论文中一致，是生成的两个补丁，eq_points是在圆的边缘均分的几个点，而points是eq_points中两点之间的中点的集合
#     px_1, py_1, px_2, py_2, eq_points, state = get_state(infrared_img, bbox)  # get the initial state
#     # points是eq_points中两点之间的中点的集合，是最初的点
#     points = list(chain.from_iterable(state[0])) + list(
#         chain.from_iterable(state[1]))  # change state from 2d to 1d for the input of network
#     infrared_score_before = prob_infrared
#     visible_score_before = prob_visible
#     min_infrared_score = prob_infrared
#     min_visible_score = prob_visible
#
#     dea = DifferentialEvolutionAlgorithm(30, 48, points, eq_points, [px_1, py_1, px_2, py_2],
#                                          [y_head, y_leg, x_left, x_right],
#                                          infrared_ori, visible_ori, threat_infrared_model, threat_visible_model,
#                                          prob_ori_infrared, prob_ori_visible, img_path, 200, [1, 0.6], H, W)
#     dea.solve()
