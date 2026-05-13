import os
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import math

from pandas._libs.parsers import k

# 使用 DPR 版本的补丁形状构建方法
from attack_utils.dpr_inria import dpr_multi_mask as spline_multi_mask
from attack_utils.spline_inria import get_multi_mask
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from yolov3.detect_infrared import detect_infrared
from yolov3.detect_visible import detect_visible
from pytorchyolov3.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from util.visualize import draw_detection_boxes
from generate_squre import find_max_iou
from generate_squre_change import generate_square_patch

# 结果路径单独放到 dpr 目录，避免和原方法混在一起
tmp_dir_vis = '../workspace/cross_modal_patch_attack/yolov3_image_dpr/doepatch_yyf/tmp_dir_visible'
mask_dir = '../workspace/cross_modal_patch_attack/yolov3_image_dpr/doepatch_yyf/mask'
final_dir = '../workspace/cross_modal_patch_attack/yolov3_image_dpr/doepatch_yyf/final'
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


def patch_final(mask, patch_img):
    mask = mask.astype(np.uint8)
    patch_img = (patch_img * 255).astype(np.uint8)
    patch_img = patch_img[0]
    patch_img = np.transpose(patch_img, (1, 2, 0))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours) == 0:
        h_mask, w_mask = mask.shape[:2]
        zeros_patch = np.zeros((3, h_mask, w_mask), dtype=np.uint8)
        return zeros_patch
    contour = contours[0]
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    h, w = patch_img.shape[:2]
    patch_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    mask_corners = np.float32(box)
    M = cv2.getPerspectiveTransform(patch_corners, mask_corners)
    warped_patch = cv2.warpPerspective(patch_img, M, (mask.shape[1], mask.shape[0]))
    warped_patch_new = warped_patch.transpose(2, 0, 1)
    return warped_patch_new


def patch(image_data, H, W, circle, s, region):
    s1 = math.ceil(region[1] - region[0])
    s2 = math.ceil(region[3] - region[2])
    image_data = image_data[0]
    image_data = np.transpose(image_data, (1, 2, 0))
    if image_data.max() <= 1:
        image_data = (image_data * 255).astype(np.uint8)
    else:
        image_data = image_data.astype(np.uint8)
    image_pil = Image.fromarray(image_data)
    image_resized = image_pil.resize((W, H))
    resized_image_data = np.array(image_resized)
    resized_image_data = np.transpose(resized_image_data, (2, 0, 1))
    scaled_image = Image.fromarray(np.transpose(resized_image_data, (1, 2, 0)).astype(np.uint8))
    scaled_image = scaled_image.resize((s2, s1))
    scaled_image_data = np.array(scaled_image)
    scaled_image_data = np.transpose(scaled_image_data, (2, 0, 1))
    output_image = np.zeros_like(resized_image_data)
    center_x, center_y = circle[1], circle[0]
    w, h = s2, s1
    x_start = int(max(center_x - w // 2, 0))
    y_start = int(max(center_y - h // 2, 0))
    x_end = int(min(center_x + w // 2, resized_image_data.shape[2]))
    y_end = int(min(center_y + h // 2, resized_image_data.shape[1]))
    output_image[:, y_start:y_end, x_start:x_end] = scaled_image_data[:, 0:(y_end - y_start), 0:(x_end - x_start)]
    output_image_pil = Image.fromarray(np.transpose(output_image, (1, 2, 0)).astype(np.uint8))
    return output_image


def GrieFunc(vardim, x, visible_ori, threat_visible_model, prob_ori_visible, img_name, step_number, h, w, circle, s,
             box, ori_squre_confidence, region):
    """
    适应度函数：使用 DPR 表示（中心 + 多条射线的半径），不再使用锚点坐标。
    - vardim: 射线条数 K
    - x: 当前个体的半径向量 [r_0, r_1, ..., r_{K-1}]
    - circle: [px, py]，补丁中心
    其他部分（掩码 -> 贴补丁 -> 置信度评估）保持不变。
    """
    K = vardim
    radii = np.array(x, dtype=np.float32)

    # 控制射线长度范围：相对基础半径 s 限制在 [0.5s, 1.5s]，避免过长/过短
    base_r = float(s)
    r_min = 0.5 * base_r
    r_max = 1.5 * base_r
    radii = np.clip(radii, r_min, r_max)
    px, py = circle[0], circle[1]  # 注意：px 对应图像的 y 方向，py 对应 x 方向，沿用原代码坐标系

    # 固定 K 条射线的角度（均匀分布）
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)

    # 由中心点和半径向量构造轮廓点列表 points[0]，供 dpr_multi_mask 使用
    contour_points = []
    for r, theta in zip(radii, angles):
        yy = px + r * np.cos(theta)  # 对应图像的行坐标
        xx = py + r * np.sin(theta)  # 对应图像的列坐标
        contour_points.append([xx,yy])

    state = [contour_points]

    # 使用 DPR 的掩码生成方式
    mask = spline_multi_mask(state, h, w)
    mask = trans(mask)
    mask = mask[0].cpu().detach().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask_path = mask_dir + '/{}.png'.format(step_number)
    mask.save(mask_path)
    mask = cv2.imread(mask_path)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)
    cv2.imwrite(mask_dir + '/{}.png'.format(step_number), mask)
    fig = mask_dir + '/{}.png'.format(step_number)
    mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask) / 255
    mask = mask.astype(np.int8)
    mask = mask ^ (mask & 1 == mask)
    image = cv2.imread('DOEPatch-yyf.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = image.astype(np.float32)
    image_array = np.transpose(image, (2, 0, 1))
    image_data = np.expand_dims(image_array, axis=0)
    Patch = patch_final(mask, image_data)
    if Patch.max() > 1:
        Patch = Patch / 255.0
    x_adv = visible_ori * (1 - mask) + mask * Patch
    adv_final = x_adv[0].cpu().detach().numpy()
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    adv_sample = Image.fromarray(adv_x_255)
    save_path = tmp_dir_vis + '/{}.png'.format(step_number)
    adv_sample.save(save_path)

    with open(tmp_dir_vis + '/{}.png'.format(step_number), 'rb') as fig:
        sample = Image.open(fig)
        visible_input = trans(sample)
        visible_ori = torch.stack([visible_input])
        visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear',
                                    align_corners=False)
        nms_thres = 0.4
        conf_thres = 0.001
        with torch.no_grad():
            detections = threat_visible_model(visible_det)
            detections = non_max_suppression(detections, h, w, conf_thres, nms_thres)
        detections = detections[0]
        detections = detections[detections[:, 5] == 0]
        boxes = detections[:, :4].tolist()
        confidences = detections[:, 4:5]
        max_iou, max_index = find_max_iou(box, boxes)
        max_confidences = confidences[max_index]
    if (max_confidences.item() < ori_squre_confidence.item()):
        return max_confidences
    else:
        return ori_squre_confidence


def ori_squre(img_path, H, W, px, py, r1, file_name):
    square_patch = generate_square_patch(img_path, H, W, px, py, r1, file_name)
    return square_patch


class DEIndividual:
    def __init__(self, vardim, points, region):
        self.vardim = vardim
        self.points = points
        self.fitness = 0
        self.region = region

    def generate(self, h, w, rg):
        length = self.vardim
        self.chrom = np.zeros(length)
        for i in range(0, length):
            # 这里的 points[i] 作为初始半径（通常为基础半径 s），
            # 在其附近随机扰动，rg 控制扰动幅度
            self.chrom[i] = self.points[i] + np.random.uniform(-rg, rg)

    def calculateFitness(self, visible_ori, threat_visible_model, prob_ori_visible, img_name, step_number, h, w, circle,
                         s, box, ori_squre_confidence, region):
        self.fitness = GrieFunc(
            self.vardim, self.chrom, visible_ori, threat_visible_model, prob_ori_visible, img_name, step_number, h, w,
            circle, s, box, ori_squre_confidence, region)


class DifferentialEvolutionAlgorithm:
    def __init__(self, sizepop, vardim, points, eq_points, circle, region, visible_ori, darknet_model,
                 prob_ori_visible, img_name, MAXGEN, params, h, w, s, box, ori_squre_confidence, rg, r2, r1):
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.points = points
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params
        self.circle = circle
        self.eq_points = eq_points  # DPR 版本中不再使用，但保留字段以兼容接口
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
        self.rg = rg
        self.r2 = r2
        self.r1 = r1

    def initialize(self):
        for i in range(0, self.sizepop):
            ind = DEIndividual(self.vardim, self.points, self.region)
            ind.generate(self.h, self.w, self.rg)
            self.population.append(ind)

    def evaluate(self, x, ori_squre_confidence):
        x.calculateFitness(self.visible_ori, self.threat_visible_model,
                           self.prob_ori_visible, self.img_name, self.step_number, self.h, self.w, self.circle, self.s,
                           self.box, ori_squre_confidence, x.region)

    def solve(self):
        self.step_number = 0
        self.t = 0
        self.initialize()
        for i in range(0, self.sizepop):
            self.evaluate(self.population[i], self.ori_squre_confidence)
            self.step_number += 1
            self.fitness[i] = self.population[i].fitness.item()
        best = np.min(self.fitness)
        bestIndex = np.argmin(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = self.best.fitness
        self.trace[self.t, 1] = self.avefitness
        print("Generation %d: optimal Uobj value is: %f; average Uobj  value is %f;ori_square Uobj value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1], self.ori_squre_confidence))
        # 根据当前最优半径向量构造一次掩码用于可视化
        x = self.best.chrom
        K = self.vardim
        radii = np.array(x, dtype=np.float32)
        base_r = float(self.s)
        r_min = 0.5 * base_r
        r_max = 1.5 * base_r
        radii = np.clip(radii, r_min, r_max)
        px, py = self.circle[0], self.circle[1]
        angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
        contour_points = []
        for r, theta in zip(radii, angles):
            yy = px + r * np.cos(theta)
            xx = py + r * np.sin(theta)
            contour_points.append([xx,yy])
        state = [contour_points]
        mask = spline_multi_mask(state, self.h, self.w)
        mask = trans(mask)
        mask = mask[0].cpu().detach().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask)
        optimical_dir = '../workspace/result_dpr'
        file_name = os.path.basename(self.img_name)
        mask_path = optimical_dir + '/{}_{}.png'.format(file_name, self.t)
        os.makedirs(optimical_dir, exist_ok=True)
        mask.save(mask_path)
        mask = cv2.imread(mask_path)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)
        k = 0
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            for i in range(0, self.sizepop):
                vi = self.mutationOperation(i)
                ui = self.crossoverOperation(i, vi, self.r2)
                xi_next = self.selectionOperation(i, ui)
                self.population[i] = xi_next
            for i in range(0, self.sizepop):
                self.fitness[i] = self.population[i].fitness
            best = np.min(self.fitness)
            bestIndex = np.argmin(self.fitness)
            if best < self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
                k = 0
            else:
                k = k + 1
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            print(
                "Generation %d: optimal function value is: %f; average function value is %f;ori_square Uobj value is %f" % (
                    self.t, self.trace[self.t, 0], self.trace[self.t, 1], self.ori_squre_confidence))
            if self.t == self.MAXGEN - 1 or k == 5:
                # 用最终最优半径向量生成掩码
                x = self.best.chrom
                K = self.vardim
                radii = np.array(x, dtype=np.float32)
                base_r = float(self.s)
                r_min = 0.5 * base_r
                r_max = 1.5 * base_r
                radii = np.clip(radii, r_min, r_max)
                px, py = self.circle[0], self.circle[1]
                angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
                contour_points = []
                for r, theta in zip(radii, angles):
                    yy = px + r * np.cos(theta)
                    xx = py + r * np.sin(theta)
                    contour_points.append([xx,yy])
                state = [contour_points]
                mask = spline_multi_mask(state, self.h, self.w)
                mask = trans(mask)
                mask = mask[0].cpu().detach().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = Image.fromarray(mask)
                final_name = os.path.basename(self.img_name)
                mask_path = mask_dir + '/{}.png'.format(final_name)
                mask.save(mask_path)
                mask = cv2.imread(mask_path)
                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)
                cv2.imwrite(mask_dir + '/{}.png'.format(final_name), mask)
                fig = mask_dir + '/{}.png'.format(final_name)
                mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
                mask = np.array(mask) / 255
                mask = mask.astype(np.int8)
                mask = mask ^ (mask & 1 == mask)
                image = cv2.imread('DOEPatch-yyf.png')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255.0
                image = image.astype(np.float32)
                image_array = np.transpose(image, (2, 0, 1))
                image_data = np.expand_dims(image_array, axis=0)
                Patch = patch_final(mask, image_data)
                tuxing = mask * Patch
                tuxing = tuxing.transpose(1, 2, 0)
                if Patch.max() > 1:
                    Patch = Patch / 255.0
                x_adv = self.visible_ori * (1 - mask) + mask * Patch
                adv_final = x_adv[0].cpu().detach().numpy()
                adv_final = (adv_final * 255).astype(np.uint8)
                adv_x_255 = np.transpose(adv_final, (1, 2, 0))
                save_path = final_dir + '/{}'.format(final_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.imsave(save_path, adv_x_255)
                break
        print("Optimal function value is: %f; " % self.trace[self.t, 0])
        print('Optimal solution is:')
        print(self.best.chrom)

    def selectionOperation(self, i, ui):
        xi_next = copy.deepcopy(self.population[i])
        xi_next.chrom = ui
        self.evaluate(xi_next, self.ori_squre_confidence)
        self.step_number += 1
        if xi_next.fitness < self.population[i].fitness:
            return xi_next
        else:
            return self.population[i]

    def crossoverOperation(self, i, vi, r2):
        """
        DPR 版本的交叉操作：标准 DE/bin 交叉，直接在半径向量上操作。
        r2 这里不再作为几何距离使用，仅保留参数以兼容接口。
        """
        k = np.random.randint(0, self.vardim - 1)  # 至少有一个基因来自变异向量
        ui = np.zeros(self.vardim)
        for j in range(0, int(self.vardim)):
            pick = random.random()
            if pick < self.params[0] or j == k:
                ui[j] = vi[j]
            else:
                ui[j] = self.population[i].chrom[j]
        return ui

    def mutationOperation(self, i):
        a = np.random.randint(0, self.sizepop - 1)
        while a == i:
            a = np.random.randint(0, self.sizepop - 1)
        b = np.random.randint(0, self.sizepop - 1)
        while b == i or b == a:
            b = np.random.randint(0, self.sizepop - 1)
        c = np.random.randint(0, self.sizepop - 1)
        while c == i or c == b or c == a:
            c = np.random.randint(0, self.sizepop - 1)
        vi = self.population[c].chrom + self.params[1] * \
             (self.population[a].chrom - self.population[b].chrom)
        return vi

    def printResult(self):
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

