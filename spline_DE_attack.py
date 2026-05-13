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
from PIL import Image
from torchvision import transforms
from yolov3.detect_infrared import load_infrared_model, detect_infrared
from yolov3.detect_visible import load_visible_model, detect_visible
from itertools import chain
from DE import DifferentialEvolutionAlgorithm
content = 1
trans = transforms.Compose([
                transforms.ToTensor(),
            ])
#导入权重，加载yolo模型
threat_infrared_model = load_infrared_model()
threat_visible_model = load_visible_model()
#设置可见光和红外光下的数据集
infrared_dir = './workspace/cross_modal_patch_attack/dataset/attack_infrared'
visible_dir = './workspace/cross_modal_patch_attack/dataset/attack_visible'

def limit_region(bbox):
    x_left = bbox[0] + (bbox[2] - bbox[0]) / 4
    x_right = bbox[2] - (bbox[2] - bbox[0]) / 4
    y_low = bbox[1]
    y_high = bbox[3]
    y_head = y_low + (y_high - y_low) / 4
    y_leg = y_low + (y_high - y_low) / 2
    return x_left, x_right, y_head, y_leg

def get_state(img_path, bbox):
    #bbox(x1,y1,x2,y2) (x1,y1)左上角坐标 (x2,y2)右下角坐标
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    points  = []
    patch_1 = []
    patch_2 = []
    #高度宽度10等分
    w_step = int(bbox_width / 10)
    h_step = int(bbox_height / 10)
    bbox = list(map(int, bbox)) #将所以元素转变为整数，并返回一个新的列表
    x_left, x_right = bbox[0], bbox[2]
    y_up, y_below   = bbox[1], bbox[3]
    #两个补丁的中心点，需要去确定一下定多少，这里的px应该是纵坐标，py指的是横坐标，我现在才明白，为什么这里px用纵坐标表示才对应图片的横坐标
    px_1 = (y_up+2.5*h_step + y_up+3*h_step) / 2
    py_1 = (x_left+5*w_step + x_right-3*w_step) / 2

    px_2 = (y_up+4*h_step + y_up+6*h_step) / 2
    py_2 = (x_right-5*w_step + x_right-4*w_step) / 2

    a = 12 
    e = 15
    eq_points = []
    state = []
    # ---patch 1
    eq_points_1 = []
    points_1 = []
    #循环生成补丁区域的圆，循环生成12个点
    for n in range(1,a+1): #生成等距的点
        xx = px_1 + round(e*np.cos(2*np.pi*(n-1)/a),2)  #
        yy = py_1 + round(e*np.sin(2*np.pi*(n-1)/a),2)
        eq_points_1.append([xx,yy]) 
    eq_points_1.append([px_1 + round(e*np.cos(0),2), py_1 + round(e*np.sin(0),2)])
    # mask = np.ones((H, W), dtype=np.int8)  # 初始化mask和原始图片的大小一致
    # for m, n in eq_points_1:
    #     mask[int(m)][int(n)] = 0
    for i in range(len(eq_points_1)-1): #利用上面生成等距的点的中点，作为锚点，跟论文中一致
        pre_x = eq_points_1[i][0]
        pre_y = eq_points_1[i][1]
        x = eq_points_1[i+1][0]
        y = eq_points_1[i+1][1]
        points_1.append([int(round((pre_x+x)/2,2)),int(round((pre_y+y)/2,2))])

    # ---patch 2
    eq_points_2 = []
    points_2 = []
    for n in range(1,a+1):
        xx = px_2 + round(e*np.cos(2*np.pi*(n-1)/a),2)  
        yy = py_2 + round(e*np.sin(2*np.pi*(n-1)/a),2)
        eq_points_2.append([xx,yy]) 
    eq_points_2.append([px_2 + round(e*np.cos(0),2), py_2 + round(e*np.sin(0),2)])
    for i in range(len(eq_points_2)-1):
        pre_x = eq_points_2[i][0]
        pre_y = eq_points_2[i][1]
        x = eq_points_2[i+1][0]
        y = eq_points_2[i+1][1]
        points_2.append([int(round((pre_x+x)/2,2)),int(round((pre_y+y)/2,2))])

    eq_points.append(eq_points_1)
    eq_points.append(eq_points_2)
    state.append(points_1)
    state.append(points_2)

    return px_1, py_1, px_2, py_2, eq_points, state

if __name__ == "__main__":
    for img_path in os.listdir(infrared_dir):
        infrared_img = infrared_dir + '/' + img_path#读取每一张图片
        visible_img = visible_dir + '/' + img_path
        infrared_sample = Image.open(infrared_img)#图片的形式打开
        visible_sample = Image.open(visible_img)  
        infrared_input = trans(infrared_sample) # transform转为张量
        visible_input = trans(visible_sample) # to tensor
        #增加维度，相当于增加批次维度，将多个大小相同的图像合并到一起
        infrared_ori = torch.stack([infrared_input]) # N C H W
        visible_ori = torch.stack([visible_input]) # N C H W
        #上采样改变图片的大小
        infrared_det = F.interpolate(infrared_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
        visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
        H, W = infrared_sample.size[1], infrared_sample.size[0]#上采样前红外图像的高和宽
        bbox, prob_infrared = detect_infrared(threat_infrared_model, infrared_det)
        #由于图像进行上采样改变过尺寸，因此得到的bbox需要按照比例进行改变，这是yolo模型裁剪图片大小后改变图片预测的bbox的方法吗？
        bbox[0], bbox[1], bbox[2], bbox[3] = int(bbox[0]*W/416), int(bbox[1]*H/416), int(bbox[2]*W/416), int(bbox[3]*H/416)
        #这里为什么不获取可见光的图片预测的bbox是因为是同一张图片，只是一个是可见光，一个是红外图像
        _, prob_visible = detect_visible(threat_visible_model, visible_det)
        print('Origin infared score: {}\nOrigin visible score: {}'.format(prob_infrared, prob_visible))
        x_left, x_right, y_head, y_leg = limit_region(bbox) # 这是限制补丁在检测框里面的区域，也就是限制补丁的大小
        print(limit_region(bbox))
        prob_ori_infrared = prob_infrared
        prob_ori_visible = prob_visible
        #get_state根据给定bbox，计算两个补丁区域的中心点坐标并为每个补丁区域生成一组等距分布的点和中间点，作为对抗补丁的状态初始化，就是论文里面补丁区域点的初始分布
        #生成的是两个补丁区域，和论文中一致，是生成的两个补丁，eq_points是在圆的边缘均分的几个点，而points是eq_points中两点之间的中点的集合
        px_1, py_1, px_2, py_2, eq_points, state = get_state(infrared_img, bbox) # get the initial state
        #points是eq_points中两点之间的中点的集合，是最初的点
        points = list(chain.from_iterable(state[0])) + list(chain.from_iterable(state[1])) # change state from 2d to 1d for the input of network
        infrared_score_before = prob_infrared
        visible_score_before = prob_visible
        min_infrared_score  = prob_infrared
        min_visible_score  = prob_visible
        dea = DifferentialEvolutionAlgorithm(30, 48, points, eq_points, [px_1, py_1, px_2, py_2],
       [y_head, y_leg, x_left, x_right],infrared_ori, visible_ori, threat_infrared_model, threat_visible_model,
        prob_ori_infrared, prob_ori_visible, img_path, 200, [1,  0.6], H, W)
        dea.solve()

