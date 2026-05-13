import os
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import math
from attack_utils.spline import spline_multi_mask, get_multi_mask
import cv2
import torch
import torch.nn as nn 
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from yolov3.detect_infrared import detect_infrared
from yolov3.detect_visible import detect_visible

tmp_dir_inf = './workspace/cross_modal_patch_attack/result/tmp_dir_infrared'
tmp_dir_vis = './workspace/cross_modal_patch_attack/result/tmp_dir_visible'
mask_dir = './workspace/cross_modal_patch_attack/result/mask'
final_dir = './workspace/cross_modal_patch_attack/result/final'

trans = transforms.Compose([
                transforms.ToTensor(),
            ])
content_inf = 0
content_vis = 1

def ifcross(p1, p2, p, px, py): #p1,p2是初始的等分点，p是指的vi里面的点，也就是上一步经过变异得到的点，px，py指的是补丁掩码的中心点
    d1 = (p1[0]-px)*(p[1]-py) - (p1[1]-py)*(p[0]-px)  #就是论文中求点到两条线的距离公式，具体怎么得来的我也不太清楚
    d2 = (p2[0]-px)*(p[1]-py) - (p2[1]-py)*(p[0]-px)
    if d1 * d2 < 0:
        return False
    else:
        return True

def compute_dis(p, px, py):
    dis = pow(pow(p[0]-px, 2) + pow(p[1]-py, 2), 0.5)
    return dis
#扰动掩码由spline_multi_mask函数生成
#GrieFunc是核心函数，计算扰动对红外和可见光模型的攻击效果，包括对抗成功率 r_attack 和距离成功的差距 dis_to_success
def GrieFunc(vardim, x, infrared_ori, visible_ori, threat_infrared_model, threat_visible_model, prob_ori_infrared, prob_ori_visible, img_name, step_number, h, w):
    state = []
    p1 = []
    p2 = []
    length = int(len(x)/2)
    #两个区域的坐标分别加入到p1和p2中去
    for i in range(length):
        if i % 2 == 0:
            p1.append([x[i], x[i+1]])
    for i in range(length, 2*length):
        if i % 2 == 0:
            p2.append([x[i], x[i+1]])
    # mask = np.ones((h, w), dtype=np.int8)  # 初始化mask和原始图片的大小一致
    # for m, n in p1:
    #     mask[int(m)][int(n)] = 0
    state.append(p1)
    state.append(p2)
    mask = spline_multi_mask(state, h, w) #这里的h,w是原始图像经过裁剪前的h和w，这一步是生成掩码
    len_x = len(mask)
    len_y = len(mask[0])

    # obtain the mask
    mask = trans(mask) #transfrom转为张量
    mask = mask[0].cpu().detach().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask_path = mask_dir + '/{}.png'.format(step_number)
    # 检查并创建目标文件夹（如果不存在）
    # os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    mask.save(mask_path, quality = 99)
    mask = cv2.imread(mask_path)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)#二值化图像，其中thresh是阈值，意味着小于该阈值的像素值会被设置为maxval 255
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#返回掩码的轮廓，coutours轮廓点的列表，hierarchy轮廓之间的层次信息
    cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)#轮廓绘制在掩码图像上， thickness=-1表示填充轮廓内部，(0, 0, 0)黑色
    cv2.imwrite(mask_dir+'/{}.png'.format(step_number), mask)
    
    # cast mask upon infrared images 掩码应用到红外图像上，红外攻击是让红外图像对应掩码的部分的张量置为0，看起来就是全黑
    fig = mask_dir +'/{}.png'.format(step_number)
    mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask) / 255
    mask = mask.astype(np.int8)
    mask = mask^(mask&1==mask)
    x_adv = infrared_ori * ( 1 - mask ) + mask * content_inf
    adv_final = x_adv[0].cpu().detach().numpy() #只是去掉批次维度
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    adv_sample = Image.fromarray(adv_x_255)
    save_path = tmp_dir_inf + '/{}.png'.format(step_number)
    adv_sample.save(save_path, quality=99)

    # cast mask upon visible images，这里就是把可见光图像中对应掩码的位置的张量变为1
    fig = mask_dir +'/{}.png'.format(step_number)
    mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask) / 255
    mask = mask.astype(np.int8)
    mask = mask^(mask&1==mask)
    x_adv = visible_ori * ( 1 - mask ) + mask * content_vis
    adv_final = x_adv[0].cpu().detach().numpy()  #只是去掉批次维度
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    adv_sample = Image.fromarray(adv_x_255)
    save_path = tmp_dir_vis + '/{}.png'.format(step_number)
    adv_sample.save(save_path, quality=99)
    
    # r_attack
    with open(tmp_dir_inf + '/{}.png'.format(step_number), 'rb') as fig:
        sample = Image.open(fig)
        infrared_input = trans(sample)
        infrared_ori = torch.stack([infrared_input]) # N C H W 加一个批次维度
        #interpolate来处理图片的大小跟resize相比，它可以处理张量数据，而且可以按照批次来处理
        infrared_det = F.interpolate(infrared_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
        _, prob_infrared = detect_infrared(threat_infrared_model, infrared_det)  

    with open(tmp_dir_vis + '/{}.png'.format(step_number), 'rb') as fig:
        sample = Image.open(fig)
        visible_input = trans(sample)
        visible_ori = torch.stack([visible_input]) # N C H W
        visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小  
        _, prob_visible = detect_visible(threat_visible_model, visible_det) 

    r_inf = math.exp(2*((prob_ori_infrared - prob_infrared) / (prob_ori_infrared - 0.7))) #论文中0.7是设置的阈值，prob_infrared和prob_visible应该小于该阈值
    r_vis = math.exp(2*((prob_ori_visible - prob_visible) / (prob_ori_visible - 0.7)))
    r_attack =  min(r_inf, r_vis)

    dis_inf = np.float((prob_ori_infrared - prob_infrared) / (prob_ori_infrared - 0.7))
    dis_vis = np.float((prob_ori_visible - prob_visible) / (prob_ori_visible - 0.7))
    dis_to_success = min(dis_inf, dis_vis)



    return r_attack, dis_to_success, dis_inf, dis_vis


class DEIndividual:

    '''
    individual of differential evolution algorithm
    '''

    def __init__(self,  vardim, points):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim #变量维度，这是48，是不是因为12个点两个补丁，因此有48个值
        self.points = points #锚点坐标
        self.fitness = 0.
        self.distance = 0.
        self.dis_inf = 0.
        self.dis_vis = 0.
    #根据初始值建立初始种群
    def generate(self):
        '''
        generate a random chromsome for differential evolution algorithm
        '''
        len = self.vardim
        #chrom首先设置为全零的数组，然后在下面根据points初始值的位置，按照正负3的差距生成初始种群，个人理解也就是初始的圆按照一定程度进行扭曲得到新的补丁形状
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.points[i] + np.random.randint(-3, 3)
        # print(self.chrom)
                        

    def calculateFitness(self, infrared_ori, visible_ori, threat_infrared_model, threat_visible_model, prob_ori_infrared, prob_ori_visible, img_name, step_number, h, w):
        '''
        calculate the fitness of the chromsome
        '''
        '''
        计算得到适应度，距离，红外图像和可见光图像的得分
        '''
        self.fitness, self.distance, self.dis_inf, self.dis_vis = GrieFunc(
            self.vardim, self.chrom, infrared_ori, visible_ori, threat_infrared_model, threat_visible_model, prob_ori_infrared, prob_ori_visible, img_name, step_number, h, w)

class DifferentialEvolutionAlgorithm:

    '''
    The class for differential evolution algorithm
    '''

    def __init__(self, sizepop, vardim, points, eq_points, circle, region, infrared_ori, visible_ori, threat_infrared_model, threat_visible_model, prob_ori_infrared,\
         prob_ori_visible, img_name, MAXGEN, params, h, w):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        param: algorithm required parameters, it is a list which is consisting of [crossover rate CR, scaling factor F]
        '''
        self.sizepop = sizepop #种群数量 30
        self.MAXGEN = MAXGEN #最大迭代代数 200
        self.vardim = vardim #变量维度 48
        self.points = points # 锚点坐标
        self.population = [] #用于
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params #算法控制参数，包括交叉概率，缩放因子（控制差分变异操作的强度）[1,0.6]
        self.circle = circle # 圆心坐标
        self.eq_points = eq_points #生成锚点的在圆上等距离分布的点
        self.region = region #对锚点变化的区域限制
        self.infrared_ori = infrared_ori #红外原始图像
        self.visible_ori = visible_ori #可见光原始图像
        self.threat_infrared_model = threat_infrared_model #在红外图像上训练的模型
        self.threat_visible_model = threat_visible_model #在可见光图像上训练的模型
        self.prob_ori_infrared = prob_ori_infrared #红外图像中目标置信度得分
        self.prob_ori_visible = prob_ori_visible #可见光图像中目标置信度得分
        self.img_name = img_name #图像路径
        self.step_number = 0 #记录变化的次数
        self.h = h #原始图像的高度
        self.w = w #原始图像的宽

    def initialize(self):
        '''
        initialize the population 初始种群，种群数量由sizepop决定
        '''
        for i in range(0, self.sizepop):
            ind = DEIndividual(self.vardim, self.points)
            ind.generate()
            self.population.append(ind)

    def evaluate(self, x):
        '''
        evaluation of the population fitnesses
        '''  #infrared_ori红外图像，visible_ori可见光图像，self.threat_infrared_model, self.threat_visible_model 模型
        #prob_ori_infrared，prob_ori_visible，置信度，img_name图片路径，step_number
        x.calculateFitness(self.infrared_ori, self.visible_ori, self.threat_infrared_model, self.threat_visible_model, self.prob_ori_infrared,\
             self.prob_ori_visible, self.img_name, self.step_number, self.h, self.w)

    def solve(self):
        '''
        evolution process of differential evolution algorithm
        '''
        '''
          建立初始种群，通过self.initialize()生成不一样形状的初始坐标
        '''
        self.step_number = 0
        self.t = 0
        self.initialize()
        '''
            下面的for循环是在计算生成的种群的适应度，个人理解是找到最优的形状，然后进行补丁形状的确定。
        '''
        for i in range(0, self.sizepop):
            self.evaluate(self.population[i])
            self.step_number += 1 #这里的fitness就是论文里面的J(S)
            self.fitness[i] = self.population[i].fitness

        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = self.best.fitness
        self.trace[self.t, 1] = self.avefitness
        print("Generation %d: optimal function value is: %f; distance to success is %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.best.distance, self.trace[self.t, 1]))
        print("dis_inf:{}, dis_vis:{}".format(self.best.dis_inf, self.best.dis_vis))

        x = self.best.chrom
        state = []
        p1 = []
        p2 = []
        length = int(len(x)/2)  #前一半是第一个补丁，后一半是第二个补丁
        for i in range(length):
            if i % 2 == 0:
                p1.append([x[i], x[i+1]])
        for i in range(length, 2*length):
            if i % 2 == 0:
                p2.append([x[i], x[i+1]])
        state.append(p1)
        state.append(p2)
        mask = spline_multi_mask(state, self.h, self.w)

        # obtain the mask
        mask = trans(mask)
        mask = mask[0].cpu().detach().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask)
        optimical_dir='./workspace/result'
        mask_path = optimical_dir + '/{}_{}.png'.format(self.img_name, self.t)
        mask.save(mask_path, quality = 99)#保存的是初始种群中最优适应度下的掩码
        mask = cv2.imread(mask_path)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1) #填充
        #下面是进行变异交叉验证操作
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            for i in range(0, self.sizepop):#种群的每一个元素进行变异交叉验证后再次计算fitness与变异前对比，如果大于则替换对应的元素，否则保留原来的
                vi = self.mutationOperation(i)  #这里是论文里面Mutation的部分，也就是变异
                ui = self.crossoverOperation(i, vi) #这里是论文里面Crossover的部分，其实就是判断是否符合论文在界内的要求，不符合就去掉变异的坐标，保留其变异前的坐标
                xi_next = self.selectionOperation(i, ui) #
                self.population[i] = xi_next
            for i in range(0, self.sizepop):
                self.fitness[i] = self.population[i].fitness
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            print("Generation %d: optimal function value is: %f; distance to success is %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.best.distance, self.trace[self.t, 1]))
            print("dis_inf:{}, dis_vis:{}".format(self.best.dis_inf, self.best.dis_vis))
            #检查最佳适应度是否超过了e^2或者已经达到了最大迭代数
            if self.best.fitness >= math.exp(2) or self.t == self.MAXGEN - 1:
                x = self.best.chrom  #获取最优适应度下的锚点坐标
                state = []
                p1 = []
                p2 = []
                length = int(len(x)/2)
                for i in range(length):
                    if i % 2 == 0:
                        p1.append([x[i], x[i+1]])
                for i in range(length, 2*length):
                    if i % 2 == 0:
                        p2.append([x[i], x[i+1]])
                state.append(p1)
                state.append(p2)
                mask = spline_multi_mask(state, self.h, self.w) #掩码获取

                # obtain the mask
                mask = trans(mask)
                mask = mask[0].cpu().detach().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = Image.fromarray(mask)
                mask_path = mask_dir + '/{}.png'.format(self.img_name)
                mask.save(mask_path, quality = 99)
                mask = cv2.imread(mask_path)
                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)#二值化处理
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=-1)
                cv2.imwrite(mask_dir+'/{}.png'.format(self.img_name), mask)
                
                # cast mask upon infrared images
                fig = mask_dir +'/{}.png'.format(self.img_name)
                mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
                mask = np.array(mask) / 255
                mask = mask.astype(np.int8)
                mask = mask^(mask&1==mask)#对mask进行按位与操作
                x_adv = self.infrared_ori * ( 1 - mask ) + mask * content_inf #论文中生成对抗补丁的方法
                adv_final = x_adv[0].cpu().detach().numpy()
                adv_final = (adv_final * 255).astype(np.uint8)
                adv_x_255 = np.transpose(adv_final, (1, 2, 0))
                adv_sample = Image.fromarray(adv_x_255)
                save_path = final_dir + '/infrared_{}_{}.png'.format(self.img_name, self.best.fitness)
                adv_sample.save(save_path, quality=99)

                # cast mask upon visible images
                fig = mask_dir +'/{}.png'.format(self.img_name)
                mask = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
                mask = np.array(mask) / 255
                mask = mask.astype(np.int8)
                mask = mask^(mask&1==mask)
                x_adv = self.visible_ori * ( 1 - mask ) + mask * content_vis
                adv_final = x_adv[0].cpu().detach().numpy()
                adv_final = (adv_final * 255).astype(np.uint8)
                adv_x_255 = np.transpose(adv_final, (1, 2, 0))
                adv_sample = Image.fromarray(adv_x_255)
                save_path = final_dir + '/visible_{}_{}.png'.format(self.img_name, self.best.fitness)
                adv_sample.save(save_path, quality=99)
                #下面是在攻击目标检测模型了
                with open(final_dir + '/infrared_{}_{}.png'.format(self.img_name, self.best.fitness), 'rb') as fig:
                    sample = Image.open(fig)
                    infrared_input = trans(sample)
                    infrared_ori = torch.stack([infrared_input]) # N C H W
                    infrared_det = F.interpolate(infrared_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
                    _, prob_infrared = detect_infrared(self.threat_infrared_model, infrared_det)  

                with open(final_dir + '/visible_{}_{}.png'.format(self.img_name, self.best.fitness), 'rb') as fig:
                    sample = Image.open(fig)
                    visible_input = trans(sample)
                    visible_ori = torch.stack([visible_input]) # N C H W
                    visible_det = F.interpolate(visible_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小  
                    _, prob_visible = detect_visible(self.threat_visible_model, visible_det) 
                os.rename(final_dir + '/infrared_{}_{}.png'.format(self.img_name, self.best.fitness), final_dir + '/infrared_{}_{}_{}.png'.format(self.img_name, self.t, prob_infrared))
                os.rename(final_dir + '/visible_{}_{}.png'.format(self.img_name, self.best.fitness), final_dir + '/visible_{}_{}_{}.png'.format(self.img_name, self.t, prob_visible))                
                break

        print("Optimal function value is: %f; " %\
              self.trace[self.t, 0]) #self.trace保存的是最优适应度和平均适应度，也就是平衡红外光和可见光下最佳的
        print ('Optimal solution is:')
        print (self.best.chrom)
        # self.printResult()

    def selectionOperation(self, i, ui):
        '''
        selection operation for differential evolution algorithm
        '''
        xi_next = copy.deepcopy(self.population[i]) #深度拷贝，在进行修改的时候不影响原个体
        xi_next.chrom = ui #这里拷贝了种群里面第i个元素的信息，并将里面的掩码坐标替换为变异后的结果
        self.evaluate(xi_next) #新的种群进行评估得到结果
        self.step_number += 1 #这里为什么要加1？
        if xi_next.fitness > self.population[i].fitness:
            # print("change")
            return xi_next
        else:
            # print("no change")
            return self.population[i]

    def crossoverOperation(self, i, vi): #实现交叉操作
        '''
        crossover operation for differential evolution algorithm
        '''
        px_1, py_1, px_2, py_2 = self.circle[0], self.circle[1], self.circle[2], self.circle[3]
        y_low, y_high, x_left, x_right = self.region[0], self.region[1], self.region[2], self.region[3]
        limit_line = (px_1 + px_2) / 2 #不明白
        k = np.random.randint(0, self.vardim - 1)
        ui = np.zeros(self.vardim)
        for j in range(0, int(self.vardim/2)): #第一个补丁
            if j % 2 == 0:
                dis = compute_dis([vi[j], vi[j + 1]], px_1, py_1) #计算vi中点离圆心的距离
                pick = random.random()
                #ifcross中五个参数，第一个和第二个是通过线段等分的圆的边界点，而vi[j], vi[j+1],是变异的点，这样计算是通过论文中向量的乘积大于0还是小于0的方式来判断是否会存在交叉的现象，
                if (pick < self.params[0] or j == k) and (ifcross(self.eq_points[0][j // 2], self.eq_points[0][j // 2 + 1], [vi[j], vi[j+1]], px_1, py_1) ==False \
                and dis > 8 and vi[j] >  y_low and vi[j] < limit_line and vi[j+1] > x_left and vi[j+1] < x_right):
                #dis>8是不在内圆，后面是为了限制在区域内
                    ui[j] = vi[j]
                    ui[j + 1] = vi[j + 1]
                else: #如果不满足条件就保留原来的个体
                    ui[j] = self.population[i].chrom[j]
                    ui[j + 1] = self.population[i].chrom[j+1]
        for j in range(int(self.vardim/2) ,int(self.vardim)): #第二个补丁
            if j % 2 == 0:
                dis = compute_dis([vi[j], vi[j + 1]], px_1, py_1)
                pick = random.random()
                if (pick < self.params[0] or j == k) and (ifcross(self.eq_points[1][(j - self.vardim // 2) // 2], self.eq_points[1][(j - self.vardim // 2) // 2 + 1], [vi[j], vi[j+1]], px_2, py_2) ==False \
                and dis > 8 and vi[j] >  limit_line and vi[j] < y_high and vi[j+1] > x_left and vi[j+1] < x_right):
                    ui[j] = vi[j]
                    ui[j + 1] = vi[j + 1]
                else:
                    ui[j] = self.population[i].chrom[j]
                    ui[j + 1] = self.population[i].chrom[j+1]
                    
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
            c = np.random.randint(0, self.sizepop - 1) #上述 a，b，c就是找了三个自身各不相同也不同于i的另外三个形状的点
        vi = self.population[c].chrom + self.params[1] * \
            (self.population[a].chrom - self.population[b].chrom)

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
