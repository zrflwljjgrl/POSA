import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from torchvision import transforms
from yolov3.detect_infrared import load_infrared_model, detect_infrared
from yolov3.detect_visible import load_visible_model, detect_visible
from itertools import chain
from adv_yolo.load_data import *
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
from generate_squre import generate_square_patch,find_max_iou
import matplotlib
from itertools import chain
from adv_yolo.load_data import *
from yolo_eval import yolo_eval
from generate_squre_ceshi import generate_square_patch, find_max_iou
from pytorchyolov3.models import load_model
from pytorchyolov3.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from PIL import Image
def out_transform(out):
    bsize, _, h, w = out.size()  # 获取批次大小，高和宽
    out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * 5, 5+80)
    xy_pred = torch.sigmoid(out[:, :, 0:2])  # 提取出xy中心坐标
    conf_pred = torch.sigmoid(out[:, :, 4:5])  # 提取出置信度
    hw_pred = torch.exp(out[:, :, 2:4])  # 提取出宽和高
    class_score = out[:, :, 5:]  # 提取出分类得分
    class_pred = F.softmax(class_score, dim=-1)  # dim是指在class_score哪个维度上进行softmax
    delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)  # 连接xy中心点和宽高两个数据
    return delta_pred, conf_pred, class_pred
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
    #加载yoloV3的模型
    model_path = '../adv_yolo/yolov3.cfg'
    weights_path = '../weights/yolov3.weights'
    yolov3 = load_model(model_path, weights_path)
    yolov3 = yolov3.eval()
    #处理图片数据为张量
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 数据集路径
    dataset_dir = '../workspace'  # 替换为你的数据集路径
    train_dir = os.path.join(dataset_dir, 'cross_modal_patch_attack', 'yolov2_image/DOEpatch-yyf')  #gai
    # 获取文件夹中的所有图片路径
    image_folder = os.path.join(train_dir, 'square')  # 假设图片存放在 'images' 文件夹中
    image_paths = [os.path.join(image_folder, img_name) for img_name in sorted(os.listdir(image_folder)) if
                   img_name.endswith(('.jpg', '.png', '.jpeg'))]

    # 创建保存对抗样本检测结果的目录
    output_dir = "../new_labels/yolov3/DOEPatch-yyf/detection-results_06/square_sample"  #gai
    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    # 循环读取每一张图片,并将预测结果保存为txt文档符合计算AP的要求
    for img_path in image_paths:
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
        # boxes = detections[:, :4].tolist()
        # confidences = detections[:, 4:5]
        # bbox, prob_infrared = detect_infrared(yolov3, visible_det,H,W)
        # 保存检测结果
        # 保存检测结果
        img_name = os.path.splitext(os.path.basename(img_path))[0]  # 获取图片名称（不带扩展名）
        output_path = os.path.join(output_dir, f"{img_name}.txt")
        with open(output_path, 'w') as f_out:
            # 遍历每个检测结果（detections是二维Tensor，形状为 [N, 7]）
            for detection in detections:
                # 解析检测结果的7个值
                x1 = detection[0].item()  # 检测框左上角x（在416x416输入下的坐标）
                y1 = detection[1].item()  # 检测框左上角y
                x2 = detection[2].item()  # 检测框右下角x
                y2 = detection[3].item()  # 检测框右下角y
                conf = detection[4].item()  # 置信度
                cls_id = int(detection[5].item())  # 类别索引（如0对应'person'）

                # 获取类别名称
                class_name = classes[cls_id]

                # 将坐标从 416x416 缩放到原始图像尺寸，这里不需要，已经进行了缩放
                # x1 = int(x1_416 * W / 416)
                # y1 = int(y1_416 * H / 416)
                # x2 = int(x2_416 * W / 416)
                # y2 = int(y2_416 * H / 416)

                # 写入文件（格式：class_name confidence x1 y1 x2 y2）
                f_out.write(f"{class_name} {conf:.6f} {x1} {y1} {x2} {y2}\n")

        print(f"检测结果已保存：{output_path}")
        #yolov2检测方法
        # out = darknet_model(visible_det)
        # out = out_transform(out)  # 是为了转换成yolov2论文中的格式，去过滤筛选出最正确的预测框
        # out = [item[0].data for item in out]
        # # 输出数据：detections是个二维的tensor变量(目标数量，7)，第二个7里面包含7个值前面四个是x1,y1,x2,y2,目标置信度，目标类别预测类别概率，目标类别在classes里面的索引
        # detections = yolo_eval(out, H, W, conf_threshold=0.6, nms_threshold=0.4)
        # # 保存检测结果
        # # 保存检测结果
        # img_name = os.path.splitext(os.path.basename(img_path))[0]  # 获取图片名称（不带扩展名）
        # output_path = os.path.join(output_dir, f"{img_name}.txt")
        # with open(output_path, 'w') as f_out:
        #     # 遍历每个检测结果（detections是二维Tensor，形状为 [N, 7]）
        #     for detection in detections:
        #         # 解析检测结果的7个值
        #         x1 = detection[0].item()  # 检测框左上角x（在416x416输入下的坐标）
        #         y1 = detection[1].item()  # 检测框左上角y
        #         x2 = detection[2].item()  # 检测框右下角x
        #         y2 = detection[3].item()  # 检测框右下角y
        #         conf = detection[4].item()  # 置信度
        #         class_prob = detection[5].item()  # 类别预测概率（通常可以忽略）
        #         cls_id = int(detection[6].item())  # 类别索引（如0对应'person'）
        #
        #         # 获取类别名称
        #         class_name = classes[cls_id]
        #
        #         # 将坐标从 416x416 缩放到原始图像尺寸，这里不需要，已经进行了缩放
        #         # x1 = int(x1_416 * W / 416)
        #         # y1 = int(y1_416 * H / 416)
        #         # x2 = int(x2_416 * W / 416)
        #         # y2 = int(y2_416 * H / 416)
        #
        #         # 写入文件（格式：class_name confidence x1 y1 x2 y2）
        #         f_out.write(f"{class_name} {conf:.6f} {x1} {y1} {x2} {y2}\n")
        #
        # print(f"检测结果已保存：{output_path}")
