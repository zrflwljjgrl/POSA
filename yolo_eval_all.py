import torch
import cv2
import numpy as np
from torchvision.ops import nms
from torchvision.transforms import transforms
from adv_yolo.darknet import Darknet
import config as cfg
from util.bbox import generate_all_anchors, xywh2xxyy, box_transform_inv, xxyy2xywh
from util.bbox import box_ious
def yolo_filter_boxes(boxes_pred, conf_pred, classes_pred, confidence_threshold=0.6):
    """
    Filter boxes whose confidence is lower than a given threshold

    Arguments:
    boxes_pred -- tensor of shape (H * W * num_anchors, 4) (x1, y1, x2, y2) predicted boxes
    conf_pred -- tensor of shape (H * W * num_anchors, 1)
    classes_pred -- tensor of shape (H * W * num_anchors, num_classes)
    threshold -- float, threshold used to filter boxes

    Returns:
    filtered_boxes -- tensor of shape (num_positive, 4)
    filtered_conf -- tensor of shape (num_positive, 1)
    filtered_cls_max_conf -- tensor of shape (num_positive, num_classes)
    filtered_cls_max_id -- tensor of shape (num_positive, num_classes)
    """

    # multiply class scores and objectiveness score
    # use class confidence score
    # TODO: use objectiveness (IOU) score or class confidence score
    cls_max_conf, cls_max_id = torch.max(classes_pred, dim=-1, keepdim=True)#得到每个框的最高类别分数和对应类别的索引
    cls_conf = conf_pred * cls_max_conf#类别分数去乘以置信度得到综合的置信度

    pos_inds = (cls_conf > confidence_threshold).view(-1)#布尔变量输出true或者flase表明cls_conf最后一维度的值是否大于置信度阈值

    filtered_boxes = boxes_pred[pos_inds, :] #[pos_inds, :]表示从boxes_pred中按照布尔变量pos_inds的值，按行进行筛选数据

    filtered_conf = conf_pred[pos_inds, :]#下面的三个变量类似

    filtered_cls_max_conf = cls_max_conf[pos_inds, :]

    filtered_cls_max_id = cls_max_id[pos_inds, :]

    return filtered_boxes, filtered_conf, filtered_cls_max_conf, filtered_cls_max_id.float()


def yolo_nms(boxes, scores, threshold):
    """
    Apply Non-Maximum-Suppression on boxes according to their scores

    Arguments:
    boxes -- tensor of shape (N, 4) (x1, y1, x2, y2)
    scores -- tensor of shape (N) confidence
    threshold -- float. NMS threshold

    Returns:
    keep -- tensor of shape (None), index of boxes which should be retain.
    """
    #按照置信度大小从高到底对索引排序，score_sort_index输出的是scores中从大到小置信度的索引
    score_sort_index = torch.sort(scores, dim=0, descending=True)[1]

    keep = []
    #直到处理完所有的预测框为止
    while score_sort_index.numel() > 0:
        #选择当前最高的预测框加入keep
        i = score_sort_index[0]
        keep.append(i)

        if score_sort_index.numel() == 1:
            break
        #cur当前最高置信度的边界框，res其余的边界框
        cur_box = boxes[score_sort_index[0], :].view(-1, 4)
        res_box = boxes[score_sort_index[1:], :].view(-1, 4)
        #计算IOU
        ious = box_ious(cur_box, res_box).view(-1)
        #找到IOU小于阈值的框并且保留，因为大于阈值的框很有可能检测的是相同的物体，只保留置信度最高的即可
        inds = torch.nonzero(ious < threshold).squeeze()

        score_sort_index = score_sort_index[inds + 1].view(-1)

    return torch.LongTensor(keep)


def generate_prediction_boxes(deltas_pred):
    """
    Apply deltas prediction to pre-defined anchors

    Arguments:
    deltas_pred -- tensor of shape (H * W * num_anchors, 4) σ(t_x), σ(t_y), σ(t_w), σ(t_h)

    Returns:
    boxes_pred -- tensor of shape (H * W * num_anchors, 4)  (x1, y1, x2, y2)
    """

    H = int(cfg.test_input_size[0] / cfg.strides)#yolov2的格子大小 13*13
    W = int(cfg.test_input_size[1] / cfg.strides)

    anchors = torch.FloatTensor(cfg.anchors)#yolov2对应训练数据集下的先验框
    #13*13所有网格的中心坐标和先验框的wh
    all_anchors_xywh = generate_all_anchors(anchors, H, W) # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
    #deltas_pred是预测框的坐标，deltas_pred.new(*all_anchors_xywh.size())创建一个与deltas_pred数据类型相同的张量
    # *all_anchors_xywh.size()展开了all_anchors_xywh的形状，*解包操作，将形状元组的每个维度一次作为参数传递
    #copy_(all_anchors_xywh)将all_anchors_xywh 中的数据复制到 deltas_pred.new() 创建的新张量中
    #返回的是先验框的信息
    all_anchors_xywh = deltas_pred.new(*all_anchors_xywh.size()).copy_(all_anchors_xywh)#确保all_anchors_xywh和deltas_pred类型和设备一致
    #下面将偏移量，和先验框的坐标送入函数得到最终的预测框
    boxes_pred = box_transform_inv(all_anchors_xywh, deltas_pred)

    return boxes_pred


def scale_boxes(boxes, H,W):
    """
    scale predicted boxes

    Arguments:
    boxes -- tensor of shape (N, 4) xxyy format
    im_info -- dictionary {width:, height:}

    Returns:
    scaled_boxes -- tensor of shape (N, 4) xxyy format

    """

    h = H
    w = W

    input_h, input_w = cfg.test_input_size
    scale_h, scale_w = input_h / h, input_w / w

    # scale the boxes
    boxes *= cfg.strides

    boxes[:, 0::2] /= scale_w
    boxes[:, 1::2] /= scale_h
    #从xywh变成左上角坐标加右下角坐标
    boxes = xywh2xxyy(boxes)

    # clamp boxes，限制在预测框以内，以外的点裁掉
    boxes[:, 0::2].clamp_(0, w-1)
    boxes[:, 1::2].clamp_(0, h-1)

    return boxes


def yolo_eval(yolo_output,H,W, conf_threshold=0.6, nms_threshold=0.4):
    """
    Evaluate the yolo output, generate the final predicted boxes
    Arguments:
    yolo_output -- list of tensors (deltas_pred, conf_pred, classes_pred)
    deltas_pred -- tensor of shape (H * W * num_anchors, 4) σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred -- tensor of shape (H * W * num_anchors, 1)
    classes_pred -- tensor of shape (H * W * num_anchors, num_classes)
    im_info -- dictionary {w:, h:}
    threshold -- float, threshold used to filter boxes
    Returns:
    detections -- tensor of shape (None, 7) (x1, y1, x2, y2, cls_conf, cls)
    """
    deltas = yolo_output[0].cpu() #预测框的信息
    conf = yolo_output[1].cpu() #置信度
    classes = yolo_output[2].cpu()#类别概率

    num_classes = classes.size(1)
    # apply deltas to anchors
    #boxes输出的是最终的预测框，因为经过模型得到的是偏移量，要根据先验框去得到真正的预测框，这一过程包括
    boxes = generate_prediction_boxes(deltas)

    if cfg.debug:
        print('check box: ', boxes.view(13*13, 5, 4).permute(1, 0, 2).contiguous().view(-1,4)[0:10,:])
        print('check conf: ', conf.view(13*13, 5).permute(1,0).contiguous().view(-1)[:10])

    #filter boxes on confidence score，利用置信度去筛选一下预测框，输入是预测框，置信度，类别概率，以及置信度的阈值
    #返回的四个值代表，筛选后综合置信度大于阈值的预测框，置信度的值，以及预测框对应的最大类别的概率和索引（去找具体类别）
    #也就是筛选过后的框，以及框对应的置信度，以及框对应的最大类别的概率值，和类别标签（数字1234....）
    boxes, conf, cls_max_conf, cls_max_id = yolo_filter_boxes(boxes, conf, classes, conf_threshold)

    # no detection !
    if boxes.size(0) == 0:
        return []

    # scale boxes 输入是筛选过的预测框和原始图片的高宽，因为yolov2图片大小的输入是416*416，输出是将预测框调整为原始图像的预测框大小
    #同时将boxes里面的值从(x,y,w,h)改为(x1,y1,x2,y2)左上角和右下角的坐标
    boxes = scale_boxes(boxes,H,W)

    if cfg.debug:
        all_boxes = torch.cat([boxes, conf, cls_max_conf, cls_max_id], dim=1)
        print('check all boxes: ', all_boxes)
        print('check all boxes len: ', len(all_boxes))
    #
    # apply nms
    # keep = yolo_nms(boxes, conf.view(-1), nms_threshold)
    # boxes_keep = boxes[keep, :]
    # conf_keep = conf[keep, :]
    # cls_max_conf = cls_max_conf[keep, :]
    # cls_max_id = cls_max_id.view(-1, 1)[keep, :]
    #
    # if cfg.debug:
    #     print('check nms all boxes len: ', len(boxes_keep))
    #
    # seq = [boxes_keep, conf_keep, cls_max_conf, cls_max_id.float()]
    #
    # return torch.cat(seq, dim=1)
    #存储所有类别经过nms筛选后的所有结果
    detections = []

    cls_max_id = cls_max_id.view(-1)

    # apply NMS classwise
    for cls in range(num_classes):#遍历每一个类别
        cls_mask = cls_max_id == cls #布尔值判断预测值中是否存在当前遍历的类别
        inds = torch.nonzero(cls_mask).squeeze()#根据cls_mask的值去得到tensor

        if inds.numel() == 0:
            continue
        #这里是确定，调整预测框，置信度，分类概率，分类类别的形状为两个维度
        boxes_pred_class = boxes[inds, :].view(-1, 4)#view改变张量形状，-1是指自动计算该维度，4表示列元素有四个
        conf_pred_class = conf[inds, :].view(-1, 1)
        cls_max_conf_class = cls_max_conf[inds].view(-1, 1)
        classes_class = cls_max_id[inds].view(-1, 1)
        #nms根据iou和置信度进行筛选，返回的nms_keep是筛选出来的框的索引
        nms_keep = yolo_nms(boxes_pred_class, conf_pred_class.view(-1), nms_threshold)

        boxes_pred_class_keep = boxes_pred_class[nms_keep, :]
        conf_pred_class_keep = conf_pred_class[nms_keep, :]
        cls_max_conf_class_keep = cls_max_conf_class.view(-1, 1)[nms_keep, :]
        classes_class_keep = classes_class.view(-1, 1)[nms_keep, :]

        seq = [boxes_pred_class_keep, conf_pred_class_keep, cls_max_conf_class_keep, classes_class_keep.float()]
        #将这四个值弄成一个一行的tensor变量，坐标，置信度，类别概率，类别索引
        detections_cls = torch.cat(seq, dim=-1)
        detections.append(detections_cls)

    return torch.cat(detections, dim=0)
