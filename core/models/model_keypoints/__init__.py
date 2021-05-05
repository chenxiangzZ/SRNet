import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .config import cfg as pose_config
from .pose_hrnet import get_pose_net
from .pose_processor import HeatmapProcessor2


class ScoremapComputer(nn.Module):

    #  norm_scale 10.0
    def __init__(self, norm_scale):
        super(ScoremapComputer, self).__init__()

        # init skeleton model
        # 文中的HRNet,在COCO数据集上进行了预训练
        self.keypoints_predictor = get_pose_net(pose_config, False)
        # 加载pose_hrnet_w48_256x192.pth的权重
        self.keypoints_predictor.load_state_dict(torch.load(pose_config.TEST.MODEL_FILE))
        # self.heatmap_processor = HeatmapProcessor(normalize_heatmap=True, group_mode='sum', gaussion_smooth=None)
        self.heatmap_processor = HeatmapProcessor2(normalize_heatmap=True, group_mode='sum', norm_scale=norm_scale)

    def forward(self, x):
        # 输入（128, 3,256,128）输出（128,17,64,32）
        heatmap = self.keypoints_predictor(x)  # before normalization
        # print(heatmap.shape)
        # scoremape(128,13,6,8)
        # keypoints_confidence(128,13)
        # keypoints_location(128,17,2)
        scoremap, keypoints_confidence, keypoints_location = self.heatmap_processor(heatmap)  # after normalization

        return scoremap.detach(), keypoints_confidence.detach(), keypoints_location.detach()


def compute_local_features(config, feature_maps, score_maps, keypoints_confidence):
    '''
    the last one is global feature
    :param config:
    :param feature_maps:(128,2048,16,8）
    :param score_maps:(128,13,16,8)
    :param keypoints_confidence:
    :return:
    '''
    fbs, fc, fh, fw = feature_maps.shape
    sbs, sc, sh, sw = score_maps.shape
    assert fbs == sbs and fh == sh and fw == sw

    # get feature_vector_list
    feature_vector_list = []
    for i in range(sc + 1):
        if i < sc:  # skeleton-based local feature vectors
            # unsqueeeze增加维度，repeat使关键点特征的Chanel与feature_map一致 2048维好像
            score_map_i = score_maps[:, i, :, :].unsqueeze(1).repeat([1, fc, 1, 1])
            # 每个关键点特征与global相乘
            feature_vector_i = torch.sum(score_map_i * feature_maps, [2, 3])
            feature_vector_list.append(feature_vector_i)
        else:  # global feature vectors
            feature_vector_i = (
                # 自适应平均和最大池化，输出的大小为1*1
                        F.adaptive_avg_pool2d(feature_maps, 1) + F.adaptive_max_pool2d(feature_maps, 1)).squeeze()
            feature_vector_list.append(feature_vector_i)
            keypoints_confidence = torch.cat([keypoints_confidence, torch.ones([fbs, 1]).cuda()], dim=1)

    # compute keypoints confidence weight_global_feature=1.0
    # 最后一个置信度
    keypoints_confidence[:, sc:] = F.normalize(
        keypoints_confidence[:, sc:], 1, 1) * config.weight_global_feature  # global feature score_confidence
    # 前13个置信度
    keypoints_confidence[:, :sc] = F.normalize(
        keypoints_confidence[:, :sc], 1, 1) * config.weight_global_feature  # partial feature score_confidence

    return feature_vector_list, keypoints_confidence
