import torch
import torch.nn as nn
import torchvision
from yacs.config import CfgNode as CN

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Encoder(nn.Module):

    def __init__(self, class_num):
        super(Encoder, self).__init__()

        self.class_num = class_num

        # backbone and optimize its architecture
        # resnet = torchvision.models.resnet50(pretrained=True)
        # resnet.layer4[0].conv2.stride = (1, 1)
        # resnet.layer4[0].downsample[0].stride = (1, 1)


        # cnn backbone
        # self.resnet_conv = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.maxpool,  # no relu
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        # self.gap = nn.AdaptiveAvgPool2d(1)

        self.backbone = self.build_model(self.class_num)

    def build_model(cfg, num_classes) -> nn.Module:
        # print('cfg.TEST.WEIGHT', cfg.TEST.WEIGHT)
        # if len(cfg.TEST.WEIGHT) > 0:
        #     print('>>>>>>>>>>>>>Load model with from pre-trained model<<<<<<<<<<<<<<<')
        #     print('>>>>>>>>>>>>>Only used in finetune or inference<<<<<<<<<<<<<<<')
        #     # from .baseline_parts_old_ft import Baseline
        #     from .baseline_parts_ft import Baseline
        # else:
            # from .baseline_parts_old import Baseline
        from .baseline_parts import Baseline
        print('>>>>>>>>>>>>>Load model with imagenet pre-trained<<<<<<<<<<<<<<<')

        model = Baseline(
            backbone='resnet50',
            num_classes=num_classes,
            last_stride=1,
            with_ibn=True,
            gcb=CN(),
            stage_with_gcb=(False, False, False, False),
            use_parts=2,
            pretrain=True,
            model_path='/content/r50_ibn_a.pth')
        return model

    def forward(self, x):
        outputs = self.backbone(x)
        return outputs


class BNClassifier(nn.Module):
    """
    BNClassifier(
  (bn): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (classifier): Linear(in_features=2048, out_features=751, bias=False)
)
    """

    # 分类器，每个branch都有一个
    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score


class BNClassifiers(nn.Module):

    def __init__(self, in_dim, class_num, branch_num):
        super(BNClassifiers, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num
        self.branch_num = branch_num

        for i in range(self.branch_num):
            # 给对象的属性设值，classifier_1: feature, cls_score
            setattr(self, 'classifier_{}'.format(i), BNClassifier(self.in_dim, self.class_num))

    def __call__(self, feature_vector_list):
        # 14个(128,2048)
        assert len(feature_vector_list) == self.branch_num

        # bnneck for each sub_branch_feature
        bned_feature_vector_list, cls_score_list = [], []
        for i in range(self.branch_num):
            feature_vector_i = feature_vector_list[i]
            # 根据不同的分类器名，找到对应的对象
            classifier_i = getattr(self, 'classifier_{}'.format(i))

            bned_feature_vector_i, cls_score_i = classifier_i(feature_vector_i)

            bned_feature_vector_list.append(bned_feature_vector_i)
            cls_score_list.append(cls_score_i)

        return bned_feature_vector_list, cls_score_list
