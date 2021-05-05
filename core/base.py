import sys

sys.path.append('..')
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
from bisect import bisect_right
import os

from losses import TripletLoss as TripletLoss1
from losses import CrossEntropyLabelSmooth as CrossEntropyLabelSmooth1
from .models import Encoder, BNClassifiers
from .models import ScoremapComputer, compute_local_features
from .models import GraphConvNet, generate_adj
from .models import GMNet, PermutationLoss, Verificator, mining_hard_pairs
from tools import CrossEntropyLabelSmooth, TripletLoss, accuracy
from .models.utils import *
from .models.se_module import SELayer, SELayer2GM, MultiScaleLayer_v2, SELayer_Local, BasicConv2d

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


class Base:

    def __init__(self, config, loaders):

        self.config = config
        self.loaders = loaders
        self.pid_num = config.pid_num
        # 702 DukeMTMC-reID
        if config.train_dataset == 'duck':
            self.pid_num = 702
        elif config.train_dataset == 'market':
            self.pid_num = 751
        elif config.train_dataset == "occluded_reid":
            self.pid_num = 200

        # margin for the triplet loss with batch hard 0.3
        self.margin = config.margin
        # 14 关键点数
        self.branch_num = config.branch_num

        # Logger Configuration
        # 1 即最多保存几个模型
        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')
        self.save_visualize_market_path = os.path.join(self.output_path, 'visualization/market/')
        self.save_visualize_duke_path = os.path.join(self.output_path, 'visualization/duke/')
        
        self.ce_loss = CrossEntropyLabelSmooth1(self.pid_num)
        #else:
        #    self.ce_loss = nn.CrossEntropyLoss()
        self.triplet = TripletLoss1(0.3)

        # Train Configuration
        # 0.00035
        self.base_learning_rate = config.base_learning_rate
        # 0.0005
        self.weight_decay = config.weight_decay
        # 转折点 [40, 70]
        self.milestones = config.milestones

        # init model
        self._init_device()
        self._init_model()
        self._init_creiteron()
        self._init_optimizer()

    def _init_device(self):
        self.device = torch.device('cuda')

    def _init_model(self):

        # feature learning
        # 修改后的ResNet50
        self.encoder = Encoder(class_num=self.pid_num)
        self.bnclassifiers = BNClassifiers(2048, self.pid_num, self.branch_num)
        # self.bnclassifiers2 = BNClassifiers(2048, self.pid_num, self.branch_num)  # for gcned features

        # 设置多GPU的训练
        self.encoder = nn.DataParallel(self.encoder).to(self.device)
        self.bnclassifiers = nn.DataParallel(self.bnclassifiers).to(self.device)
        # self.bnclassifiers2 = nn.DataParallel(self.bnclassifiers2).to(self.device)

        # keypoints model norm_scale=10
        self.scoremap_computer = ScoremapComputer(self.config.norm_scale).to(self.device)
        # self.scoremap_computer = nn.DataParallel(self.scoremap_computer).to(self.device)
        self.scoremap_computer = self.scoremap_computer.eval()
        # summary(self.scoremap_computer,(3,256,128),32,)

        # GCN
        self.linked_edges = \
            [[13, 0], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10],
             [13, 11], [13, 12],  # global
             [0, 1], [0, 2],  # head
             [1, 2], [1, 7], [2, 8], [7, 8], [1, 8], [2, 7],  # body
             [1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12],  # libs
             # [3,4],[5,6],[9,10],[11,12], # semmetric libs links
             ]
        self.adj = generate_adj(self.branch_num, self.linked_edges, self_connect=0.0).to(self.device)
        # gcn_scale 20.0
        # 里面堆叠了两个ADGC层，输出的维度都是一致的
        # self.gcn = GraphConvNet(self.adj, 2048, 2048, 2048, self.config.gcn_scale).to(self.device)

        self.senet = SELayer2GM(14).to(self.device)

        # graph matching
        self.gmnet = GMNet().to(self.device)

        # verification
        # 下面这句相当于论文中的公式10
        self.verificator = Verificator(self.config).to(self.device)

    def _init_creiteron(self):
        self.ide_creiteron = CrossEntropyLabelSmooth(self.pid_num, reduce=False)
        self.triplet_creiteron = TripletLoss(self.margin, 'euclidean')
        self.bce_loss = nn.BCELoss()
        self.permutation_loss = PermutationLoss()

    def compute_ide_loss(self, score_list, pids, weights):
        loss_all = 0
        for i, score_i in enumerate(score_list):
            loss_i = self.ide_creiteron(score_i, pids)
            loss_i = (weights[:, i] * loss_i).mean()
            loss_all += loss_i
        return loss_all

    def compute_triplet_loss(self, feature_list, pids):
        '''we suppose the last feature is global, and only compute its loss'''
        loss_all = 0
        for i, feature_i in enumerate(feature_list):
            if i == len(feature_list) - 1:
                loss_i = self.triplet_creiteron(feature_i, feature_i, feature_i, pids, pids, pids)
                loss_all += loss_i
        return loss_all

    def compute_accuracy(self, cls_score_list, pids):
        overall_cls_score = 0
        for cls_score in cls_score_list:
            overall_cls_score += cls_score
        acc = accuracy(overall_cls_score, pids, [1])[0]
        return acc

    def _init_optimizer(self):
        params = []

        for key, value in self.senet.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.base_learning_rate
            weight_decay = self.weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        for key, value in self.encoder.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.base_learning_rate
            weight_decay = self.weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        for key, value in self.bnclassifiers.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.base_learning_rate
            weight_decay = self.weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        # for key, value in self.bnclassifiers2.named_parameters():
        #     if not value.requires_grad:
        #         continue
        #     lr = self.base_learning_rate
        #     weight_decay = self.weight_decay
        #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        # for key, value in self.gcn.named_parameters():
        #     if not value.requires_grad:
        #         continue
        #     lr = self.base_learning_rate
        #     weight_decay = self.weight_decay
        #     params += [{"params": [value], "lr": self.config.gcn_lr_scale * lr, "weight_decay": weight_decay}]

        for key, value in self.gmnet.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.base_learning_rate
            weight_decay = self.weight_decay
            params += [{"params": [value], "lr": self.config.gm_lr_scale * lr, "weight_decay": weight_decay}]

        for key, value in self.verificator.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.base_learning_rate
            weight_decay = self.weight_decay
            params += [{"params": [value], "lr": self.config.ver_lr_scale * lr, "weight_decay": weight_decay}]

        self.optimizer = optim.Adam(params)
        self.lr_scheduler = WarmupMultiStepLR(
            self.optimizer, self.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)

    ## save model as save_epoch
    def save_model(self, save_epoch):

        torch.save(self.encoder.state_dict(), os.path.join(self.save_model_path, 'encoder_{}.pkl'.format(save_epoch)))
        torch.save(self.bnclassifiers.state_dict(),
                   os.path.join(self.save_model_path, 'bnclassifiers_{}.pkl'.format(save_epoch)))
        # torch.save(self.bnclassifiers2.state_dict(),
        #            os.path.join(self.save_model_path, 'bnclassifiers2_{}.pkl'.format(save_epoch)))
        # # torch.save(self.gcn.state_dict(), os.path.join(self.save_model_path, 'gcn_{}.pkl'.format(save_epoch)))
        torch.save(self.gmnet.state_dict(), os.path.join(self.save_model_path, 'gmnet_{}.pkl'.format(save_epoch)))
        torch.save(self.verificator.state_dict(),
                   os.path.join(self.save_model_path, 'verificator_{}.pkl'.format(save_epoch)))
        torch.save(self.senet.state_dict(),
                   os.path.join(self.save_model_path, 'senet_{}.pkl'.format(save_epoch)))
        # if saved model is more than max num, delete the model with smallest iter
        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            new_file = []
            for file_ in files:
                if file_.endswith('.pkl'):
                    new_file.append(file_)
            file_iters = sorted(list(set([int(file.replace('.', '_').split('_')[-2]) for file in new_file])),
                                reverse=False)

            if len(file_iters) > self.max_save_model_num:
                for i in range(len(file_iters) - self.max_save_model_num):
                    file_path = os.path.join(root, '*_{}.pkl'.format(file_iters[i]))
                    print('remove files:', file_path)
                    os.system('rm -rf {}'.format(file_path))

    ## resume model from resume_epoch
    def resume_model(self, resume_epoch):
        self.encoder.load_state_dict(
            torch.load(os.path.join(self.save_model_path, 'encoder_{}.pkl'.format(resume_epoch))))
        self.bnclassifiers.load_state_dict(
            torch.load(os.path.join(self.save_model_path, 'bnclassifiers_{}.pkl'.format(resume_epoch))))
        # self.bnclassifiers2.load_state_dict(
        #     torch.load(os.path.join(self.save_model_path, 'bnclassifiers2_{}.pkl'.format(resume_epoch))))
        # # self.gcn.load_state_dict(torch.load(
        # #     os.path.join(self.save_model_path, 'gcn_{}.pkl'.format(resume_epoch))))
        self.gmnet.load_state_dict(
            torch.load(os.path.join(self.save_model_path, 'gmnet_{}.pkl'.format(resume_epoch))))
        self.verificator.load_state_dict(
            torch.load(os.path.join(self.save_model_path, 'verificator_{}.pkl'.format(resume_epoch))))
        self.senet.load_state_dict(
            torch.load(os.path.join(self.save_model_path, 'senet_{}.pkl'.format(resume_epoch))))

    ## resume model from resume_epoch
    def resume_model_from_path(self, path, resume_epoch):
        self.encoder.load_state_dict(
            torch.load(os.path.join(path, 'encoder_{}.pkl'.format(resume_epoch))))
        self.bnclassifiers.load_state_dict(
            torch.load(os.path.join(path, 'bnclassifiers_{}.pkl'.format(resume_epoch))))
        # self.bnclassifiers2.load_state_dict(
        #     torch.load(os.path.join(path, 'bnclassifiers2_{}.pkl'.format(resume_epoch))))
        # # self.gcn.load_state_dict(torch.load(
        # #     os.path.join(path, 'gcn_{}.pkl'.format(resume_epoch))))
        self.gmnet.load_state_dict(torch.load(
            os.path.join(path, 'gmnet_{}.pkl'.format(resume_epoch))))
        self.verificator.load_state_dict(
            torch.load(os.path.join(path, 'verificator_{}.pkl'.format(resume_epoch))))
        self.senet.load_state_dict(
            torch.load(os.path.join(path, 'senet_{}.pkl'.format(resume_epoch))))

    ## set model as train mode
    def set_train(self):
        self.encoder = self.encoder.train()
        self.bnclassifiers = self.bnclassifiers.train()
        # self.bnclassifiers2 = self.bnclassifiers2.train()
        # # self.gcn = self.gcn.train()
        self.gmnet = self.gmnet.train()
        self.verificator = self.verificator.train()
        self.senet = self.senet.train()

    ## set model as eval mode
    def set_eval(self):
        self.encoder = self.encoder.eval()
        self.bnclassifiers = self.bnclassifiers.eval()
        # self.bnclassifiers2 = self.bnclassifiers2.eval()
        # # self.gcn = self.gcn.eval()
        self.gmnet = self.gmnet.eval()
        self.verificator = self.verificator.eval()
        self.senet = self.senet.eval()

    def forward(self, images, pids, training):

        #  backnone提取的global feature输入（128,3,256,128) 输出（128,2048,16,8）
        outputs = self.encoder(images)
        # lf_xent 计算CELoss local_feat 计算三元组Loss
        lf_xent, local_feat = [], []
        if training:
            feature_maps, lf_xent, local_feat = outputs
        else:
            # up down global 拼接成一个feature
            feature_maps, local_feat = outputs



        # HRNet提取的14个关键点和对应的置信度（经过norm）,输入128张图片
        with torch.no_grad():
            # scoremape(128,13,16,8)
            # keypoints_confidence(128,13)
            score_maps, keypoints_confidence, _ = self.scoremap_computer(images)
        # 得到 13个local特征和1个全局特征以及对应的置信度   14个(128,2048)  128个14
        feature_vector_list, keypoints_confidence = compute_local_features(
            self.config, feature_maps, score_maps, keypoints_confidence)

        # 经过14个分类器后的输出（特征，分类结果）
        # bned_feature_vector_list 14个(128,2048)
        # cls_score_list 14个(128,751)
        # bned_feature_vector_list, cls_score_list = self.bnclassifiers(feature_vector_list)

        # gcn
        # 里面堆叠了两个ADGC层，输出的维度都是一致的
        # gcned_feature_vector_list = self.gcn(feature_vector_list)

        # bned_gcned_feature_vector_list, gcned_cls_score_list = self.bnclassifiers2(gcned_feature_vector_list)

        if training:

            # mining hard samples 难样本挖掘
            # 获得 bned_gcned_feature_vector_list 和正负样本的list
            # new_bned_gcned_feature_vector_list, bned_gcned_feature_vector_list_p, bned_gcned_feature_vector_list_n = mining_hard_pairs(
            #   bned_gcned_feature_vector_list, pids)

            # 进入到SENet中，进行权值的再分配
            # torch.Size([128, 10, 2048])
            input_senet_feature = torch.cat([i.unsqueeze(1) for i in feature_vector_list], dim=1)
            # 信道数为分支数14 (128,14)
            output_senet_feature = self.senet(input_senet_feature)
            # 转换成list
            bned_feature_vector_list = []
            for i in range(14):
                bned_feature_vector_list.append(output_senet_feature[:, i, :].squeeze())

            # 经过14个分类器后的输出（特征，分类结果）
            # bned_feature_vector_list 14个(128,2048)
            # cls_score_list 14个(128,751)
            bned_feature_vector_list, cls_score_list = self.bnclassifiers(bned_feature_vector_list)
            # 换成gcn前的特征

            new_bned_feature_vector_list, bned_feature_vector_list_p, bned_feature_vector_list_n = mining_hard_pairs(
                bned_feature_vector_list, pids)

            # graph matching
            # new_bned_feature_vector_list 14个(128,2048)
            # s_p 相似度矩阵  emb_p, emb_pp 对应于输入的两个list在T模块的输出
            s_p, emb_p, emb_pp = self.gmnet(new_bned_feature_vector_list, bned_feature_vector_list_p, None)
            s_n, emb_n, emb_nn = self.gmnet(new_bned_feature_vector_list, bned_feature_vector_list_n, None)

            # verificate
            # ver_prob_p = self.verificator(bned_gcned_feature_vector_list, bned_gcned_feature_vector_list_p)
            # ver_prob_n = self.verificator(bned_gcned_feature_vector_list, bned_gcned_feature_vector_list_n)
            # ver_prob_p即论文中的Sx1,x2,即样本x1与x2之间的相似度，即当前batch内所有的
            ver_prob_p = self.verificator(emb_p, emb_pp)
            ver_prob_n = self.verificator(emb_n, emb_nn)

            return lf_xent, local_feat, \
                   feature_vector_list,  \
                   cls_score_list, \
                   (ver_prob_p, ver_prob_n), \
                   (s_p, emb_p, emb_pp), \
                   (s_n, emb_n, emb_nn), \
                   keypoints_confidence
        else:
            bs, keypoints_num = keypoints_confidence.shape
            keypoints_confidence = torch.sqrt(keypoints_confidence).unsqueeze(2).repeat([1, 1, 2048]).view(
                [bs, 2048 * keypoints_num])

            # features = keypoints_confidence * torch.cat(feature_vector_list, dim=1)
            # bned_features = keypoints_confidence * torch.cat(bned_feature_vector_list, dim=1)
            # gcned_features = keypoints_confidence * torch.cat(gcned_feature_vector_list, dim=1)
            # bned_gcned_features = keypoints_confidence * torch.cat(bned_gcned_feature_vector_list, dim=1)
            # return (features, bned_features), (gcned_features, bned_gcned_features)

            # features = torch.cat([i.unsqueeze(1) for i in feature_vector_list], dim=1)
            # bned_features = keypoints_confidence * torch.cat(bned_feature_vector_list, dim=1)
            # gcned_features = torch.cat([i.unsqueeze(1) for i in gcned_feature_vector_list], dim=1)
            # bned_gcned_features = keypoints_confidence * torch.cat(bned_gcned_feature_vector_list, dim=1)
            # return (features, bned_features), (gcned_features, bned_gcned_features)

            # 进入到SENet中，进行权值的再分配
            # torch.Size([128, 10, 2048])
            input_senet_feature = torch.cat([i.unsqueeze(1) for i in feature_vector_list], dim=1)
            # 信道数为分支数14 (128,14)
            output_senet_feature = self.senet(input_senet_feature)
            # 转换成list
            bned_feature_vector_list = []
            for i in range(14):
                bned_feature_vector_list.append(output_senet_feature[:, i, :].squeeze())

            # 经过14个分类器后的输出（特征，分类结果）
            # bned_feature_vector_list 14个(128,2048)
            # cls_score_list 14个(128,751)
            bned_feature_vector_list, cls_score_list = self.bnclassifiers(bned_feature_vector_list)

            features_stage1 = keypoints_confidence * torch.cat(bned_feature_vector_list, dim=1)
            features_satge2 = torch.cat([i.unsqueeze(1) for i in bned_feature_vector_list], dim=1)
            # gcned_features_stage1 = keypoints_confidence * torch.cat(bned_gcned_feature_vector_list, dim=1)
            # gcned_features_stage2 = torch.cat([i.unsqueeze(1) for i in bned_gcned_feature_vector_list], dim=1)

            return local_feat, (features_stage1, features_satge2)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
