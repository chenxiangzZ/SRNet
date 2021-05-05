import torch
from .models import mining_hard_pairs, analyze_ver_prob
from tools import *


# 这里针对某个epoch进行训练
def train_an_epoch(config, base, loaders, epoch):
    # 设为训练模式，BN和Dropout层的更新和trian不一致
    base.set_train()
    # 计量器
    meter = MultiItemAverageMeter()

    ### we assume 200 iterations as an epoch
    for _ in range(200):
        ### load a batch data
        imgs, pids, _ = loaders.train_iter.next_one()
        imgs, pids = imgs.to(base.device), pids.to(base.device)
        ### forward
        lf_xent, local_feat, feature_info, cls_score_info, ver_probs, gmp_info, gmn_info, keypoints_confidence = base.forward(imgs, pids, training=True)

        feature_vector_list = feature_info
        cls_score_list = cls_score_info
        ver_prob_p, ver_prob_n = ver_probs
        s_p, emb_p, emb_pp = gmp_info
        s_n, emb_n, emb_nn = gmn_info
        ### backnbone的loss
        ce_backbone_loss = base.ce_loss(lf_xent, pids)
        triplet_backbone_loss = base.triplet(local_feat, pids) * 0.4
        ### S模块的loss
        ide_loss = base.compute_ide_loss(cls_score_list, pids, keypoints_confidence)
        # triplet_loss = base.compute_triplet_loss(feature_vector_list, pids)
        ### R模块的gcn loss
        # gcned_ide_loss = base.compute_ide_loss(gcned_cls_score_list, pids, keypoints_confidence)
        # gcned_triplet_loss = base.compute_triplet_loss(gcned_feature_vector_list, pids)
        ### graph matching loss  正样本之间的全排列损失+output_senet_weight结合
        s_gt = torch.eye(14).unsqueeze(0).repeat([s_p.shape[0], 1, 1]).detach().to(base.device)
        pp_loss = base.permutation_loss(s_p, s_gt)
        pn_loss = base.permutation_loss(s_n, s_gt)
        p_loss = pp_loss # + pn_loss
        ### verification loss 交叉熵损失BCE (input,taget),正样本的话targets都为1
        ver_loss = base.bce_loss(ver_prob_p, torch.ones_like(ver_prob_p)) + base.bce_loss(ver_prob_n, torch.zeros_like(ver_prob_n))

        # overall loss
        loss = ce_backbone_loss + triplet_backbone_loss + ide_loss # + triplet_loss
        # 20个epoch后计算gm损失weight_p_loss:1.0  weight_ver_loss:0.1
        if epoch >= config.use_gm_after:
            loss += \
               config.weight_p_loss * p_loss + \
               config.weight_ver_loss * ver_loss
        acc = base.compute_accuracy(cls_score_list, pids)
        # gcned_acc = base.compute_accuracy(gcned_cls_score_list, pids)
        ver_p_ana = analyze_ver_prob(ver_prob_p, True)
        ver_n_ana = analyze_ver_prob(ver_prob_n, False)

        ### optimize
        base.optimizer.zero_grad()
        loss.backward()
        base.optimizer.step()

        ### recored
        meter.update({'ce_backbone_loss': ce_backbone_loss.data.cpu().numpy(), 'triplet_backbone_loss': triplet_backbone_loss.data.cpu().numpy(),
                      'ide_loss': ide_loss.data.cpu().numpy(),
                      # 'triplet_loss': triplet_loss.data.cpu().numpy(),
                      'acc': acc,
                      'ver_loss': ver_loss.data.cpu().numpy(), 'ver_p_ana': torch.tensor(ver_p_ana).data.cpu().numpy(), 'ver_n_ana': torch.tensor(ver_n_ana).data.cpu().numpy(),
                      'pp_loss': pp_loss.data.cpu().numpy(), 'pn_loss': pn_loss.data.cpu().numpy()})

    return meter.get_val(), meter.get_str()



