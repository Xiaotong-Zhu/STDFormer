import torch
import torch.nn as nn

from src.models.ops.mlp import MLP
from src.models.ops.snce import SocialNCE
from src.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import torch.nn.functional as F


def _focal_loss(conf, conf_gt, mask, alpha=0.25, gamma=2.0, pos_weight=1.0, neg_weight=1.0):
    """ Focal Loss with 0 / 1 confidence as gt.
    Args:
        conf (torch.Tensor): (N, L, S)
        conf_gt (torch.Tensor): (N, L, S)
        mask: (N,L,S) , bool, valid==True
        
    """
    pos_mask = (conf_gt == 1) * mask # 有效区内并且gt为1是正样本
    # neg_mask = (conf_gt == 0) * mask # 有效区内并且gt为0是负样本

    conf = torch.clamp(conf, 1e-6, 1-1e-6)
    
    pos_conf = conf[pos_mask]
    loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()  
    loss =  pos_weight * loss_pos.mean()
    return loss

def _ce_loss(conf, conf_gt):
    """ 参考snce
    Args:
        conf (torch.Tensor): (N, L, S)
        conf_gt (torch.Tensor): (N, L, S)
        mask: (N,L,S) , bool, valid==True
        
    """
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    batch, query_num, key_num = conf.shape
    conf = conf.reshape((batch*query_num, key_num))
    conf_gt = conf_gt.reshape((batch*query_num))
    loss = criterion(conf, conf_gt)

    return loss      

def _reg_loss(regr, gt_regr, mask, regr_weight=5.0, giou_weight=2.0):
    ''' L1 regression loss
    Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects) ,targets['box_reg_region'] ,mask掉为了并行训练添加的无用结果，有效区为1，无效为0
    '''
    num = mask.float().sum() # 有效预测个数
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    # regr_loss = F.l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-9)

    batch = regr.shape[0]
    # regr = regr.chunk(batch,dim=0)
    # gt_regr = gt_regr.chunk(batch,dim=0)

    giou_loss = torch.tensor([]).double().cuda()
    for b in range(batch):
        giou_loss = torch.cat((giou_loss, torch.tensor([((1-torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(regr[b,:,:]), box_cxcywh_to_xyxy(gt_regr[b,:,:])))).masked_fill(~(mask[b,:,0].bool()), 0.0)).sum()]).cuda())) # 1-giou
    giou_loss = giou_loss.sum() / (num + 1e-9)

    loss = regr_weight * regr_loss + giou_weight * giou_loss
    # loss = giou_loss
    # loss = regr_loss

    return loss

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()

    def forward(self, pred, target, mask, alpha, gamma, pos_weight, neg_weight):
        loss =  _focal_loss(pred, target, mask, alpha, gamma, pos_weight, neg_weight)
        # loss =  _ce_loss(pred, target)
        return loss

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
    
    def forward(self, pred, target, mask, regr_weight, giou_weight):
        loss = _reg_loss(pred, target, mask, regr_weight, giou_weight)
        return loss

class STDLoss(nn.Module):
    def __init__(self, cfg):
        super(STDLoss, self).__init__()
        self.mot_loss = IDLoss()
        self.tp_loss = RegLoss()

        # focal loss
        self.focal_alpha = cfg.LOSS.FOCAL_ALPHA
        self.focal_gamma = cfg.LOSS.FOCAL_GAMMA
        self.focal_pos_weight = cfg.LOSS.FOCAL_POS_WEIGHT
        self.focal_neg_weight = cfg.LOSS.FOCAL_NEG_WEIGHT

        # reg
        self.regr_weight = cfg.LOSS.REGR_WEIGHT
        self.giou_weight = cfg.LOSS.GIOU_WEIGHT


        # snce
        self.snce = SocialNCE(temperature=cfg.SOCIAL_NCE.TEMPERATURE)
        self.snce_weight = cfg.LOSS.SOCIAL_NCE_WEIGHT

        self.motloss_weight = cfg.LOSS.MOTLOSS_WEIGHT
        self.tploss_weight = cfg.LOSS.TPLOSS_WEIGHT

        # task & snce
        self.social_nce_track_loss_weight = nn.Parameter(-1.05 * torch.ones(1), requires_grad=True)
        self.task_loss_weight = nn.Parameter(-1.85 * torch.ones(1), requires_grad = True)
  
    # def forward(self, conf_matrix, tp, track_enc_emb, current_track_emb, temp_token_mask, current_track_mask, det_tp, det_gt, det_mask, targets):
    def forward(self, conf_matrix, tp, track_enc_emb, current_track_emb, temp_token_mask, current_track_mask, targets):
        # cur_boxes = torch.zeros(tp.shape).double().cuda()
        # cur_boxes[:,:,0] = tp[:,:,0] * targets['img_size'][:, None, 0]
        # cur_boxes[:,:,1] = tp[:,:,1] * targets['img_size'][:, None, 1]
        # cur_boxes[:,:,2] = tp[:,:,2] * targets['img_size'][:, None, 0]
        # cur_boxes[:,:,3] = tp[:,:,3] * targets['img_size'][:, None, 1]

        # cur_boxes_gt = torch.zeros(tp.shape).double().cuda()
        # cur_boxes_gt[:,:,0] = targets['box_reg'][:,:,0] * targets['img_size'][:, None, 0]
        # cur_boxes_gt[:,:,1] = targets['box_reg'][:,:,1] * targets['img_size'][:, None, 1]
        # cur_boxes_gt[:,:,2] = targets['box_reg'][:,:,2] * targets['img_size'][:, None, 0]
        # cur_boxes_gt[:,:,3] = targets['box_reg'][:,:,3] * targets['img_size'][:, None, 1]

        loss_mot = self.mot_loss(conf_matrix, targets['id_cls'], targets['id_cls_mask'].bool(), self.focal_alpha, self.focal_gamma, self.focal_pos_weight, self.focal_neg_weight)
        loss_tp = self.tp_loss(tp, targets['box_reg'], targets['box_reg_mask'], self.regr_weight, self.giou_weight)
        # loss_det_tp = self.tp_loss(det_tp, det_gt, det_mask, self.regr_weight, self.giou_weight)

        loss_task = self.motloss_weight * loss_mot + self.tploss_weight * loss_tp # + self.tploss_weight * loss_det_tp
        
        assert temp_token_mask.equal(current_track_mask)
        loss_social_nce_track = self.snce(current_track_emb, track_enc_emb, targets['contrastive_lable_track'], current_track_mask, temp_token_mask)

        total_loss =  torch.exp(-self.task_loss_weight) * loss_task + torch.exp(-self.social_nce_track_loss_weight) * loss_social_nce_track + (self.task_loss_weight + self.social_nce_track_loss_weight)  # adapt_weight
        
        # total_loss = loss_task
        
        return total_loss, loss_task, loss_social_nce_track
