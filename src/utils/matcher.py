import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    def __init__(self, cost_bbox: float = 5, cost_giou: float = 2, min_giou: float = 0.5):
        """Creates the matcher
        Params:
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.max_cost_giou = 1 - min_giou

    @torch.no_grad()
    def forward(self, out_bbox, tgt_bbox, association=None):
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox[:,:2], tgt_bbox[:,:2], p=1)
        # cost_bbox_mask = (cost_bbox>=5.0)
        # cost_bbox = cost_bbox.masked_fill(cost_bbox_mask, torch.finfo(torch.float32).max)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[:,:4]), box_cxcywh_to_xyxy(tgt_bbox[:,:4])) # -giou
        # cost_giou_mask = (cost_giou>=self.max_cost_giou)
        # cost_giou = cost_giou.masked_fill(cost_giou_mask, torch.finfo(torch.float32).max)


        C = self.cost_bbox * cost_bbox + self.cost_giou * (1+cost_giou)
        if association!=None:
            # cost_ass_mask = (association<=0.3)
            # association = association.masked_fill(cost_ass_mask, torch.finfo(torch.float32).max)

            C = C + 1.0 * (1-association) #1.5
            # C = C

        out_ids = out_bbox[:,4]
        tgt_ids = tgt_bbox[:,4]
        out_ids_m = torch.unsqueeze(out_ids,1).repeat(1, len(tgt_ids))
        tgt_ids_m = torch.unsqueeze(tgt_ids,0).repeat(len(out_ids), 1)
        cls_m = torch.eq(out_ids_m, tgt_ids_m)
        C.masked_fill_(~cls_m, torch.finfo(torch.float64).max)

        # 匈牙利匹配
        row_ind, col_ind = linear_sum_assignment(C)
        indices = [(r,c) for r,c in zip(row_ind, col_ind) if ((1 + cost_giou[r,c] <= self.max_cost_giou) and (C[r,c]<torch.finfo(torch.float64).max))]
 

        
        # if association!=None:
        #     C = 1-association
        #     row_ind, col_ind = linear_sum_assignment(C)
        #     indices = [(r,c) for r,c in zip(row_ind, col_ind) if ((1 + cost_giou[r,c] <= self.max_cost_giou) and (association[r,c]>=0.5))]  # 定义GIOU Loss = 1 - GIOU，注意到GIOU范围在[-1, 1]，那么GIOU Loss的范围在[0, 2]
        # else:
        #     # Final cost matrix
        #     C = self.cost_bbox * cost_bbox + self.cost_giou * (cost_giou)
        #     # 匈牙利匹配
        #     row_ind, col_ind = linear_sum_assignment(C)
        #     indices = [(r,c) for r,c in zip(row_ind, col_ind) if (1 + cost_giou[r,c] <= self.max_cost_giou)]
        
        mconf = [-cost_giou[r,c] for r,c in indices]

        return indices, mconf
