import torch

def collate_fn_dets(batch):
    det = []
    det_pad_mask = []
    temp = []
    temp_pad_mask = []
    id_cls = []
    id_cls_mask = []
    box_reg = []
    box_reg_mask = []
    contrastive_lable_track = []
    contrastive_lable_det = []
    img_size  = []
    
    inputs = {}
    targets = {}

   
    for sample in batch:
        # sample:(input_(dict), target_(dict))
        if (sample[0] == None) & (sample[1] == None):
            continue

        det.append(torch.tensor(sample[0]['det'])) # (max_obj, 5)
        det_pad_mask.append(torch.tensor(sample[0]['det_pad_mask'])) # (max_obj,)
        temp.append(torch.tensor(sample[0]['temp'])) # (max_obj, f_nums, 5)
        temp_pad_mask.append(torch.tensor(sample[0]['temp_pad_mask'])) # (max_obj,f_nums)

        id_cls.append(torch.tensor(sample[1]['id_cls'])) # (max_obj, max_obj)
        id_cls_mask.append(torch.tensor(sample[1]['id_cls_mask'])) # (max_obj,max_obj)
        box_reg.append(torch.tensor(sample[1]['box_reg'])) # (max_obj, 4)
        box_reg_mask.append(torch.tensor(sample[1]['box_reg_mask'])) # (max_obj,)
        contrastive_lable_track.append(torch.tensor(sample[1]['contrastive_lable_track']))
        contrastive_lable_det.append(torch.tensor(sample[1]['contrastive_lable_det']))
        img_size.append(torch.tensor(sample[1]['img_size'])) # (max_obj, 4)

    inputs['det'] = torch.stack(det, 0) # (batch, max_obj, 5)
    inputs['det_pad_mask'] = torch.stack(det_pad_mask, 0) # (batch, max_obj)
    inputs['temp'] = torch.cat(temp, 0) # (batch * max_obj , f_nums, 5)
    inputs['temp_pad_mask'] = torch.cat(temp_pad_mask, 0) # (batch*max_obj , 1+f_nums)
    targets['id_cls'] = torch.stack(id_cls, 0) # (batch, max_obj, max_obj)
    targets['id_cls_mask'] = torch.stack(id_cls_mask, 0) # (batch, max_obj, max_obj)
    targets['box_reg'] = torch.stack(box_reg, 0) # (batch, max_obj, 4)
    targets['box_reg_mask'] = torch.stack(box_reg_mask, 0) # (batch, max_obj)
    targets['contrastive_lable_track'] = torch.cat(contrastive_lable_track, 0) # (batch*max_obj)
    targets['contrastive_lable_det'] = torch.cat(contrastive_lable_det, 0) # (batch*max_obj)

    # 便于代码阅读与编写，增加轨迹预测输入索引
    inputs['current_track'] = targets['box_reg']
    inputs['current_track_mask'] = targets['box_reg_mask']
    targets['img_size'] = torch.stack(img_size, 0) # (batch, max_obj, 4)

    return inputs, targets
