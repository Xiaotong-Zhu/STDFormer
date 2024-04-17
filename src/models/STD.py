# STD_NCE_v2
import torch
from torch import nn
import torch.nn.functional as F
import copy

from src.models.ops.mlp import MLP # def MLP(channels: List[int], do_ln: bool = True) -> nn.Module
from src.models.ops.project import TrackEmbedding, DetectEmbedding # def __init__(self, feature_dim: int, layers: List[int], pos_emb_type: str, frame_nums: int, emb_dropout = 0.) / def __init__(self, feature_dim: int, layers: List[int], emb_dropout = 0.)
from src.models.ops.STDEncoderLayer import STDEncoderLayer # def __init__(self,d_model,nhead,attention='full',attention_dropout=0.)

import math
import warnings

from einops.layers.torch import Reduce
class TemporalMLP(nn.Sequential):
    '''
    XXX:需要加一些线性变换/非线性激活函数对轨迹时间信息做进一步融合提取吗
    '''
    def __init__(self, emb_size: int = 32):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'), # (batch,1+f_nums,emb_size)
            nn.LayerNorm(emb_size)
            )

class STD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        ##########################################################
        # Embedding层: 将输入history和det嵌入高维空间
        ##########################################################
        self.emb = nn.Embedding(cfg.MODEL.NUM_EMBEDDINGS+1, cfg.MODEL.EMBEDDING_SIZE)
        self.proj = MLP([cfg.MODEL.EMBEDDING_SIZE*5, cfg.MODEL.EMBEDDING_SIZE])
        # self.proj = MLP([4]+cfg.MODEL.EMBEDDING_LAYERS)

        self.t_emb = TrackEmbedding(cfg.MODEL.EMBEDDING_SIZE, cfg.MODEL.TRACKEMBEDDING_LAYERS, cfg.MODEL.TRACK_POS_EMB_TYPE, cfg.DATA.HISTORY_FRAME_NUM, cfg.MODEL.TRACKEMBEDDING_DROPOUT)
        self.d_emb = DetectEmbedding(cfg.MODEL.EMBEDDING_SIZE, cfg.MODEL.DETECTEMBEDDING_LAYERS, cfg.MODEL.DETECTEMBEDDING_DROPOUT)
        
        ##########################################################
        # STD 编码层：交叉编码时空与检测信息
        ##########################################################
        self.std_encoder_layer_names = cfg.MODEL.STD_ENCODER_LAYER_NAMES # [slef, cross, ...]
        std_encoder_layer = STDEncoderLayer(cfg.MODEL.EMBEDDING_SIZE, cfg.MODEL.STD_HEAD, cfg.MODEL.STD_ATTENTION, cfg.MODEL.FULL_ATTENTION_DROPOUT)
        self.std_encoder_layers = nn.ModuleList([copy.deepcopy(std_encoder_layer) for _ in range(len(self.std_encoder_layer_names))])

        # self.std_decoder_layer_names = cfg.MODEL.STD_DECODER_LAYER_NAMES # [slef, cross, ...]
        # std_decoder_layer = STDEncoderLayer(cfg.MODEL.EMBEDDING_SIZE, cfg.MODEL.STD_HEAD, cfg.MODEL.STD_ATTENTION, cfg.MODEL.FULL_ATTENTION_DROPOUT)
        # self.std_decoder_layers = nn.ModuleList([copy.deepcopy(std_decoder_layer) for _ in range(len(self.std_decoder_layer_names))])

        self.temp_token_proj = TemporalMLP(cfg.MODEL.EMBEDDING_SIZE)
        ##########################################################
        # 任务输出头：轨迹当前帧预测头
        # 关联匹配任务的实现在forwad，直接计算相似度输出,这里仅定义温度系数
        ##########################################################
        self.TPHead = MLP(cfg.MODEL.TPHEAD_LAYERS + [4])
        # self.displacement_proj = MLP([cfg.MODEL.EMBEDDING_SIZE, 1])
        self.temperature = cfg.MODEL.SIM_TEMPERATURE
        
        ##########################################################
        # 权重初始化
        ##########################################################
        self._init_weights()

    
    def forward(self, inputs, is_train=True):
        batch = inputs['det'].shape[0]
        track_nums = inputs['temp'].shape[0] // batch
        
        # ################################ 0.输入整数化 ##################################################
        temp = inputs['temp'].clamp(min=0.).int()
        det = inputs['det'].clamp(min=0.).int()
        temp_emb = self.proj(self.emb(temp).flatten(2))
        det_emb = self.proj(self.emb(det).flatten(2))

        last_boxes = inputs['temp'][:,0,:].reshape(batch, track_nums, 5)
        last_boxes_clamp = (last_boxes).clamp(min=0).int()
        last_boxes_emb = self.proj(self.emb(last_boxes_clamp).flatten(2))

        if is_train:
            last_cls = last_boxes[:,:,-1].unsqueeze(2)
            current_track = torch.cat((inputs['current_track'].clamp(min=0.).int(), last_cls.int()),2)
            current_track_emb = self.proj(self.emb(current_track).flatten(2))
        
        ##################### 1. 时空信息与检测的embedding & 当前帧轨迹预测编码###########################
        temp_emb = self.t_emb(temp_emb) # (batch*max_obj, f_nums, 5) -> (batch*max_obj, 1+f_nums, emb)
        det_emb = self.d_emb(det_emb) # (batch, max_obj, 4) -> (batch, max_obj, emb)
        if is_train:
            current_track_emb = self.d_emb(current_track_emb) # (batch, max_obj, 4) -> (batch, max_obj, emb)
        last_boxes_emb = self.d_emb(last_boxes_emb) # (batch, max_obj, 4) -> (batch, max_obj, emb)


        

        ############################# 2. 交叉编码 ###################################
        # 2.1 mask确定，encoder要的mask是，非pad为true
        if is_train:
            temp_token_pad_mask = inputs['temp_pad_mask'][:, 0] # (batch*max_object, 1+f_nums) -> (batch*max_object, 1)
            
            temp_token_mask = ~(temp_token_pad_mask.reshape((batch,track_nums)) > 0) # (batch*max_object, 1) -> (batch,max_object)
            temp_mask = ~(inputs['temp_pad_mask'] > 0)  # (batch*max_object, 1+f_nums)
            det_mask = ~(inputs['det_pad_mask'] > 0) #(batch,max_object)
            current_track_mask = inputs['current_track_mask'].bool()

        else:
            temp_token_mask = None
            temp_mask = None
            det_mask = None
            current_track_mask = None
            
        # 2.2 T、D交叉注意力编码
        track_enc_emb = None
        for layer, name in zip(self.std_encoder_layers, self.std_encoder_layer_names):
            if name == 'TEncoder':
                # TEncoder
                temp_emb = layer(temp_emb, temp_emb, temp_mask, temp_mask)
            
            elif name == 'TimeTokenCross':
                # TimeTokenCross
                temp_token = self.temp_token_proj(temp_emb).unsqueeze(0).contiguous().view(batch, track_nums, -1) # (bacth, max_obj, emb)
                temp_token = layer(temp_token, temp_token, temp_token_mask, temp_token_mask)
                track_enc_emb = temp_token
                temp_emb[:, 0, :] = temp_token.reshape(temp_emb.shape[0], temp_emb.shape[2])
            
            elif name == 'DEncoder':
                # DEncoder
                det_emb = layer(det_emb, det_emb, det_mask, det_mask)
                if is_train:
                    current_track_emb = layer(current_track_emb, current_track_emb, current_track_mask, current_track_mask)
                last_boxes_emb = layer(last_boxes_emb, last_boxes_emb, current_track_mask, current_track_mask)

                det_enc_emb = det_emb

            elif name == 'TDDecoder':
                # TDDecoder
                temp_token = self.temp_token_proj(temp_emb).unsqueeze(0).contiguous().view(batch, track_nums, -1) # (bacth, max_obj, emb)
                temp_token = layer(temp_token, det_emb, temp_token_mask, det_mask)
                track_enc_emb = temp_token
                temp_emb[:, 0, :] = temp_token.reshape(temp_emb.shape[0], temp_emb.shape[2])

            else:
                raise KeyError
        
        ############################## 3.任务输出处理 ###############################
        # 3.1 相似度矩阵计算
        ############################################################################
        # det_enc_emb = nn.functional.normalize(det_enc_emb, dim=-1)
        # track_enc_emb = nn.functional.normalize(track_enc_emb, dim=-1)
        sim_matrix = torch.einsum("nlc,nsc->nls", det_emb, track_enc_emb) # exp3
        # sim_matrix = torch.einsum("nlc,nsc->nls", det_enc_emb, track_enc_emb)  #/ self.temperature # dsl-1
        if (det_mask is not None) or (temp_token_mask is not None):
            sim_matrix.masked_fill_(~(det_mask[:,:, None] * temp_token_mask[:, None, :]), torch.finfo(torch.float32).min)
        
        # dual-softmax-1
        # conf_matrix = F.softmax(sim_matrix// self.temperature, 1) * F.softmax(sim_matrix// self.temperature, 2)
        conf_matrix = F.softmax(sim_matrix * F.softmax(sim_matrix / self.temperature, dim=1)*sim_matrix.shape[1], dim = 2)

        ############################## 3.任务输出处理 ###############################
        # 3.2 位移动预测计算
        ############################################################################
        # displacement_emb = temp_token[:,:, None,:] + det_emb[:,None,:,:]
        # displacement_emb = displacement_emb * conf_matrix.permute(0,2,1)[:,:,:,None]
        # displacement_emb = torch.cumsum(displacement_emb, dim = 2)[:,:,-1,:]

        # displacement_emb = self.displacement_proj(displacement_emb).squeeze(-1)

        

        # for layer, name in zip(self.std_decoder_layers, self.std_decoder_layer_names):
        #     if name == 'Track_Decoder':
        #         # Track_Decoder
        #         # temp_token = temp_emb[:,0,:].unsqueeze(0).contiguous().view(batch, track_nums, -1) # (bacth, max_obj, emb)
        #         # temp_token = layer(temp_token, temp_token, temp_token_mask, temp_token_mask)
        #         # temp_emb[:, 0, :] = temp_token.reshape(temp_emb.shape[0], temp_emb.shape[2])
        #         # if is_train:
        #         #     det_emb = layer(det_emb, det_emb, det_mask, det_mask)
        #         displacement_emb = layer(displacement_emb, displacement_emb, temp_token_mask, temp_token_mask)
        #         # det_emb = layer(det_emb, det_emb, det_mask, det_mask)

        #     else:
        #         raise KeyError

        displacement_emb = temp_token - last_boxes_emb
        #displacement_emb = torch.concat((temp_token , last_boxes_emb),-1)
        tp = self.TPHead(displacement_emb)#.tanh() # (bacth, max_obj, 4)

        # tp = self.TPHead(temp_token)#.sigmoid() # (bacth, max_obj, 4)

        # det_tp = self.TPHead(det_emb).relu()

        # last_boxes = inputs['temp'][:,0,:].reshape(batch, track_nums, 4)
        # cur_boxes = torch.zeros((batch, track_nums, 4)).double().cuda()
        # cur_boxes[...,0] = tp[...,0] * last_boxes[...,2] + last_boxes[...,0] # dx*w1 + x1
        # cur_boxes[...,1] = tp[...,1] * last_boxes[...,3] + last_boxes[...,1] # dy*h + y1
        # cur_boxes[...,2] = torch.exp(tp[...,2])*last_boxes[...,2]
        # cur_boxes[...,3] = torch.exp(tp[...,3])*last_boxes[...,3]
        # if is_train:
        #     cur_boxes_mask = current_track_mask.unsqueeze(2).repeat_interleave(4, dim=2)
        #     cur_boxes.masked_fill(~cur_boxes_mask, 0.0)

        # dance , new mot
        # cur_boxes = torch.zeros((batch, track_nums, 4)).double().cuda()
        # cur_boxes[...,0] = torch.exp(tp[...,0])*last_boxes[...,0]
        # cur_boxes[...,1] = torch.exp(tp[...,1])*last_boxes[...,1]
        # cur_boxes[...,2] = torch.exp(tp[...,2])*last_boxes[...,2]
        # cur_boxes[...,3] = torch.exp(tp[...,3])*last_boxes[...,3]
        # if is_train:
        #     cur_boxes_mask = current_track_mask.unsqueeze(2).repeat_interleave(4, dim=2)
        #     cur_boxes.masked_fill(~cur_boxes_mask, 0.0)

        # mot17 20
        cur_boxes = torch.zeros((batch, track_nums, 4)).double().cuda()
        cur_boxes[...,0] = tp[...,0] + last_boxes[...,0]#.clamp(min=0.) # dx*w1 + x1
        cur_boxes[...,1] = tp[...,1] + last_boxes[...,1]#.clamp(min=0.) # dy*h + y1
        cur_boxes[...,2] = (tp[...,2] + last_boxes[...,2])#.clamp(min=0.)
        cur_boxes[...,3] = (tp[...,3] + last_boxes[...,3])#.clamp(min=0.)
     
        if is_train:
            cur_boxes_mask = current_track_mask.unsqueeze(2).repeat_interleave(4, dim=2)
            cur_boxes.masked_fill(~cur_boxes_mask, 0.0)

        # last_boxes = inputs['temp'][:,0,:].reshape(batch, track_nums, 4)
        # cur_boxes = torch.zeros((batch, track_nums, 4)).double().cuda()
        # cur_boxes[...,0] = tp[...,0] * last_boxes[...,2] + last_boxes[...,0] # dx*w1 + x1
        # cur_boxes[...,1] = tp[...,1] * last_boxes[...,3] + last_boxes[...,1] # dy*h + y1
        # cur_boxes[...,2] = (tp[...,2] * last_boxes[...,2] + last_boxes[...,2]).clamp(min=0.)
        # cur_boxes[...,3] = (tp[...,3] * last_boxes[...,3] + last_boxes[...,3]).clamp(min=0.)
        # if is_train:
        #     cur_boxes_mask = current_track_mask.unsqueeze(2).repeat_interleave(4, dim=2)
        #     cur_boxes.masked_fill(~cur_boxes_mask, 0.0)
        
        ################################## 4.返回值 ##################################
        if is_train:
            return conf_matrix, cur_boxes, track_enc_emb, current_track_emb, temp_token_mask, current_track_mask#, det_tp, inputs['det'], det_mask
        else:
            cur_boxes_cls = last_boxes[:,:,-1].unsqueeze(2)
            cur_boxes = torch.concat( (cur_boxes, cur_boxes_cls), -1) # (cx,cy,w,h,cls)
            return conf_matrix, cur_boxes

        
    def _init_weights(self):
    # weight initialization
        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            def norm_cdf(x):
                return (1. + math.erf(x / math.sqrt(2.))) / 2.

            if (mean < a - 2 * std) or (mean > b + 2 * std):
                warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                              "The distribution of values may be incorrect.", stacklevel=2)
            with torch.no_grad():
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)
                tensor.uniform_(2 * l - 1, 2 * u - 1)
                tensor.erfinv_()
                tensor.mul_(std * math.sqrt(2.))
                tensor.add_(mean)
                tensor.clamp_(min=a, max=b)
                return tensor


        for m in self.modules():
            ################## TrackEmbedding ###########################
            if isinstance(m, TrackEmbedding):
                # trunc_normal_(m.time_pos_emb, std=0.02) if (hasattr(m, "time_pos_emb") and m.pos_emb_type=='learned') else None
                trunc_normal_(m.track_token, std=0.02) if hasattr(m, "track_token") else None

                # for i in range(len(m.encoder)):
                #     if isinstance(m.encoder[i], nn.Linear):
                #         trunc_normal_(m.encoder[i].weight, std=0.02)
                #         if m.encoder[i].bias is not None:
                #             nn.init.constant_(m.encoder[i].bias, 0)
                
            ################## DetectEmbedding ###########################
            elif isinstance(m, DetectEmbedding):
                # for j in range(len(m.encoder)):
                #     if isinstance(m.encoder[j], nn.Linear):
                #         trunc_normal_(m.encoder[j].weight, std=0.02)
                #         if m.encoder[j].bias is not None:
                #             nn.init.constant_(m.encoder[j].bias, 0)
                pass
            
            ########################### MLP ##############################
            elif isinstance(m, nn.Sequential):
                for j in range(len(m)):
                    if isinstance(m[j], nn.Linear):
                        trunc_normal_(m[j].weight, std=0.02)
                        if m[j].bias is not None:
                            nn.init.constant_(m[j].bias, 0)



        