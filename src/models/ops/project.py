import torch
from torch import nn, Tensor
from einops import repeat

from src.models.ops.mlp import MLP # def MLP(channels: List[int], do_ln: bool = True) -> nn.Module
from src.models.ops.position_encoding import build_position_encoding # def build_position_encoding(pos_emb_type, d_model, max_len = 100)

class TrackEmbedding(nn.Module):
    '''
    跟踪信息嵌入至高维空间
    '''
    def __init__(self, feature_dim: int, layers, pos_emb_type, frame_nums, emb_dropout = 0., num_embeddings = 80):
    # '''
    # pos_emb_type: option[ sine/learned ] 
    # '''
        super().__init__()
        # self.encoder = MLP([4] + layers + [feature_dim])
        # self.encoder = nn.Embedding(num_embeddings+1, feature_dim//4)
        # self.proj = MLP([feature_dim, feature_dim])

        self.track_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.time_pos_emb = build_position_encoding(pos_emb_type, feature_dim, 1+frame_nums)        
        self.dropout = nn.Dropout(emb_dropout)
        
        self.pos_emb_type = pos_emb_type # 便于初始化判断
 
    def forward(self, x: Tensor):
        batch, _, _= x.shape
        # x = self.encoder(x) # (batch, f_nums, 5) ->(batch, f_nums, feature_dim)
        # x = x.flatten(2)
        # x = self.proj(x)

        track_tokens = repeat(self.track_token, '() n e -> b n e', b=batch) # 将track_token 扩展batch次,(batch, 1,feature_dim)
        x = torch.cat([track_tokens, x], dim=1) # 将track token加到proj(x)上 ,(batch,1+f_nums,feature_dim)
        
        x = self.time_pos_emb(x) # 增加时间位置编码信息,广播机制
        x = self.dropout(x)
        
        return x


class DetectEmbedding(nn.Module):
# '''检测信息嵌入至高维空间'''
    def __init__(self, feature_dim: int, layers, emb_dropout = 0., num_embeddings = 80):
        super().__init__()
        # self.encoder = MLP([4] + layers + [feature_dim])
        # self.encoder = nn.Embedding(num_embeddings+1, feature_dim//4)
        # self.proj = MLP([feature_dim, feature_dim])

        self.dropout = nn.Dropout(emb_dropout)
 
    def forward(self, x: Tensor):
        # x = self.encoder(x) #+ self.det[:obj_num,:] # 进行线性变换操作 (batch, max_obj, 256)
        # x = x.flatten(2)
        # x = self.proj(x)
        x = self.dropout(x)
        return x

