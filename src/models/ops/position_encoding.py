import torch
from torch import nn
import math

class PositionEmbeddingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 1-dimensional historys
    """
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        ## 上面代码获取之后得到的pe:[max_len*d_model]

        ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        '''
        Args:
            x: [N, 1 + frames, d_model], N = batch * max_obj
        '''
        pos_emb = x + self.pe[:, :x.size(1), :x.size(2)]
        
        return self.dropout(pos_emb)

   
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # self.pe = nn.Parameter(torch.randn((1, max_len, d_model)))
        self.pe = nn.Embedding(max_len, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pe.weight)

    def forward(self, x):
        '''
        Args:
            x: [N, 1 + frames, d_model], N = batch * max_obj
        '''
        # pos_emb = x + self.pe[None, :x.size(1), :x.size(2)]
        # return pos_emb

        b,l = x.shape[:-1]
        i = torch.arange(l, device=x.device)
        x_emb = self.pe(i)
        pos = x_emb.unsqueeze(0).repeat(b, 1, 1)
        return x+pos
        
        
        

def build_position_encoding(pos_emb_type, d_model, max_len = 100):
    if pos_emb_type=='sine':
        position_embedding = PositionEmbeddingSine(d_model, max_len)
    elif pos_emb_type=='learned':
        position_embedding = PositionEmbeddingLearned(d_model, max_len)
    else:
        raise ValueError(f"not supported {pos_emb_type}")

    return position_embedding