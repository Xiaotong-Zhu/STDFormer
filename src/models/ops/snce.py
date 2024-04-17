import torch
import torch.nn as nn
import torch.nn.functional as F

class dual_softmax_loss(nn.Module):
    def __init__(self,):
        super(dual_softmax_loss, self).__init__()
        
    def forward(self, sim_matrix, temp=1000, mask=None):
        sim_matrix = sim_matrix * F.softmax(sim_matrix/temp, dim=0)*len(sim_matrix) #With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return 

class SocialNCE(nn.Module):
    '''
    Social contrastive loss, encourage the extracted motion representation to be aware of socially unacceptable events
    '''

    def __init__(self, encoder=None, temperature=0.1):
        super().__init__()
        # encoder
        self.encoder = encoder
        # nce
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
       
    def forward(self, queries, keys, labels, query_mask=None, key_mask=None):
        # current_track_emb, track_enc_emb, targets['contrastive_lable_track'], current_track_mask, temp_token_mask
        '''
        Input:
            query: a tensor of shape (B, obj_num, emb) for track/det embedding  #  [N, L, D]
            keys: a tensor of shape (B, obj_num, 4) for track/det state, i.e. [cx, cy, w, h],encoder编码变成[N,S,D]
            query_mask: [N,L]
            key_mask: [N,S]

        Output:
            social nce loss
        '''
        # embedding，相当于MSA里的project
        if self.encoder != None:
            keys = self.encoder(keys) # (B, obj_num, 4) -> (B, obj_num, emb)

        # normalization,相当于layernorm
        # queries = nn.functional.normalize(queries, dim=-1)
        # keys = nn.functional.normalize(keys, dim=-1)
        
        # similarity
        similarity = torch.einsum("nld,nsd->nls", queries, keys)
        if key_mask is not None:
            # fill_value = torch.finfo(torch.float64).min # torch.finfo(torch.float64).min/0.1=-inf
            fill_value = torch.finfo(torch.float32).min
            sim_mask = query_mask[:,:,None]*key_mask[:,None,:]
            similarity.masked_fill_(~(sim_mask), fill_value)

        # logits
        batch, query_num, key_num = similarity.shape

        # dual-softmax-1
        # sim_matrix = (similarity / self.temperature)
        # logits = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        # softmax
        # logits = F.softmax(similarity / self.temperature, 2)

        # dual-softmax-2
        logits = similarity * F.softmax(similarity / self.temperature, dim=1)*similarity.shape[1]

        logits = logits.reshape((batch*query_num, key_num))


        # loss
        # labels = torch.arange( queries.shape[1] ).unsqueeze(0).repeat_interleave(queries.shape[0], dim=0) # cpu
        labels_masked = (labels.masked_fill_(~(query_mask).flatten(), -1))
        loss = self.criterion(logits, labels_masked)

        return loss