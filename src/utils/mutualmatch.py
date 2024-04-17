import torch
def MutualMatching(m):
    # mutual matching
    # batch_size, det, track = m.shape


    # get max
    m_r_max,_=torch.max(m,dim=-2,keepdim=True)
    m_c_max,_=torch.max(m,dim=-1,keepdim=True)

    # eps = 1e-9
    m_r=m/(m_r_max)
    m_c=m/(m_c_max)


    m=m*(m_r*m_c) # parenthesis are important for symmetric output 
        
    return m