from torch import nn

def MLP(channels: list, do_ln: bool = False) -> nn.Module:
    """ Multi-layer perceptron with ReLU"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Linear(channels[i - 1], channels[i], bias=False))
        if i < (n-1):
            if do_ln:
                layers.append(nn.LayerNorm(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)