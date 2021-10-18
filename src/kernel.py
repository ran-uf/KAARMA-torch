import torch
import numpy as np


class Kernel(torch.nn.Module):
    def __init__(self, n, m, kernel_size, w=None):
        super(Kernel, self).__init__()
        self.n = n
        self.m = m
        self.kernel_size = kernel_size
        if w is None:
            self.register_buffer('w', torch.randn((n, m)))
        else:
            if type(w) is torch.Tensor:
                self.register_buffer('w', w)
            elif type(w) is np.ndarray:
                self.register_buffer('w', torch.from_numpy(w))
            else:
                self.register_buffer('w', torch.randn((n, m)))
                raise ValueError('Unknown data type!')

    def forward(self, x):
        o = []
        for _x in x:
            _o = (self.w - _x.repeat(self.n, 1)) ** 2
            o.append(torch.exp(
                -self.kernel_size * torch.sum((self.w - _x.repeat(self.n, 1)) ** 2, dim=1)
            ))
        return torch.stack(o, dim=0)


# if __name__ == "__main__":
#     weights = torch.Tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
#     model = Kernel(4, 3, 10, [weights]).cuda()
#     x = torch.Tensor([[1, 1, 1], [2, 2, 2]]).cuda()
#     output = model(x)
