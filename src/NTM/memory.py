# Copyright (c) 2017, Guy Zana <guyzana@gmail.com>
# All rights reserved.
# Github repository: https://github.com/loudinthecloud/pytorch-ntm.git


"""An NTM's memory implementation."""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c


class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M, fn):
        """Initialize the NTM Memory matrix.
        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.
        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()

        self.batch_size = None
        self.memory = None
        self.prev_mem = None

        if N is None:
            self.mem_bias = None
            self.load_memory(fn)
        else:
            self.N = N
            self.M = M

            # The memory bias allows the heads to learn how to initially address
            # memory locations by content
            self.register_buffer('mem_bias', torch.Tensor(N, M))

            # Initialize memory bias
            stdev = 1 / (np.sqrt(N + M))
            nn.init.uniform_(self.mem_bias, -stdev, stdev)

    def load_memory(self, fn):
        files = os.listdir(fn)
        m = []
        for file in files:
            m.append(np.load(fn + '/' + file))
        m = np.vstack(m)
        self.mem_bias = torch.from_numpy(m)
        self.M = self.mem_bias.shape[1]
        self.N = self.mem_bias.shape[0]

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)
        self.memory.require_grad = False

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, k, β, g, s, γ, w_prev):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        # Content focus
        wc = self._similarity(k, β)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)

        return w

    def _similarity(self, k, β):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(β * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    @staticmethod
    def _interpolate(w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = []
        for b in range(self.batch_size):
            result.append(_convolve(wg[b], s[b]).unsqueeze(0))
        result = torch.cat(result, dim=0)
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w