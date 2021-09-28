from KAARMA import KernelNode as BaseModel
import torch
import os


class Controller(torch.nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_m):
        super(Controller, self).__init__()
        self.rnn_cell = torch.nn.RNNCell(dim_input, dim_hidden)
        self.context_encoding = torch.nn.Linear(dim_hidden, dim_hidden)
        self.context_decodeing = torch.nn.Linear(dim_hidden, dim_hidden)
        self.gate_r
        self.gate_w


class DynamicModel(torch.nn.Module):
    def __init__(self, sz_ex):
        super(DynamicModel, self).__init__()
        self.controller = Controller(10, 10)
        self.ex_memory = torch.nn.ModuleList()
        self.sz_ex = sz_ex

    def load_external_memory(self, fn):
        _, _, files = os.walk(fn)
        for file in files:
            state_dict = torch.load(fn + '/' + file)
            node = BaseModel()
            node.load_state_dict(state_dict)
            self.ex_memory.append(node)
