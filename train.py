import numpy as np
import torch

from src.NTM.memory import NTMMemory
from src.model import Controller, NTMCell
from src.NTM.head import NTMReadHead, NTMWriteHead
from src.KAARMA import KernelNode as BaseModel


def net():
    memory = NTMMemory(N=100, M=4)
    memory.reset(1)
    controller = Controller(BaseModel, input_size=17, hidden_size=100, num_layers=2)
    controller.load_external_memory('./model')
    readHead = NTMReadHead(memory, 100)
    writeHead = NTMWriteHead(memory, 100)
    return NTMCell(1, 1, controller, memory, [readHead, writeHead])


model = net()
inp = torch.from_numpy(np.array([[1.]], dtype=np.float32))
error = torch.from_numpy(np.array([[0.]], dtype=np.float32))
_, state = model((inp, error), (None, None, [torch.zeros((1, 100)), torch.zeros((1, 100))], None))
out, state = model((inp, error), state)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
criterion = torch.nn.MSELoss()

