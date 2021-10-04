import numpy as np
import torch

from src.NTM.memory import NTMMemory
from src.model import Controller, NTMCell
from src.NTM.head import NTMReadHead, NTMWriteHead
from src.KAARMA import KernelNode as BaseModel

from src.tomita import generate_tomita_sequence


def net():
    memory = NTMMemory(N=100, M=4)
    memory.reset(1)
    controller = Controller(BaseModel, input_size=17, hidden_size=100, num_layers=2)
    controller.load_external_memory('./model')
    heads = torch.nn.ModuleList()
    heads.append(NTMReadHead(memory, 100))
    heads.append(NTMWriteHead(memory, 100))
    return NTMCell(1, 1, controller, memory, heads)


def train_step(network, train_x, train_y, opt, cri):
    network.reset(1)
    opt.zero_grad()

    length = train_x.shape[1]
    state, error = network.init_states()
    e = []
    network.train()
    for i in range(length):
        _x, _y = train_x[:, i].unsqueeze(1), train_y[:, i].unsqueeze(1)
        output, state = network((_x, error), state)
        error = _y - output
        e.append(error)
    e = torch.cat(e)
    loss = cri(e, torch.zeros_like(e))
    loss.backward()
    opt.step()
    return loss.data


strings, labels = generate_tomita_sequence(1, 100, 5)
x = torch.from_numpy(strings.astype(np.float32))
y = torch.from_numpy(labels.astype(np.float32))

model = net()

# model.cuda()
# x = x.cuda()
# y = y.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

epochs = 100000
report_freq = 50
for i in range(epochs):
    ls = train_step(model, x, y, optimizer, criterion)
    if (i + 1) % report_freq == 0:
        print(ls)


