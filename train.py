import numpy as np
import torch

from src.NTM.memory import NTMMemory
from src.model import Controller, NTMCell, NTM
from src.NTM.head import NTMReadHead, NTMWriteHead
from src.KAARMA import KernelNode as BaseModel

from src.tomita import generate_tomita_sequence
from src.KAARMA import KAARMA


def myloss(x, y):
    return torch.mean((x - y) ** 2 * torch.pow(3, y))


def get_data(batch_size, length, m, tomita_type):
    s, _ = generate_tomita_sequence(batch_size, length, tomita_type)
    _y, _ = m(s[0], True)
    return s, _y[np.newaxis, :]


def net():
    # memory = NTMMemory(N=100, M=4)
    memory = NTMMemory(N=None, M=None, fn='trajectory')
    memory.reset(1)
    controller = Controller(BaseModel, input_size=17, hidden_size=100, num_layers=2)
    controller.load_external_memory('./model')
    heads = torch.nn.ModuleList()
    heads.append(NTMReadHead(memory, 100))
    # heads.append(NTMWriteHead(memory, 100))
    return NTM(1, 1, controller, memory, heads)


# def train_step(network, train_x, train_y, opt, cri):
#     network.reset(1)
#     opt.zero_grad()
#
#     length = train_x.shape[1]
#     state, error = network.init_states()
#     e = []
#     network.train()
#     for i in range(length):
#         _x, _y = train_x[:, i].unsqueeze(1), train_y[:, i].unsqueeze(1)
#         output, state = network((_x, error), state)
#         error = _y - output
#         e.append(error)
#     e = torch.cat(e)
#     loss = cri(e, torch.zeros_like(e))
#     loss.backward()
#     opt.step()
#     return loss.data


def train_ntm(network, train_x, train_y, opt, cri):
    network.reset(1)
    opt.zero_grad()
    pred = network(train_x, train_y)
    loss = cri(pred, train_y)
    loss.backward()
    opt.step()
    return loss.data


model = net()

# model.cuda()
# x = x.cuda()
# y = y.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
# criterion = myloss

epochs = 100000
report_freq = 50
min_loss = 0.01

models = torch.nn.ModuleList()
for i in [0, 1, 2, 3]:
    _m = KAARMA(4, 1, 2, 2)
    _m.node.load_state_dict(torch.load('model/%d.pkl' % (i + 4)))
    _m.eval()
    models.append(_m)

for i in range(epochs):
    loss = []
    for j in [0, 1, 2, 3]:
        length = np.random.randint(3, 100)
        x, y = get_data(1, length, models[j], j + 4)
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        _loss = train_ntm(model, x, y, optimizer, criterion)
        loss.append(_loss)
    if (i + 1) % report_freq == 0:
        print(np.mean(loss))
    if np.mean(loss) / 2 < min_loss:
        torch.save(model.state_dict(), 'model.pkl')
        min_loss = np.mean(loss)

j = 1
x, y = get_data(1, 100, models[j], j + 4)
x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32))
model(x, y)
gate = torch.cat(model.gate_trajectory).data.numpy()[:, 0, :]
