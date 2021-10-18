import numpy as np
import torch

from src.NTM.memory import NTMMemory
from src.NTM.controller import LSTMController
from src.NTM.ntm import NTM
from src.mKAARMA import MKAARMACell, DiscMaker
from src.NTM.head import NTMReadHead, NTMWriteHead
from src.KAARMA import KernelNode as BaseModel

from src.tomita import generate_tomita_sequence
from src.KAARMA import KAARMA

import matplotlib.pyplot as plt


def myloss(x, y):
    return torch.mean((x - y) ** 2 * torch.pow(3, y))


def get_data(batch_size, length, m, tomita_type):
    s, _ = generate_tomita_sequence(batch_size, length, tomita_type)
    _y, _ = m(s, True)
    return s, _y.data.numpy()


def train(model, train_x, train_y, cri, optim):
    optim.zero_grad()
    pred = model(train_x, train_y)
    loss = cri(pred, train_y)
    loss.backward()
    optim.step()
    return loss.cpu().data.numpy()


def train_single_seq(model, models, ls, epochs, freq_rep, cri, optim):
    min_loss = 1
    for epoch in range(epochs):
        tp = np.random.choice(ls)
        # string_x, string_y = get_data(1, np.random.randint(10, 50), models[tp], tp + 4)
        string_x, string_y = generate_tomita_sequence(1, np.random.randint(10, 50), tp + 4)

        x = torch.from_numpy(string_x.astype(np.float32))
        y = torch.from_numpy(string_y.astype(np.float32))
        _loss = train(model, x, y, cri, optim)

        if (epoch + 1) % freq_rep == 0:
            print(_loss)
        if _loss < min_loss:
            torch.save(model.state_dict(), 'model.pkl')
            min_loss = np.mean(_loss)


models = torch.nn.ModuleList()
trajectories = []
for i in [4, 5, 6, 7]:
    m = BaseModel(4, 1, 2, 2)
    m.load_state_dict(torch.load('model/%d.pkl' % i))
    m.requires_grad_(False)
    models.append(m)
    trajectories.append(torch.from_numpy(np.load('trajectory/%d.npy' % i).astype(np.float32)))
# trajectories = np.vstack(trajectories)
# trajectories = torch.from_numpy(trajectories)
decoder = MKAARMACell(models, trajectories).eval()

m = 20
n = 32
controller_size = 100
controller = LSTMController(26 + m, controller_size, 2)
memory = NTMMemory(n, m, None)
readhead = NTMReadHead(memory, controller_size)
writehead = NTMWriteHead(memory, controller_size)
heads = torch.nn.ModuleList([readhead, writehead])
ntm = NTM(25, 25, controller, memory, heads)

discmaker = DiscMaker(decoder, ntm)


# def train_step(network, train_x, train_y, opt, cri):
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


# model = net()

# model.cuda()
# x = x.cuda()
# y = y.cuda()

# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
#                              lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# criterion = torch.nn.MSELoss()
# criterion = myloss
#
# epochs = 100000
# report_freq = 50
# min_loss = 0.01
#
models = torch.nn.ModuleList()
for i in [0, 1, 2, 3]:
    _m = KAARMA(4, 1, 2, 2)
    _m.node.load_state_dict(torch.load('model/%d.pkl' % (i + 4)))
    _m.eval()
    models.append(_m)

# x = []
# y = []
# order = np.random.permutation([0, 1, 2, 3])
# l = []
# for j in order:
#     _l = np.random.randint(10, 50)
#     l.append(_l)
#     string_x, string_y = get_data(1, _l, models[j], j + 4)
#     x.append(string_x)
#     y.append(string_y)
# x = np.hstack(x)
# y = np.hstack(y)
# x = torch.from_numpy(x.astype(np.float32))
# y = torch.from_numpy(y.astype(np.float32))
# pred = model(x, y)
# gate = torch.cat(model.gate_trajectory).data.numpy()[:, 0, :]
# pred = pred > 0.5
# print(pred == (y > 0.5))
#
# grand_truth = np.zeros((x.shape[1], 4))
# prev = 0
# grand_truth[prev:prev + l[0], 0] = 1
# prev += l[0]
# grand_truth[prev:prev + l[1], 1] = 1
# prev += l[1]
# grand_truth[prev:prev + l[2], 2] = 1
# prev += l[2]
# grand_truth[prev:prev + l[3], 3] = 1
#
# plt.title('grand truth')
# plt.plot(grand_truth)
# plt.show()
#
# plt.title('result')
# plt.plot(gate)
# plt.show()


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discmaker.parameters()),
                             lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, discmaker.parameters()), lr=0.01)
min_loss = 1
report_freq = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
# train_single_seq(discmaker, models, [0, 1, 2, 3], 100000, report_freq, criterion, optimizer)
for epoch in range(100000):
    x = []
    y = []
    avg_loss = []
    order = np.random.permutation([0, 1, 2, 3])
    for j in order:
        string_x, string_y = get_data(1, np.random.randint(10, 50), models[j], j + 4)
        # string_x, string_y = generate_tomita_sequence(1, np.random.randint(10, 50), j + 4)
        # .append(string_x)
        # y.append(string_y)
    # x = np.hstack(x)
    # y = np.hstack(y)
    # x = torch.from_numpy(x.astype(np.float32))
    # y = torch.from_numpy(y.astype(np.float32))
        x = torch.from_numpy(string_x.astype(np.float32))
        y = torch.from_numpy(string_y.astype(np.float32))
        # scheduler.step()
        _loss = train(discmaker, x, y, criterion, optimizer)
        avg_loss.append(_loss)

    if (epoch + 1) % report_freq == 0:
        print(np.mean(avg_loss))
    if np.mean(avg_loss) < min_loss:
        torch.save(discmaker.state_dict(), 'model.pkl')
        min_loss = np.mean(avg_loss)
np.save('gates.npy', discmaker.gate_trajectories)

j = 2
string_x, string_y = get_data(1, 100, models[j], j + 4)
string_x = torch.from_numpy(string_x.astype(np.float32))
string_y = torch.from_numpy(string_y.astype(np.float32))
pred = discmaker(string_x, string_y)
gate = torch.stack(discmaker.gate_trajectories, dim=1)
plt.title('result grammar #%d' % (j + 4))
plt.plot(gate.data.numpy()[0])
plt.legend(['4', '5', '6', '7'])
plt.show()

# print(gate.data.numpy()[0])
torch.save(discmaker.state_dict(), 'model.pkl')