import numpy as np
import torch

from src.NTM.memory import NTMMemory
from src.NTM.controller import LSTMController
from src.NTM.ntm import NTM
from src.mKAARMA import MKAARMACell, DiscMaker
from src.NTM.head import NTMReadHead, NTMWriteHead
from src.KAARMA import KernelNode as BaseModel
from src.kernel import Kernel

from src.tomita import generate_tomita_sequence
from src.KAARMA import KAARMA

import matplotlib.pyplot as plt

from optparse import OptionParser


parser = OptionParser()
parser.add_option("--batch_size", type="int", dest="batch_size", default=1)
parser.add_option("--device", type="str", dest="device", default="cpu")
parser.add_option("--epochs", type="int", dest="epochs", default=100000)
(options, args) = parser.parse_args()


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
# trajectories = []
trajectories = torch.nn.ModuleList()
for i in [4, 5, 6]:
    m = BaseModel(4, 1, 2, 2)
    m.load_state_dict(torch.load('model/%d.pkl' % i))
    m.requires_grad_(False)
    models.append(m)
    tra = np.load('trajectory/%d.npy' % i).astype(np.float32)
    trajectories.append(Kernel(tra.shape[0], tra.shape[1], 10, tra))
    # trajectories.append(torch.from_numpy(np.load('trajectory/%d.npy' % i).astype(np.float32)))
# trajectories = np.vstack(trajectories)
# trajectories = torch.from_numpy(trajectories)
decoder = MKAARMACell(models, trajectories).eval()

m = 20
n = 32
controller_size = 50
controller = LSTMController(19 + m, controller_size, 2)
memory = NTMMemory(n, m, None)
readhead = NTMReadHead(memory, controller_size)
writehead = NTMWriteHead(memory, controller_size)
heads = torch.nn.ModuleList([readhead, writehead])
ntm = NTM(25, 25, controller, memory, heads)


device = options.device
discmaker = DiscMaker(decoder, ntm).to(device)
for tra in discmaker.mkaarma.trajectories:
    tra.to(device)

# discmaker.load_state_dict(torch.load('model_grand_true.pkl'))

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
for i in [0, 1, 2]:
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

test_x = []
test_y = []
for i in range(3):
    _x, _y = generate_tomita_sequence(64, 128, i + 4)
    test_x.append(_x)
    test_y.append(_y)
test_x = np.vstack(test_x)
test_y = np.vstack(test_y)
test_x = torch.from_numpy(test_x.astype(np.float32)).to(device)
test_y = torch.from_numpy(test_y.astype(np.float32)).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discmaker.parameters()),
                             lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, discmaker.parameters()), lr=0.0001)
min_loss = 1
report_freq = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
# train_single_seq(discmaker, models, [0, 1, 2, 3], 100000, report_freq, criterion, optimizer)
for epoch in range(options.epochs):
    print('\r', epoch, end='', flush=True)
    x = []
    y = []
    avg_loss = []
    order = np.random.permutation([0, 1, 2])
    length = np.random.randint(10, 50)
    for j in order:
        string_x, string_y = get_data(options.batch_size, length, models[j], j + 4)
        # string_x, string_y = generate_tomita_sequence(options.batch_size, length, j + 4)
        x.append(string_x)
        y.append(string_y)
    x = np.hstack(x)
    y = np.hstack(y)
    # x = torch.from_numpy(x.astype(np.float32))
    # y = torch.from_numpy(y.astype(np.float32))
    x = torch.from_numpy(x.astype(np.float32)).to(device)
    y = torch.from_numpy(y.astype(np.float32)).to(device)
    # scheduler.step()
    _loss = train(discmaker, x, y, criterion, optimizer)
    avg_loss.append(_loss)

    if (epoch + 1) % report_freq == 0:
        print('\r', np.mean(avg_loss))

    if epoch > options.epochs - 10000:
        if (epoch + 1) % report_freq == 0:
            pred = discmaker(test_x, test_y)
            test_loss = torch.mean((test_y - pred) ** 2)
            print('\r', 'loss test: %f' % test_loss)
            if test_loss < min_loss:
                torch.save(discmaker.state_dict(), 'model_grand_true.pkl')
                min_loss = test_loss
                print('\r', 'new model saved, min_loss: %f' % test_loss)
    # scheduler.step()


for j in range(3):
    # string_x, string_y = generate_tomita_sequence(1, 100, j + 4)
    string_x, string_y = get_data(1, 100, models[j], j + 4)
    string_x = torch.from_numpy(string_x.astype(np.float32))
    string_y = torch.from_numpy(string_y.astype(np.float32))
    pred = discmaker(string_x, string_y)
    gate = torch.stack(discmaker.gate_trajectories, dim=1)
    plt.title('result grammar #%d' % (j + 4))
    plt.plot(gate.data.numpy()[0])
    plt.legend(['4', '5', '6'])
    plt.show()

# print(gate.data.numpy()[0])
torch.save(discmaker.state_dict(), 'model_real.pkl')
