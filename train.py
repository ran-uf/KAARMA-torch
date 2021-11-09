import numpy as np
import torch

from src.NTM.memory import NTMMemory
from src.NTM.controller import LSTMController
from src.NTM.ntm import NTM
from src.mKAARMA import MKAARMACell, DiscMaker, DiscMaker2
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
parser.add_option("--epochs", type="int", dest="epochs", default=1000)
(options, args) = parser.parse_args()
device = options.device


def myloss(x, y):
    return torch.mean((x - y) ** 2 * torch.pow(3, y))


def get_data(batch_size, length, m, tomita_type):
    s, _ = generate_tomita_sequence(batch_size, length, tomita_type)
    _y, _ = m(s, True)
    return s, _y.data.numpy()


def train(model, train_x, train_y, cri, optim):
    optim.zero_grad()
    pred, penalty = model(train_x, train_y)
    loss = cri(train_y, pred)  # + 0.0001 * torch.mean(penalty[:, 5:])
    loss.backward()
    optim.step()
    return loss.cpu().data.numpy()


def test_data(grammars, dev):
    t_x = []
    t_y = []
    for g in grammars:
        _x, _y = generate_tomita_sequence(64, 128, g)
        t_x.append(_x)
        t_y.append(_y)
    t_x = np.vstack(t_x)
    t_y = np.vstack(t_y)
    t_x = torch.from_numpy(t_x.astype(np.float32)).to(dev)
    t_y = torch.from_numpy(t_y.astype(np.float32)).to(dev)
    return t_x, t_y


def train_data(grammars, batch_size, dev):
    t_x = []
    t_y = []
    for _ in range(int(2000 / batch_size)):
        for g in grammars:
            lgh = np.random.randint(10, 80)
            _x, _y = generate_tomita_sequence(batch_size, lgh, g)
            # string_x, string_y = generate_tomita_sequence(options.batch_size, length, j + 4)
            t_x.append(torch.from_numpy(_x.astype(np.float32)).to(dev))
            t_y.append(torch.from_numpy(_y.astype(np.float32)).to(dev))

    return t_x, t_y


models = torch.nn.ModuleList()
# trajectories = []
trajectories = torch.nn.ModuleList()
n_tra = []
for i in [5, 6]:
    m = BaseModel(4, 1, 2, 2)
    m.load_state_dict(torch.load('model/n_%d.pkl' % i))
    m.requires_grad_(False)
    models.append(m)
    tra = np.load('trajectory/n_%d.npy' % i).astype(np.float32)
    trajectories.append(Kernel(tra.shape[0], tra.shape[1], 20, tra))
    n_tra.append(tra.shape[0])
    # trajectories.append(torch.from_numpy(np.load('trajectory/%d.npy' % i).astype(np.float32)))
# trajectories = np.vstack(trajectories)
# trajectories = torch.from_numpy(trajectories)
decoder = MKAARMACell(models, trajectories).eval()

n_tra = np.sum(n_tra)
m = n_tra + 1
n = 32

# 1
controller_size = 100
controller = LSTMController(n_tra + 1 + m, controller_size, 3)
# controller = LSTMController(3 + 1 + m, controller_size, 2)
memory = NTMMemory(n, m, None)
readhead = NTMReadHead(memory, controller_size)
writehead = NTMWriteHead(memory, controller_size)
heads = torch.nn.ModuleList([readhead, writehead])
ntm = NTM(25, 25, controller, memory, heads)

lstm = torch.nn.LSTM(25, 50, 2)
discmaker = DiscMaker(decoder, ntm).to(device)

# 2
# controller_size = 100
# controller = LSTMController(n_tra + m, controller_size, 2)
# # controller = LSTMController(3 + 1 + m, controller_size, 2)
# memory = NTMMemory(n, m, None)
# readhead = NTMReadHead(memory, controller_size)
# writehead = NTMWriteHead(memory, controller_size)
# heads = torch.nn.ModuleList([readhead, writehead])
# ntm = NTM(25, 25, controller, memory, heads)
# discmaker = DiscMaker2(decoder, ntm).to(device)


for tra in discmaker.mkaarma.trajectories:
    tra.to(device)

# models = torch.nn.ModuleList()
# for i in [1, 2]:
#     _m = KAARMA(4, 1, 2, 2)
#     _m.node.load_state_dict(torch.load('model/n_%d.pkl' % (i + 4)))
#     _m.eval()
#     models.append(_m)


train_x, train_y = train_data([5, 6], options.batch_size, options.device)
test_x, test_y = test_data([5, 6], options.device)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discmaker.parameters()),
                             lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
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
    idx_shuffle = np.arange(0, len(train_x))
    np.random.shuffle(idx_shuffle)

    for idx in idx_shuffle:
        x = train_x[idx]
        y = train_y[idx]
        _loss = train(discmaker, x, y, criterion, optimizer)
        avg_loss.append(_loss)
    print('\r', np.mean(avg_loss))
    pred, _ = discmaker(test_x, test_y)
    test_loss = criterion(test_y, pred)
    print('\r', 'loss test: %f' % test_loss)
    # if epoch > options.epochs - 10000:
        # if (epoch + 1) % report_freq == 0:

    if test_loss < min_loss:
        torch.save(discmaker.state_dict(), 'model_grand_true.pkl')
        min_loss = test_loss
        print('\r', 'new model saved, min_loss: %f' % test_loss)
    scheduler.step()


for j in range(2):
    # string_x, string_y = generate_tomita_sequence(1, 100, j + 4)
    string_x, string_y = get_data(1, 100, models[j], j + 5)
    string_x = torch.from_numpy(string_x.astype(np.float32))
    string_y = torch.from_numpy(string_y.astype(np.float32))
    pred = discmaker(string_x, string_y)
    gate = torch.stack(discmaker.gate_trajectories, dim=1).detach()
    similarity = torch.stack(discmaker.similarities, dim=1).detach()
    switches = torch.stack(discmaker.switches, dim=1).squeeze(2).detach()

    plt.figure()
    plt.suptitle('result grammar #%d' % (j + 5))
    plt.subplot(311)
    plt.plot(gate.data.numpy()[0])
    plt.legend(['5', '6'])
    plt.title('gate')
    plt.subplot(312)
    plt.title('similarity')
    plt.imshow(similarity.T)
    plt.yticks([0, 4])
    plt.subplot(313)
    plt.title(r'$\theta$')
    plt.plot(switches.T)
    plt.tight_layout()
    plt.show()

# print(gate.data.numpy()[0])
torch.save(discmaker.state_dict(), 'model_real.pkl')
