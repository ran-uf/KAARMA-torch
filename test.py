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


def plot_ground_truth(num_gates, odr, lens):
    gs = np.zeros((num_gates, np.sum(lens)))
    s = 0
    for g, l in zip(odr, lens):
        gs[g, s: s + l] = 1
        s = s + l

    plt.title('ground truth')
    plt.plot(gs.T)
    plt.legend(['4', '5', '6'])
    plt.show()


parser = OptionParser()
parser.add_option("--batch_size", type="int", dest="batch_size", default=1)
parser.add_option("--device", type="str", dest="device", default="cpu")
parser.add_option("--epochs", type="int", dest="epochs", default=100000)
(options, args) = parser.parse_args()


def get_data(batch_size, length, m, tomita_type):
    s, _ = generate_tomita_sequence(batch_size, length, tomita_type)
    _y, _ = m(s, True)
    return s, _y.data.numpy()


models = torch.nn.ModuleList()
# trajectories = []
trajectories = torch.nn.ModuleList()
n_tra = []
for i in [4, 5, 6]:
    m = BaseModel(4, 1, 2, 2)
    m.load_state_dict(torch.load('model/n_%d.pkl' % i))
    m.requires_grad_(False)
    models.append(m)
    tra = np.load('trajectory/n_%d.npy' % i).astype(np.float32)
    trajectories.append(Kernel(tra.shape[0], tra.shape[1], 10, tra))
    n_tra.append(tra.shape[0])
    # trajectories.append(torch.from_numpy(np.load('trajectory/%d.npy' % i).astype(np.float32)))
# trajectories = np.vstack(trajectories)
# trajectories = torch.from_numpy(trajectories)
decoder = MKAARMACell(models, trajectories).eval()

n_tra = np.sum(n_tra)

m = 15
n = 48
controller_size = 100
controller = LSTMController(n_tra + 1 + m, controller_size, 2)
memory = NTMMemory(n, m, None)
readhead = NTMReadHead(memory, controller_size)
writehead = NTMWriteHead(memory, controller_size)
heads = torch.nn.ModuleList([readhead, writehead])
ntm = NTM(25, 25, controller, memory, heads)


device = options.device
discmaker = DiscMaker(decoder, ntm)
discmaker.load_state_dict(torch.load("model_1104.pkl"))
discmaker.to(device)
for tra in discmaker.mkaarma.trajectories:
    tra.to(device)
models = torch.nn.ModuleList()

for i in [0, 1, 2]:
    _m = KAARMA(4, 1, 2, 2)
    _m.node.load_state_dict(torch.load('model/n_%d.pkl' % (i + 4)))
    _m.eval()
    models.append(_m)


order = [0]
# order = np.random.permutation(order)
x = []
y = []
lengths = []
for j in order:
    length = np.random.randint(60, 100)
    lengths.append(length)
    string_x, string_y = get_data(1, length, models[j], j + 4)
    # string_x, string_y = generate_tomita_sequence(options.batch_size, length, j + 4)
    x.append(string_x)
    y.append(string_y)
x = np.hstack(x)
y = np.hstack(y)
x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32))
pred = discmaker(x, y)
gate = torch.vstack(discmaker.gate_trajectories).detach().numpy()
simlarity = torch.vstack(discmaker.similarities).detach().numpy()
error = torch.vstack(discmaker.errors).detach()

plot_ground_truth(3, order, lengths)

plt.figure()
plt.subplot(311)
plt.plot(gate)
plt.subplot(312)
plt.imshow(simlarity.T)
plt.subplot(313)
plt.plot(error)
plt.show()