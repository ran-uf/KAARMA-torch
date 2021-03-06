import numpy as np
import torch

from src.NTM.memory import NTMMemory
from src.NTM.controller import LSTMController
from src.NTM.ntm import NTM
from src.mKAARMA import MKAARMACell
from src.mKAARMA import MMoE as DiscMaker
from src.NTM.head import NTMReadHead, NTMWriteHead
from src.KAARMA import KernelNode as BaseModel
from src.kernel import Kernel
from src.myLrScheduler import get_lr

from src.tomita import generate_tomita_sequence, train_data, train_data_seq, test_data, test_data_seq
from src.KAARMA import KAARMA

import matplotlib.pyplot as plt

import argparse
from progressbar import progressbar


parser = argparse.ArgumentParser(description='...')
parser.add_argument("--batch_size", type=int, dest="batch_size", default=2)
parser.add_argument("--device", type=str, dest="device", default="cpu")
parser.add_argument("--epochs", type=int, dest="epochs", default=1000)
parser.add_argument('-m', '--model_list', type=str, nargs='+', action='append', help='model list', default=[4, 5, 6])
args = parser.parse_args()


def myloss(x, y):
    return torch.mean((x - y) ** 2 * torch.pow(3, y))


def get_data(batch_size, length, m, tomita_type):
    s, _ = generate_tomita_sequence(batch_size, length, tomita_type)
    _y, _ = m(s, True)
    return s, _y.data.numpy()


def plot(model, lst):
    for j in lst:
        torch.cuda.empty_cache()
        string_x, string_y = generate_tomita_sequence(1, 100, j)
        # string_x, string_y = get_data(1, 100, models[j], j + 5)
        string_x = torch.from_numpy(string_x.astype(np.float32)).to(args.device)
        string_y = torch.from_numpy(string_y.astype(np.float32)).to(args.device)
        _ = model(string_x, string_y)
        gate = torch.stack(model.gate_trajectories, dim=1).detach().cpu()
        similarity = torch.stack(model.similarities, dim=1).detach().cpu()
        # switches = torch.stack(model.switches, dim=1).squeeze(2).detach().cpu()

        plt.figure()
        plt.suptitle('result grammar #%d' % j)
        plt.subplot(311)
        plt.plot(gate.data.numpy()[0])
        plt.legend(lst)
        plt.title('gate')
        plt.subplot(312)
        plt.title('similarity')
        plt.imshow(similarity.T)
        # plt.yticks([0, 4])
        plt.subplot(313)
        plt.title(r'$\theta$')
        # plt.plot(switches.T)
        plt.tight_layout()
        plt.show()


def plot_seq(model, lst):
    string_x, string_y = [], []
    shuffle = np.random.permutation(lst)
    for j in shuffle:
        torch.cuda.empty_cache()
        _x, _y = generate_tomita_sequence(1, 60, j)
        # string_x, string_y = get_data(1, 100, models[j], j + 5)
        string_x.append(_x)
        string_y.append(_y)

    string_x = np.hstack(string_x)
    string_y = np.hstack(string_y)
    string_x = torch.from_numpy(string_x.astype(np.float32)).to(args.device)
    string_y = torch.from_numpy(string_y.astype(np.float32)).to(args.device)
    p, _ = model(string_x, string_y)
    e = (string_y - p).cpu().data.numpy()
    gate = torch.stack(model.gate_trajectories, dim=1).detach().cpu()
    # similarity = torch.stack(model.similarities, dim=1).detach().cpu()
    # switches = torch.stack(model.switches, dim=1).squeeze(2).detach().cpu()

    plt.figure()
    plt.suptitle('Recurrent mixture of experts')
    # plt.suptitle('result grammar #%d #%d #%d' % (shuffle[0], shuffle[1], shuffle[2]))
    plt.subplot(211)
    plt.plot(gate.data.numpy()[0])
    plt.legend(lst)
    plt.title('gate')
    # plt.subplot(312)
    # plt.title('similarity')
    # plt.imshow(similarity.T)
    # plt.yticks([0, 4])
    plt.subplot(212)
    plt.title(r'$error$')
    plt.plot(e.T)
    plt.tight_layout()
    plt.show()


# def plot_seq(model, lst):
#     lst_s = np.random.permutation(lst)
#     _x, _y = [], []
#     for j in lst_s:
#         string_x, string_y = generate_tomita_sequence(1, 60, j)
#         _x.append(string_x)
#         _y.append(string_y)
#
#     _x = np.hstack(_x)
#     _y = np.hstack(_y)
#
#     _x = torch.from_numpy(_x.astype(np.float32))
#     _y = torch.from_numpy(_y.astype(np.float32))
#     _ = model(_x, _y)
#
#     gate = torch.stack(model.gate_trajectories, dim=1).detach()
#     similarity = torch.stack(model.similarities, dim=1).detach()
#     switches = torch.stack(model.switches, dim=1).squeeze(2).detach()
#
#     plt.figure()
#     plt.suptitle('result grammar %d %d %d' % (lst_s[0], lst_s[1], lst_s[2]))
#     plt.subplot(311)
#     plt.plot(gate.data.numpy()[0])
#     plt.legend(lst)
#     plt.title('gate')
#     plt.subplot(312)
#     plt.title('similarity')
#     plt.imshow(similarity.T)
#     # plt.yticks([0, 4])
#     plt.subplot(313)
#     plt.title(r'$\theta$')
#     plt.plot(switches.T)
#     plt.tight_layout()
#     plt.show()


def train(model, train_x, train_y, cri, optim):
    optim.zero_grad()
    pred, penalty = model(train_x, train_y)
    loss = cri(pred, train_y) # + 0.0001 * torch.mean(penalty)
    loss.backward()
    optim.step()
    return loss.cpu().data.numpy()


models = torch.nn.ModuleList()
# trajectories = []
trajectories = torch.nn.ModuleList()
n_tra = []
for i in args.model_list:
    m = BaseModel(4, 1, 2, 2)
    m.load_state_dict(torch.load('model/nn_%d.pkl' % i))
    m.requires_grad_(False)
    models.append(m)
    tra = np.load('trajectory/nn_%d.npy' % i).astype(np.float32)
    trajectories.append(Kernel(tra.shape[0], tra.shape[1], 10, tra))
    n_tra.append(tra.shape[0])
    # trajectories.append(torch.from_numpy(np.load('trajectory/%d.npy' % i).astype(np.float32)))
# trajectories = np.vstack(trajectories)
# trajectories = torch.from_numpy(trajectories)
decoder = MKAARMACell(models, trajectories).eval()

n_tra = np.sum(n_tra)
m = n_tra + 1
n = 128

# 1
controller_size = 100
controller = LSTMController((n_tra + 1 + m * 2), controller_size, 3)
# controller = LSTMController(8 + 1 + 9, controller_size, 3)

# controller = LSTMController(3 + 1 + m, controller_size, 2)
memory = NTMMemory(n, m, None)
# memory = NTMMemory(n, 9, None)
readhead_1 = NTMReadHead(memory, controller_size)
readhead_2 = NTMReadHead(memory, controller_size)
writehead = NTMWriteHead(memory, controller_size)
heads = torch.nn.ModuleList([readhead_1, readhead_2, writehead])
ntm = NTM(25, 25, controller, memory, heads)

lstm = torch.nn.LSTM(22, 25, 2)
# discmaker = DiscMaker(decoder, ntm).to(args.device)
discmaker = DiscMaker(decoder).to(args.device)

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
    tra.to(args.device)

# models = torch.nn.ModuleList()
# for i in [1, 2]:
#     _m = KAARMA(4, 1, 2, 2)
#     _m.node.load_state_dict(torch.load('model/n_%d.pkl' % (i + 4)))
#     _m.eval()
#     models.append(_m)


train_x, train_y = train_data_seq(args.model_list, args.batch_size, 4096)
test_x, test_y = test_data_seq(args.model_list)
test_x = torch.from_numpy(test_x).to(args.device)
test_y = torch.from_numpy(test_y).to(args.device)
criterion = torch.nn.MSELoss()
# criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discmaker.parameters()),
                             lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, discmaker.parameters()), lr=0.001)
min_loss = 0.07
report_freq = 100
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
# train_single_seq(discmaker, models, [0, 1, 2, 3], 100000, report_freq, criterion, optimizer)
for epoch in range(args.epochs):

    print('epoch:', epoch, ' lr:', scheduler.get_last_lr()[0])
    x = []
    y = []
    avg_loss = []
    idx_shuffle = np.arange(0, len(train_x))
    np.random.shuffle(idx_shuffle)

    for i in progressbar(range(len(idx_shuffle))):
    # for idx in idx_shuffle:
    #     x = train_x[idx]
    #     y = train_y[idx]
        x = torch.from_numpy(train_x[idx_shuffle[i]]).to(args.device)
        y = torch.from_numpy(train_y[idx_shuffle[i]]).to(args.device)
        _loss = train(discmaker, x, y, criterion, optimizer)
        avg_loss.append(_loss)
    print('\r', np.mean(avg_loss))

    with torch.no_grad():
        pred, _ = discmaker(test_x, test_y)
        test_loss = criterion(pred, test_y)
        print('\r', 'loss test: %f' % test_loss)
    # if epoch > options.epochs - 10000:
        # if (epoch + 1) % report_freq == 0:

        if test_loss < min_loss:
            torch.save(discmaker.state_dict(), './MMoE/MMOE_%03d.pkl' % (min_loss * 1000))
            min_loss = test_loss
            print('\r', 'new model saved, min_loss: %f' % test_loss)
    # # scheduler.step()

    plot_seq(discmaker, args.model_list)

    scheduler.step()

