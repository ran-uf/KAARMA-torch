import numpy as np
import torch

from src.mKAARMA import MKAARMACell
from src.mKAARMA import CMoE as Gate
from src.KAARMA import KernelNode as BaseModel
from src.kernel import Kernel

from src.tomita import generate_tomita_sequence, train_data, train_data_seq, test_data, test_data_seq
from src.KAARMA import KAARMA

import matplotlib.pyplot as plt

import argparse
from progressbar import progressbar


parser = argparse.ArgumentParser(description='...')
parser.add_argument("--batch_size", type=int, dest="batch_size", default=1)
parser.add_argument("--device", type=str, dest="device", default="cpu")
parser.add_argument("--epochs", type=int, dest="epochs", default=1000)
parser.add_argument('-m', '--model_list', type=str, nargs='+', action='append', help='model list', default=[4, 5, 6])
args = parser.parse_args()


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
    _pred = model(string_x)
    e = (string_y - _pred).cpu().data.numpy()
    gate = torch.stack(model.gate_trajectories, dim=1).detach().cpu()
    # similarity = torch.stack(model.similarities, dim=1).detach().cpu()
    # switches = torch.stack(model.switches, dim=1).squeeze(2).detach().cpu()

    plt.figure()
    # plt.suptitle('result grammar #%d #%d #%d' % (shuffle[0], shuffle[1], shuffle[2]))
    plt.suptitle('Conventional mixture of experts')
    plt.subplot(211)
    plt.plot(gate.data.numpy()[0])
    plt.legend(lst)
    plt.title('gate')
    # plt.subplot(212)
    # plt.title('similarity')
    # plt.imshow(similarity.T)
    # plt.yticks([0, 4])
    plt.subplot(212)
    plt.title(r'$error$')
    plt.plot(e.T)
    # plt.plot(switches.T)
    plt.tight_layout()
    plt.show()


models = torch.nn.ModuleList()
# trajectories = []
trajectories = torch.nn.ModuleList()
n_tra = []
for i in args.model_list:
    m = BaseModel(4, 1, 2, 2)
    m.load_state_dict(torch.load('model/relu_%d.pkl' % i))
    m.requires_grad_(False)
    models.append(m)
    tra = np.load('trajectory/relu_%d.npy' % i).astype(np.float32)
    trajectories.append(Kernel(tra.shape[0], tra.shape[1], 10, tra))
    n_tra.append(tra.shape[0])
    # trajectories.append(torch.from_numpy(np.load('trajectory/%d.npy' % i).astype(np.float32)))
# trajectories = np.vstack(trajectories)
# trajectories = torch.from_numpy(trajectories)
decoder = MKAARMACell(models, trajectories).eval()

model = Gate(decoder).to(args.device)
for tra in model.mkaarma.trajectories:
    tra.to(args.device)

train_x, train_y = train_data_seq(args.model_list, args.batch_size)
test_x, test_y = test_data_seq(args.model_list)
test_x = torch.from_numpy(test_x).to(args.device)
test_y = torch.from_numpy(test_y).to(args.device)
criterion = torch.nn.MSELoss()
# criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=0.0001 * args.batch_size, betas=(0.9, 0.999), eps=1e-08)
min_loss = 0.1
for epoch in range(args.epochs):
    print('\r', epoch, end='', flush=True)
    x = []
    y = []
    avg_loss = []
    idx_shuffle = np.arange(0, len(train_x))
    np.random.shuffle(idx_shuffle)

    for i in progressbar(range(len(idx_shuffle))):

        x = torch.from_numpy(train_x[idx_shuffle[i]]).to(args.device)
        y = torch.from_numpy(train_y[idx_shuffle[i]]).to(args.device)

        p = model(x)
        loss = criterion(p, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())
    print('\r', np.mean(avg_loss))

    with torch.no_grad():
        pred = model(test_x)
        test_loss = criterion(pred, test_y)
        print('\r', 'loss test: %f' % test_loss)
    # if epoch > options.epochs - 10000:
        # if (epoch + 1) % report_freq == 0:

        if test_loss < min_loss:
            torch.save(model.state_dict(), 'model_2_single_rand_state.pkl')
            min_loss = test_loss
            print('\r', 'new model saved, min_loss: %f' % test_loss)
    # scheduler.step()

    plot_seq(model, args.model_list)
