import copy

import numpy as np
import torch
import torch.nn as nn


class KernelNode(nn.Module):
    def __init__(self, dim_s=None, dim_y=None, _as=None, _au=None):
        super(KernelNode, self).__init__()
        self._as = nn.Parameter(torch.from_numpy(np.array([_as], dtype=np.float32)))
        self._au = nn.Parameter(torch.from_numpy(np.array([_au], dtype=np.float32)))
        self.II = torch.zeros((dim_y, dim_s))
        self.II[:, dim_s - dim_y:] = torch.eye(dim_y)
        self.II = nn.Parameter(self.II)
        self.II.requires_grad = False
        self.initial_state = None
        self.S = None
        self.Phi = None
        self.A = None

    def initialization(self, s, phi, a):
        self.initial_state = nn.Parameter(copy.deepcopy(s))
        self.S = nn.Parameter(copy.deepcopy(s))
        self.Phi = nn.Parameter(torch.unsqueeze(phi, 0))
        self.A = nn.Parameter(torch.unsqueeze(a, 0))
        self.initial_state.requires_grad = False
        self.S.requires_grad = False
        self.Phi.requires_grad = False
        self.A.requires_grad = False

    def forward(self, inp, state):
        if state is None:
            state = self.initial_state
        new_state = torch.mm(self.A.T, torch.exp(-self._as * torch.sum((self.S - state) ** 2, dim=1, keepdim=True)) *
                             torch.exp(-self._au * torch.sum((self.Phi - inp) ** 2, dim=1, keepdim=True)))

        out = torch.mm(self.II, new_state)
        return out, new_state.T

    def update_memory(self, phi, s, a, dq):
        for (_phi, _s, _a) in zip(phi, s, a):
            dis = 1 - torch.exp(-self._as * torch.sum((self.S - _s) ** 2, dim=1, keepdim=True)) * torch.exp(-self._au * torch.sum((self.Phi - _phi) ** 2, dim=1, keepdim=True))
            index = torch.argmin(dis).detach().numpy()
            if dis[index] < dq:
                self.A.data[index] += _a.squeeze()
            else:
                self.S = nn.Parameter(torch.cat([self.S.data, _s], dim=0))
                self.Phi = nn.Parameter(torch.cat([self.Phi.data, _phi.unsqueeze(0)], dim=0))
                if _a.ndim == 1:
                    _a = _a.unsqueeze(0)
                self.A = nn.Parameter(torch.cat([self.A.data, _a], dim=0))

    def load_state_dict(self, state_dict, strict=True):
        self._au = nn.Parameter(state_dict['_au'])
        self._as = nn.Parameter(state_dict['_as'])
        self.Phi = nn.Parameter(state_dict['Phi'])
        self.II = nn.Parameter(state_dict['II'])
        self.initial_state = nn.Parameter(state_dict['initial_state'])
        self.A = nn.Parameter(state_dict['A'])
        self.S = nn.Parameter(state_dict['S'])


class KAARMA(nn.Module):
    def __init__(self, ns, ny, _as, _au):
        super(KAARMA, self).__init__()
        self._ns = ns
        self.node = KernelNode(ns, ny, _as, _au)
        np.random.seed(1)
        s = torch.from_numpy(np.random.random((1, ns))).float()
        # self._s.requires_grad = False
        temp = np.random.random(ns)
        self.node.initialization(s, torch.Tensor([0]), torch.from_numpy(temp).float())
        self.loss = torch.nn.MSELoss()

    def forward(self, x, ls=False):
        state = None
        output = None
        outputs = []
        for _x in x:
            output, state = self.node(_x, state)
            if ls:
                outputs.append(output.data.numpy())
        if ls:
            return np.array(outputs).reshape(-1)
        else:
            return output

    def custom_train(self, x, y, lr, dq):
        for (_x, _y) in zip(x, y):
            truncated_length = 6
            states_0 = [None]
            states_1 = [self.node.initial_state]
            a = []
            s = []
            phi = []
            for inp in _x:
                state = states_1[-1].detach()
                state.requires_grad = True
                phi.append(inp)
                s.append(state)
                if len(phi) > truncated_length:
                    del phi[0]
                    del s[0]
                out, new_state = self.node(inp, state)
                states_0.append(state)
                states_1.append(new_state)

            state = states_1[-1].detach()
            state.requires_grad = True

            loss = self.loss(torch.mm(self.node.II, state.T), torch.Tensor([[_y]]))

            states_0.append(state)
            states_1.append(loss)
            loss.backward(retain_graph=True)
            a.append(- lr * states_0[-1].grad)

            for i in range(len(states_0) - 3):
                if states_0[-i - 2] is None:
                    break
                curr_grad = states_0[-i-1].grad
                states_1[-i - 2].backward(curr_grad, retain_graph=True)
                a.append(- lr * states_0[-i-2].grad)
                if len(a) >= truncated_length:
                    break

            self.node.update_memory(phi, s, a[::-1], dq)

    def test(self, x, y):
        loss = []
        acc = []
        for (_x, _y) in zip(x, y):
            out = self.forward(_x)
            loss.append(((out - _y) ** 2).detach().numpy())
            acc.append(((out > 0.5) == _y).numpy())

        return np.mean(loss), np.mean(acc)


if __name__ == "__main__":
    from tomita import generate_tomita
    import time
    x_train = []
    y_train = []
    tomita_type = 5
    a = [3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    a.sort()
    for i in a:
        strings, target = generate_tomita(40, i, tomita_type)
        for xx, yy in zip(strings, target):
            x_train.append(torch.from_numpy(xx[:, np.newaxis]).float())
            y_train.append(yy)

    x_test, y_test = generate_tomita(100, 16, tomita_type)
    print("test set", np.mean(y_test))
    model = KAARMA(4, 1, 2, 2)
    # model.cuda()
    start = time.time()
    print('start training')
    for i in range(100):
        model.custom_train(x_train, y_train, 0.01, 0.3)
        print(model.node.A.shape, model.test(x_test, y_test))
    print(time.time() - start)
    torch.save(model.node.state_dict(), 'model/%d.pkl' % tomita_type)

    for i in range(50):
        x_test, y_test = generate_tomita(100, 16, tomita_type)
        print(model.test(x_test, y_test))
