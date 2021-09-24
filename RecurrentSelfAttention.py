import numpy as np
import torch
import torch.nn as nn


class RecurrentAttentionLayer(nn.Module):
    def __init__(self, dim_s, dim_y, size_m):
        super(RecurrentAttentionLayer, self).__init__()
        self.II = torch.zeros((dim_y, dim_s))
        self.II[:, dim_s - dim_y:] = torch.eye(dim_y)
        self.II = nn.Parameter(self.II)
        self.II.requires_grad = False
        self.K = nn.Parameter(torch.rand((dim_s + 1, size_m)))
        self.A = nn.Parameter(torch.rand((dim_s, size_m))) * 0.5

        self.softmax = nn.Softmax(dim=0)
        self.linear = nn.Linear(4, 1)
        self.tanh = nn.Tanh()

    def forward(self, inp, state):
        new_inp = torch.cat([inp, state], dim=0)

        new_state = torch.mm(self.K.T, new_inp) / 2
        new_state = self.softmax(new_state)
        new_state = torch.mm(self.A, new_state)
        # out = torch.mm(self.II, new_state)
        out = self.linear(new_state.T)
        out = self.tanh(out)
        return out, new_state


class Model:
    def __init__(self, dim_s, dim_y, size_m):
        super(Model, self).__init__()
        self._s = torch.from_numpy(np.random.random((dim_s, 1))).float()
        self._s.requires_grad = True
        self.layer = RecurrentAttentionLayer(dim_s, dim_y, size_m)
        self.loss = nn.MSELoss()

    def train(self, x, y, optimizer):
        for (_x, _y) in zip(x, y):
            # truncated_length = 6
            # states_0 = [None]
            # states_1 = [self._s]

            state = self._s
            for inp in _x:
                inp = torch.from_numpy(np.array([[inp]], dtype=np.float32))
                out, state = self.layer(inp, state)
                # states_0.append(state)
                # states_1.append(new_state)
                # if len(states_0) > truncated_length:
                #     del states_0[0]
                #     del states_1[0]
            loss = self.loss(torch.mm(self.layer.II, state), torch.Tensor([[_y]]))
            # loss.detach_()

            # self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

    def test(self, x, y):
        loss = []
        for (_x, _y) in zip(x, y):
            state = self._s
            for inp in _x:
                inp = torch.from_numpy(np.array([[inp]], dtype=np.float32))
                out, state = self.layer(inp, state)
            _loss = self.loss(torch.mm(self.layer.II, state), torch.Tensor([[_y]]))
            loss.append(_loss.data)
        # print(loss)
        return np.mean(loss)

    def predict(self, x):
        out = None
        for _x in x:
            state = self._s
            for inp in _x:
                inp = torch.from_numpy(np.array([[inp]], dtype=np.float32))
                out, state = self.layer(inp, state)
        return out


if __name__ == "__main__":
    from tomita import generate_tomita
    import time
    x_train = []
    y_train = []
    tomita_type = 4
    a = [3, 3, 4, 5, 6, 7, 8, 9, 10]
    a.sort()
    for i in a:
        strings, target = generate_tomita(40, i, tomita_type)
        for xx, yy in zip(strings, target):
            x_train.append(torch.from_numpy(xx[:, np.newaxis]).float())
            y_train.append(yy)

    x_test, y_test = generate_tomita(100, 40, tomita_type)

    model = Model(4, 1, 150)

    start = time.time()
    print('start training')
    optimizer = torch.optim.Adam(model.layer.parameters(), lr=0.1)
    for i in range(2000):
        model.train(x_train, y_train, optimizer)
        if i % 1 == 0:
            print(model.test(x_train, y_train))
        # print(model.test(x_test, y_test))
    print(time.time() - start)
