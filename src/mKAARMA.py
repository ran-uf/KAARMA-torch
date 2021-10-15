import torch


class MKAARMACell(torch.nn.Module):
    def __init__(self, models, trajectories):
        super(MKAARMACell, self).__init__()
        self.models = models
        self.trajectories = trajectories
        self.similarity_size = 1
        self.n_trajectories = trajectories.shape[0]

    def similarity(self, s):
        batch_size = s.shape[0]
        num_models = s.shape[1]
        r = []
        for _s in s:
            rr = []
            for __s in _s:
                _r = (self.trajectories - __s.repeat(self.n_trajectories, 1)) ** 2
                _r = torch.sum(_r, dim=1)
                _r = torch.exp(-self.similarity_size * _r)
                rr.append(_r)
            r.append(torch.stack(rr, dim=0))
        return torch.stack(r, dim=0)

    def forward(self, phi, state):
        new_state = []
        for m in self.models:
            _, _state = m(phi, state)
            new_state.append(_state)
        new_state = torch.stack(new_state, dim=1)
        output = self.similarity(new_state)
        return output.view(phi.shape[0], -1), new_state


class DiscMaker(torch.nn.Module):
    def __init__(self, mkaarma, controller):
        super(DiscMaker, self).__init__()
        self.mkaarma = mkaarma
        self.controller = controller
        self.gate_trajectories = None

    def forward(self, x, y):
        self.gate_trajectories = []
        seq_len = x.shape[1]
        kaarma_state = None
        controller_state = self.controller.create_new_state(x.shape[0])
        self.controller.memory.reset(x.shape[0])
        error = torch.zeros(x.shape[0])
        o = []
        for i in range(seq_len):
            encoded, new_state = self.mkaarma(x[:, i], kaarma_state)
            inp = torch.cat([encoded, error.unsqueeze(1)], dim=1)
            gate, controller_state = self.controller(inp, controller_state)
            gate = torch.softmax(gate, dim=1)
            self.gate_trajectories.append(gate)
            kaarma_state = torch.matmul(gate, new_state)[:, 0, :]
            pred = kaarma_state[:, -1]
            error = pred - y[:, i]
            o.append(pred)
        return torch.stack(o, dim=1)


if __name__ == "__main__":
    from tomita import generate_tomita_sequence
    import numpy as np

    tomita_type = 4

    # model = mKAARMA(50, 4)
    # model.load("../model/%d.pkl" % tomita_type)
    # for para in model.cell.kaarma.parameters():
    #     para.requires_grad = False
    #
    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                              lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    #
    # for epoch in range(10000):
    #     x, y = generate_tomita_sequence(1, np.random.randint(3, 100), tomita_type)
    #     x = torch.from_numpy(x.astype(np.float32))
    #     y = torch.from_numpy(y.astype(np.float32))
    #
    #     optimizer.zero_grad()
    #     o = model.forward(x, None)
    #     loss = criterion(o, y)
    #     loss.backward()
    #     optimizer.step()
    #     if (epoch + 1) % 50 == 0:
    #         print(loss.data.numpy())
