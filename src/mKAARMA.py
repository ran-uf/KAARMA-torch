import torch


class MKAARMACell(torch.nn.Module):
    def __init__(self, models, trajectories):
        super(MKAARMACell, self).__init__()
        self.models = models
        self.trajectories = trajectories
        self.similarity_size = 10
        # self.n_trajectories = trajectories.shape[0]

    def similarity(self, s, idx):
        batch_size = s.shape[0]
        r = []
        # for _s in s:
        #     _r = (self.trajectories[idx] - _s.repeat(self.trajectories[idx].shape[0], 1)) ** 2
        #     _r = torch.sum(_r, dim=1)
        #     _r = torch.exp(-self.similarity_size * _r)
        #     r.append(_r)
        # return torch.stack(r, dim=0)
        return self.trajectories[idx](s)

    def forward(self, phi, state):
        new_state = []
        output = []
        for idx, m in enumerate(self.models):
            _, _state = m(phi, state)
            new_state.append(_state)
            output.append(self.similarity(_state, idx))
        new_state = torch.stack(new_state, dim=1)
        # output = self.similarity(new_state)
        output = torch.cat(output, dim=1)
        return output.view(phi.shape[0], -1), new_state


class DiscMaker(torch.nn.Module):
    def __init__(self, mkaarma, controller):
        super(DiscMaker, self).__init__()
        self.mkaarma = mkaarma
        self.controller = controller
        self.gate_trajectories = None
        # self.linear_encode = torch.nn.Linear(100, 20)
        self.linear_decode = torch.nn.Linear(controller.num_outputs - 1, 3)
        self.register_buffer('init_error', torch.zeros(1))
        self.register_buffer('init_gate', torch.Tensor([[0.33, 0.33, 0.34]]))

    def forward(self, x, y):
        self.gate_trajectories = []
        seq_len = x.shape[1]
        kaarma_state = None
        controller_state = self.controller.create_new_state(x.shape[0], next(self.parameters()).device)
        self.controller.memory.reset(x.shape[0])

        error = self.init_error.repeat(x.shape[0])
        o = []
        # gate_state = self.init_gate.repeat(x.shape[0], 1)
        gate_state = torch.randn((x.shape[0], 3))
        gate_state = torch.softmax(gate_state, dim=1)
        # gate_state = torch.Tensor([[0.25, 0.25, 0.25, 0.25]]).repeat(x.shape[0], 1)
        for i in range(seq_len):
            encoded, new_state = self.mkaarma(x[:, i], kaarma_state)
            # kaarma_state = torch.matmul(gate_state, new_state)[:, 0, :]
            # encoded = self.linear_encode(encoded)
            inp = torch.cat([encoded, error.unsqueeze(1)], dim=1)
            controller_output, controller_state = self.controller(inp, controller_state)
            gate = self.linear_decode(controller_output[:, :-1])
            # gate = gate * 100
            gate = torch.softmax(gate, dim=1)
            theta = torch.sigmoid(controller_output[:, -1]).unsqueeze(1)
            gate = gate * theta + gate_state * (1 - theta)
            self.gate_trajectories.append(gate)
            kaarma_state = torch.bmm(gate.unsqueeze(1), new_state)[:, 0, :]
            gate_state = gate
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
