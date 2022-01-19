import torch
from src.NTM.ntm import NTM
from dnc import DNC
from src.LSTMCells import LSTMCells
import numpy as np


class MKAARMACell(torch.nn.Module):
    def __init__(self, models, trajectories):
        super(MKAARMACell, self).__init__()
        self.models = models
        self.trajectories = trajectories
        self.similarity_size = 10
        self.num_cells = len(models)
        self.num_trajectories = sum([x.w.shape[0] for x in trajectories])

    def rand_state(self):
        idx = np.random.randint(0, self.num_cells)
        idx_0 = np.random.randint(0, self.trajectories[idx].w.shape[0])
        return self.trajectories[idx].w[idx_0]

    def similarity(self, s, idx):
        return self.trajectories[idx](s)

    def forward(self, phi, state):
        new_state = []
        output = []
        for idx, m in enumerate(self.models):
            _, _state = m(phi, state)
            new_state.append(_state)
            # output.append(torch.sum(self.similarity(_state, idx), dim=1, keepdim=True))
            output.append(self.similarity(_state, idx))
        new_state = torch.stack(new_state, dim=1)
        # output = self.similarity(new_state)
        output = torch.cat(output, dim=1)
        return output.view(phi.shape[0], -1), new_state


class MKAARMACellmerge(torch.nn.Module):
    def __init__(self, models, trajectories):
        super(MKAARMACellmerge, self).__init__()
        self.models = models
        self.trajectories = trajectories
        self.similarity_size = 10
        self.num_cells = len(models)
        self.num_trajectories = trajectories.w.shape[0] * self.num_cells

    def rand_state(self):
        idx = np.random.randint(0, self.num_cells)
        idx_0 = np.random.randint(0, self.trajectories[idx].w.shape[0])
        return self.trajectories[idx].w[idx_0]

    def similarity(self, s):
        return self.trajectories(s)

    def forward(self, phi, state):
        new_state = []
        output = []
        for idx, m in enumerate(self.models):
            _, _state = m(phi, state)
            new_state.append(_state)
            # output.append(torch.sum(self.similarity(_state, idx), dim=1, keepdim=True))
            output.append(self.similarity(_state))
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
        self.similarities = None
        # self.linear_encode = torch.nn.Linear(100, 20)
        self.linear_decode = torch.nn.Linear(controller.hidden_size - 1, mkaarma.num_cells)
        self.register_buffer('init_error', torch.ones(1))

        self.register_buffer('init_gate', torch.softmax(torch.ones((1, mkaarma.num_cells)), dim=1))
        # self.p_coeff = 3.14 * mkaarma.num_cells / 2 / (mkaarma.num_cells - 1)
        # self.register_buffer('init_gate', torch.softmax(torch.rand((1, 3)), dim=1))

    def rand_state(self):
        return self.mkaarma.rand_state()

    def forward(self, x, y):
        self.gate_trajectories = []
        self.similarities = []
        self.errors = []
        self.switches = []
        seq_len = x.shape[1]
        # kaarma_state = None
        kaarma_state = self.rand_state().unsqueeze(0).repeat(x.shape[0], 1)
        if type(self.controller) is NTM:
            controller_state = self.controller.create_new_state(x.shape[0], next(self.parameters()).device)
            self.controller.memory.reset(x.shape[0])
        else:
            controller_state = None

        # error = torch.rand(x.shape[0]).to(next(self.parameters()).device)
        # error = self.init_error.repeat(x.shape[0]).to(next(self.parameters()).device)
        # gate_state = self.init_gate.repeat(x.shape[0], 1).to(next(self.parameters()).device)

        error = torch.rand(x.shape[0]).to(next(self.parameters()).device)
        gate_state = torch.softmax(torch.rand(x.shape[0], self.mkaarma.num_cells), dim=1).to(next(self.parameters()).device)
        o = []
        # p = []
        for i in range(seq_len):
            encoded, new_state = self.mkaarma(x[:, i], kaarma_state)

            self.similarities.append(encoded.data)
            # kaarma_state = torch.matmul(gate_state, new_state)[:, 0, :]
            # encoded = self.linear_encode(encoded)
            # lstm
            # inp = torch.cat([encoded, error.unsqueeze(1)], dim=1).unsqueeze(1)
            # controller_output, controller_state = self.controller(inp, controller_state)
            # controller_output = controller_output.view(-1, 25)
            # NTM
            inp = torch.cat([encoded, error.unsqueeze(1)], dim=1)
            controller_output, controller_state = self.controller(inp, controller_state)
            controller_output = torch.sigmoid(controller_output)
            gate = self.linear_decode(controller_output[:, :-1])
            gate = torch.softmax(gate, dim=1)
            # theta_0 = self.normalize()
            theta = torch.sigmoid(controller_output[:, -1].unsqueeze(1))
            # theta = torch.abs(error.unsqueeze(1)) * theta
            # self.switches.append(theta.data)
            gate = gate * theta + gate_state * (1 - theta)
            self.gate_trajectories.append(gate.data)
            kaarma_state = torch.bmm(gate.unsqueeze(1), new_state)[:, 0, :]
            gate_state = gate
            pred = kaarma_state[:, -1]
            pred = torch.relu(pred) - torch.relu(pred - 1)
            error = (pred - y[:, i])  # * 0.9 + 0.1 * error
            self.errors.append(error)
            o.append(pred)
            # penalty = torch.sum(gate * (1 - gate), dim=1, keepdim=True)
            # penalty = torch.tan(self.p_coeff * penalty)
            # penalty = (1 - error ** 2) * theta
            # p.append(penalty)

        return torch.stack(o, dim=1)# , None# torch.stack(p, dim=1)


class DiscMaker2(torch.nn.Module):
    def __init__(self, mkaarma, controller):
        super(DiscMaker2, self).__init__()
        self.mkaarma = mkaarma
        self.controller = controller
        self.gate_trajectories = None
        self.similarities = None
        # self.linear_encode = torch.nn.Linear(100, 20)
        # self.linear_decoder = torch.nn.Linear(controller.hidden_size - 1, mkaarma.num_cells)
        self.linear_encoder = torch.nn.Sequential(
            torch.nn.Linear(mkaarma.num_trajectories, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8)
        )
        self.linear_decoder = torch.nn.Sequential(
            torch.nn.Linear(controller.hidden_size - 1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, mkaarma.num_cells)
        )
        self.register_buffer('init_error', torch.ones(1))

        self.register_buffer('init_gate', torch.softmax(torch.ones((1, mkaarma.num_cells)), dim=1))

    def rand_state(self):
        return self.mkaarma.rand_state()

    def forward(self, x, y):
        self.gate_trajectories = []
        self.similarities = []
        self.errors = []
        self.switches = []
        seq_len = x.shape[1]
        # kaarma_state = None
        kaarma_state = self.rand_state().unsqueeze(0).repeat(x.shape[0], 1)
        if type(self.controller) is NTM:
            controller_state = self.controller.create_new_state(x.shape[0], next(self.parameters()).device)
            self.controller.memory.reset(x.shape[0])
        else:
            controller_state = None

        # error = self.init_error.repeat(x.shape[0]).to(next(self.parameters()).device)
        # gate_state = self.init_gate.repeat(x.shape[0], 1).to(next(self.parameters()).device)

        error = torch.rand(x.shape[0]).to(next(self.parameters()).device)
        gate_state = torch.softmax(torch.rand(x.shape[0], self.mkaarma.num_cells), dim=1).to(next(self.parameters()).device)
        o = []
        p = []
        for i in range(seq_len):
            encoded, new_state = self.mkaarma(x[:, i], kaarma_state)

            self.similarities.append(encoded.data)
            # kaarma_state = torch.matmul(gate_state, new_state)[:, 0, :]
            # encoded = self.linear_encode(encoded)
            # lstm
            # inp = torch.cat([encoded, error.unsqueeze(1)], dim=1).unsqueeze(1)
            # controller_output, controller_state = self.controller(inp, controller_state)
            # controller_output = controller_output.view(-1, 25)
            # NTM

            inp = torch.cat([encoded, error.unsqueeze(1)], dim=1)
            controller_output, controller_state = self.controller(inp, controller_state)

            gate = self.linear_decoder(controller_output[:, :-1])
            gate = torch.softmax(gate, dim=1)

            theta = torch.sigmoid(controller_output[:, -1].unsqueeze(1))

            self.switches.append(theta.data)
            gate = gate * theta + gate_state * (1 - theta)
            self.gate_trajectories.append(gate.data)
            kaarma_state = torch.bmm(gate.unsqueeze(1), new_state)[:, 0, :]
            gate_state = gate
            pred = kaarma_state[:, -1]
            pred = torch.relu(pred) - torch.relu(pred - 1)
            error = (pred - y[:, i])
            self.errors.append(error)
            o.append(pred)
            penalty = (1 - error ** 2) * theta
            p.append(penalty)

        return torch.stack(o, dim=1), torch.stack(p, dim=1)


class DiscMakerDNC(torch.nn.Module):
    def __init__(self, mkaarma):
        super(DiscMakerDNC, self).__init__()
        self.mkaarma = mkaarma
        self.controller = DNC(input_size=self.mkaarma.num_trajectories + 1,
                              hidden_size=32,
                              rnn_type='lstm',
                              num_layers=2,
                              nr_cells=128,
                              cell_size=32,
                              read_heads=8,
                              batch_first=True,
                              gpu_id=-1)
        self.gate_trajectories = None
        self.similarities = None
        # self.linear_encode = torch.nn.Linear(100, 20)
        # self.linear_decoder = torch.nn.Linear(controller.hidden_size - 1, mkaarma.num_cells)
        self.linear = torch.nn.Linear(18, self.controller.hidden_size)
        self.linear_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.controller.hidden_size - 1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, mkaarma.num_cells)
        )
        self.register_buffer('init_error', torch.ones(1))

        self.register_buffer('init_gate', torch.softmax(torch.ones((1, mkaarma.num_cells)), dim=1))

        self.controller_inital = ([(0.1 * torch.rand((2, 1, 32)), 0.1 * torch.rand((2, 1, 32))),
                                   (0.1 * torch.rand((2, 1, 32)), 0.1 * torch.rand((2, 1, 32)))],
                                  None,
                                  0.1 * torch.rand((1, 256)))

    def rand_state(self):
        return self.mkaarma.rand_state()

    def forward(self, x, y):
        self.gate_trajectories = []
        self.similarities = []
        self.errors = []
        self.switches = []
        seq_len = x.shape[1]
        # kaarma_state = None
        kaarma_state = self.rand_state().unsqueeze(0).repeat(x.shape[0], 1)

        controller_state = (None, None, None)
        # controller_state = self.controller_inital
        # error = self.init_error.repeat(x.shape[0]).to(next(self.parameters()).device)
        # gate_state = self.init_gate.repeat(x.shape[0], 1).to(next(self.parameters()).device)

        error = torch.rand(x.shape[0]).to(next(self.parameters()).device)
        gate_state = torch.softmax(torch.rand(x.shape[0], self.mkaarma.num_cells), dim=1).to(next(self.parameters()).device)
        o = []
        p = []
        first = True
        for i in range(seq_len):
            encoded, new_state = self.mkaarma(x[:, i], kaarma_state)

            self.similarities.append(encoded.data)
            inp = torch.cat([encoded, error.unsqueeze(1)], dim=1).unsqueeze(1)
            controller_output, controller_state = self.controller(inp, controller_state, reset_experience=first)
            first = False
            controller_output = self.linear(controller_output)
            gate = self.linear_decoder(controller_output[:, 0, :-1])
            gate = torch.softmax(gate, dim=1)

            theta = torch.sigmoid(controller_output[:, 0, -1].unsqueeze(1))

            self.switches.append(theta.data)
            gate = gate * theta + gate_state * (1 - theta)
            self.gate_trajectories.append(gate.data)
            kaarma_state = torch.bmm(gate.unsqueeze(1), new_state)[:, 0, :]
            gate_state = gate
            pred = kaarma_state[:, -1]
            pred = torch.relu(pred) - torch.relu(pred - 1)
            error = (pred - y[:, i])
            self.errors.append(error)
            o.append(pred)

        return torch.stack(o, dim=1), None


class CMoE(torch.nn.Module):
    def __init__(self, mkaarma):
        super(CMoE, self).__init__()
        self.mkaarma = mkaarma
        self.gate_trajectories = None
        self.similarities = None
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, mkaarma.num_cells),
            torch.nn.Softmax(dim=1)
        )

    def rand_state(self):
        return self.mkaarma.rand_state()

    def forward(self, x):
        self.gate_trajectories = []
        self.similarities = []
        self.errors = []
        self.switches = []
        seq_len = x.shape[1]
        kaarma_state = None
        # kaarma_state = self.rand_state().unsqueeze(0).repeat(x.shape[0], 1)

        o = []
        for i in range(seq_len):
            encoded, new_state = self.mkaarma(x[:, i], kaarma_state)
            gate = self.gate(x[:, i].unsqueeze(1))
            self.gate_trajectories.append(gate)
            kaarma_state = torch.bmm(gate.unsqueeze(1), new_state)[:, 0, :]
            o.append(kaarma_state[:, -1])

        return torch.stack(o, dim=1)


class MMoE(torch.nn.Module):
    def __init__(self, mkaarma):
        super(MMoE, self).__init__()
        self.mkaarma = mkaarma
        self.memory_size = 20

        mstep = torch.zeros(self.memory_size, self.memory_size)
        for i in range(1, self.memory_size):
            mstep[i, i - 1] = 1

        self.register_parameter('mstep', torch.nn.Parameter(mstep, requires_grad=False))

        self.memory_encode = torch.nn.Sequential(
            torch.nn.Linear(self.memory_size * self.mkaarma.num_trajectories + self.memory_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
        )

        self.gate_linear = torch.nn.Sequential(
            torch.nn.Linear(16, self.mkaarma.num_cells),
            torch.nn.Softmax(dim=1)
        )

        self.switch = torch.nn.Sequential(
            torch.nn.Linear(self.memory_size, 1),
            torch.nn.Sigmoid(),
        )

        self.agent = LSTMCells(16, 16, 2)

        self.register_buffer('memory', None, persistent=False)

    def register_memory(self, batch_size):
        mm = torch.zeros(batch_size, self.memory_size, self.mkaarma.num_trajectories + 1)
        mm[:, :, -1] = 0.5
        self.register_buffer('memory', mm, False)

    def memory_step(self, w):
        self.memory = torch.matmul(self.mstep, self.memory)
        self.memory[:, 0, :] = w

    def forward(self, x, y):
        self.gate_trajectories = []
        self.similarities = []
        self.errors = []
        self.switches = []
        seq_len = x.shape[1]
        kaarma_state = None
        error = torch.rand(x.shape[0]).to(next(self.parameters()).device)
        gate_state = torch.softmax(torch.rand(x.shape[0], self.mkaarma.num_cells), dim=1).to(next(self.parameters()).device)
        o = []
        # p = []
        self.register_memory(x.shape[0])
        agent_state = None
        for i in range(seq_len):
            encoded, new_state = self.mkaarma(x[:, i], kaarma_state)

            # self.similarities.append(encoded.data)
            inp = torch.cat([encoded, error.unsqueeze(1)], dim=1)
            self.memory_step(inp)
            agent_input = self.memory_encode(self.memory.view(x.shape[0], -1))

            agent_output, agent_state = self.agent(agent_input, agent_state)
            gate = self.gate_linear(agent_output)
            # theta = self.switch(self.memory[:, :, -1])
            theta = torch.mean(torch.abs(self.memory[:, :10, -1]), dim=1) * 2.5
            theta = torch.relu(theta) - torch.relu(theta - 1)
            theta = theta.unsqueeze(1)
            self.switches.append(theta.data.cpu())
            gate = gate * theta + gate_state * (1 - theta)
            self.gate_trajectories.append(gate.data)

            kaarma_state = torch.bmm(gate.unsqueeze(1), new_state)[:, 0, :]
            gate_state = gate
            pred = kaarma_state[:, -1]
            pred = torch.relu(pred) - torch.relu(pred - 1)
            error = (pred - y[:, i])  # * 0.9 + 0.1 * error
            self.errors.append(error)
            o.append(pred)

        return torch.stack(o, dim=1), None


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
