import os

import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


class Controller(nn.Module):
    def __init__(self, base_model, input_size, hidden_size, num_layers):
        super(Controller, self).__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ex_memory = torch.nn.ModuleList()
        self.num_inputs = input_size
        self.lstm = torch.nn.LSTMCell(input_size, hidden_size, num_layers)

    def load_external_memory(self, fn):
        files = os.listdir(fn)
        for file in files:
            state_dict = torch.load(fn + '/' + file)
            node = self.base_model()
            node.load_state_dict(state_dict)
            self.ex_memory.append(node)

        for m in self.ex_memory:
            m.requires_grad_(False)

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs + self.hidden_size))
                nn.init.uniform_(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.hidden_size

    def forward(self, x, prev_state):
        inp, error, r = x
        states = []
        for m in self.ex_memory:
            o, state = m(inp, r)
            states.append(state)
        hidden_state, cell_state = self.lstm(torch.cat(states + [error], dim=1), prev_state)
        return states, hidden_state, (hidden_state, cell_state)


class NTMCell(nn.Module):
    """A Neural Turing Machine."""
    def __init__(self, num_inputs, num_outputs, controller, memory, heads):
        """Initialize the NTM.
        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`LSTMController`
        :param memory: :class:`NTMMemory`
        :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`
        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(NTMCell, self).__init__()

        # Save arguments
        # self.num_inputs = num_inputs
        # self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads

        self.N, self.M = memory.size()
        _, self.controller_size = controller.size()
        self.linear = torch.nn.Linear(self.controller_size, 4)
        # Initialize the initial previous read values to random biases
        self.num_read_heads = 0
        self.init_r = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "heads list must contain at least a single read head"

        # Initialize a fully connected layer to produce the actual output:
        #   [controller_output; previous_reads ] -> output
        self.reset_parameters()

        self.linear_gate = torch.nn.Linear(self.controller_size, 1)
        self.gate = None

    def init_states(self):
        if not next(self.parameters()).is_cuda:
            return (None, None, [torch.zeros((1, self.N))], torch.Tensor([[0, 0., 1.0, 0]])), \
                   torch.from_numpy(np.array([[0.]], dtype=np.float32))
        else:
            return (None, None, [torch.zeros((1, self.N)).cuda(), torch.zeros((1, 100)).cuda()], 0.25 * torch.ones((1, 4)).cuda()), \
                   torch.from_numpy(np.array([[0.]], dtype=np.float32)).cuda()

    def reset(self, batchsize):
        self.memory.reset(batchsize)

    def create_new_state(self, batch_size):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]

        return init_r, controller_state, heads_state

    def reset_parameters(self):
        # Initialize the linear layer
        # nn.init.xavier_uniform_(self.fc.weight, gain=1)
        # nn.init.normal_(self.fc.bias, std=0.01)
        return

    def forward(self, x, prev_state):
        """NTM forward function.
        :param x: input vector (batch_size x num_inputs)
        :param prev_state: The previous state of the NTM
        """
        # Unpack the previous state
        self.gates = []

        inp, error = x
        prev_reads, prev_controller_state, prev_heads_states, prev_gate = prev_state

        states, controller_outp, controller_state = self.controller((inp, error, prev_reads), prev_controller_state)

        # Read/Write from the list of heads
        reads = None
        heads_states = []

        gate = self.linear(controller_outp).unsqueeze(1)
        # gate = torch.softmax(gate, dim=1)
        gate = torch.softmax(gate, dim=2)
        crol = self.linear_gate(controller_outp)
        crol = torch.sigmoid(crol)
        gate = gate * crol + (1 - crol) * prev_gate
        self.gate = gate.clone()
        # print(self.gate.data)
        states = torch.cat(states, dim=0).unsqueeze(0)
        out_state = torch.bmm(gate, states).squeeze(1)
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                r, head_state = head(controller_outp, prev_head_state)
                reads = r
            else:
                head_state = head((controller_outp, out_state), prev_head_state)
            heads_states += [head_state]

        state = (reads, controller_state, heads_states, gate)

        return out_state[:, -1], state


class NTM(nn.Module):
    def __init__(self, num_inputs, num_outputs, controller, memory, heads):
        super(NTM, self).__init__()
        self.ntm_cell = NTMCell(num_inputs, num_outputs, controller, memory, heads)
        self.gate_trajectory = []

    def reset(self, batch_size):
        self.ntm_cell.reset(batch_size)

    def forward(self, x, y):
        length = x.shape[1]
        state, error = self.ntm_cell.init_states()
        o = []
        self.gate_trajectory = []
        for i in range(length):
            _x, _y = x[:, i].unsqueeze(1), y[:, i].unsqueeze(1)
            output, state = self.ntm_cell((_x, error), state)
            error = _y - output
            o.append(output)
            self.gate_trajectory.append(self.ntm_cell.gate)
        return torch.cat(o, dim=0).unsqueeze(0)

