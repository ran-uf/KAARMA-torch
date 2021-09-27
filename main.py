import torch
from KAARMA import *
from tomita import generate_tomita
import time

x_train = []
y_train = []
tomita_type = 7
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

model.node.load_state_dict(torch.load('model/%d.pkl' % tomita_type))
model.train()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)

for epoch in range(5000):
    for (x, y) in zip(x_train, y_train):
        model.train()
        pred = model.forward(x)
        loss = criterion(pred, torch.Tensor([[y]]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    print(model.test(x_test, y_test))