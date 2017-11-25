import torch
from torch.autograd import Variable
import torch.nn as nn
import cifar10
import torch.optim as optim


d, l, ln = cifar10.unpickle("cifar10/test_set.pkl")
# Define a NN module
dtest = d[9000:]
ltest = l[9000:]
d = d[0: 9000]
l = l[0: 9000]


class MyNetwork(nn.Module):
    """My neural network"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)

        self.dense1 = nn.Linear(16 * 5 * 5, 120)
        self.dense2 = nn.Linear(120, 84)
        self.dense3 = nn.Linear(84, 10)

    def forward(self, x):
        """Pass forward through the model"""
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)

        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)
        x = self.dense1(x)
        x = nn.functional.relu(x)

        x = self.dense2(x)
        x = nn.functional.relu(x)

        x = self.dense3(x)

        return x


net = MyNetwork()
print(net)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


nMini = 10
for epoch in range(7):
    running_loss = 0
    for pointer in range(0, d.size(0) // nMini):
        inputs = d[pointer * nMini: (pointer + 1) * nMini]
        labels = l[pointer * nMini: (pointer + 1) * nMini]

        inputs = Variable(inputs)
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.data[0]
        if pointer % 100 == 0:
            print('(%d, %5d) loss: %.3f' %
                  (epoch, pointer, running_loss / 100))
            running_loss = 0.0


test_outputs = net(Variable(dtest))
test_outputs = nn.functional.softmax(test_outputs)

m, am = torch.max(test_outputs, dim=1)
print("Accuracy:", torch.sum(am.data == ltest) / am.size(0))
