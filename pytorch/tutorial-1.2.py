import torch
from torch.autograd import Variable
import torch.nn as nn


# Define a NN module
class MyNetwork(nn.Module):
    """My neural network"""

    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(10, 10)
        self.dense2 = nn.Linear(10, 2)

    def forward(self, x):
        """Pass forward through the model"""
        x = self.dense1(x)
        x = nn.functional.relu(x)

        x = self.dense2(x)
        x = nn.functional.softmax(x)


net = MyNetwork()
print(net)
