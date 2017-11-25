"""Control script for using the module"""
import pickle
import sentiment_toolkit as st
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


filename = "dataset/train_dataset.pkl"
with open(filename, "rb") as fp:
    data, labs, labels2names = pickle.load(fp)

data = data[0:100]
labs = labs[0:100]

labs = torch.LongTensor(labs).unsqueeze_(1)
n_input, n_output = data[0].size(2), int(labs.max() + 1)
n_hidden = 128
n_layers = 2

net = st.MySecondRNN(n_input, n_hidden, n_layers, n_output)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(50):
    running_loss = 0
    for iteration in range(len(data)):
        inputs = data[iteration]
        local_labels = labs[iteration]

        inputs = Variable(inputs)
        local_labels = Variable(local_labels)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, local_labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.data[0]
        if (iteration % 99 == 0) and (iteration != 0):
            print('(%d, %5d) loss: %.5e' %
                  (epoch, iteration, running_loss / 100))
            running_loss = 0.0
