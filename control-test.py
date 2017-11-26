"""Control script for using the module"""
import pickle
import sentiment_toolkit as st
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


filename = "dataset/test_dataset.pkl"
with open(filename, "rb") as fp:
    data, labs, labels2names = pickle.load(fp)

data = data
labs = labs

filename = "trained_model.pkl"
with open(filename, "rb") as fp:
    net = pickle.load(fp)

n_input, n_output = data[0].size(2), int(labs.max() + 1)
n_hidden = net.n_hidden
n_layers = net.n_layers

print(net)

score = 0
tries = 0

for iteration in range(len(data)):
    inputs = data[iteration]
    local_labels = labs[iteration]

    inputs = Variable(inputs)
    local_labels = Variable(local_labels)

    outputs = nn.functional.softmax(net(inputs))
    m, am = torch.max(outputs.data, dim=1)

    result = am == local_labels.data
    score += result.numpy()[0]
    tries += 1
    
    if (iteration % 1000 == 0) and (iteration != 0):
        print("GOing through iteration", iteration)

print("Score:", score, "Tries:", tries, "Accuracy:", score / tries)
