"""Control script for using the module"""
import pickle
import sentiment_toolkit.model_rnn as st
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


filename = "trained_model.pkl"
with open(filename, "rb") as fp:
    net = pickle.load(fp)
    net.eval()
print(net)


sentence = ["I", "am", "really", "happy", "today"]
name, label = st.sample_network(net, sentence)
print(name, sentence)

sentence = ["I", "am", "really", "sad", "today"]
name, label = st.sample_network(net, sentence)
print(name, sentence)

sentence = ["I", "am", "worried", "we", "might", "not", "finish", "on", "time"]
name, label = st.sample_network(net, sentence)
print(name, sentence)


# ------ test score
filename = "dataset/test_dataset.pkl"
with open(filename, "rb") as fp:
    data, labs, labels2names = pickle.load(fp)


n_input, n_output = data[0].size(2), int(labs.max() + 1)
n_hidden = net.n_hidden
n_layers = net.n_layers
print(net)
accuracy, score, tries = net.test_network(data, labs)
print("Score:", score, "Tries:", tries, "Accuracy:", accuracy)
