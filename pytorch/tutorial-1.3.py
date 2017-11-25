import torch
import torch.nn as nn
from torch.autograd import Variable 

dtype = torch.FloatTensor

N, nF, nH, nOut = 64, 100, 50, 10

x = Variable(torch.rand(N, nF).type(dtype))
y = Variable(torch.rand(N, nOut).type(dtype))

w1 = Variable(torch.rand(nF, nH), requires_grad=True)
w2 = Variable(torch.rand(nH, nOut), requires_grad=True)

learning_rate = 1e-8
cum_loss = 0

for i in range(500):
    h = torch.matmul(x, w1)
    h = nn.functional.relu(h)

    o = torch.matmul(h, w2)
    o = nn.functional.relu(o)

    loss = torch.pow(o - y, 2)
    loss = torch.sum(loss)

    loss.backward()

    w1.data -= w1.grad.data * learning_rate
    w2.data -= w2.grad.data * learning_rate

    w1.grad.data.zero_()
    w2.grad.data.zero_()

    cum_loss += loss.data[0]
    if (i % 10 == 0):
        print("(%d) loss: %f.3" % (i, cum_loss / 10))
        cum_loss = 0
