import torch
from torch.autograd import Variable

xT = torch.rand(5, 3)
x = Variable(xT, requires_grad=True)

y = Variable(torch.ones(1, 3))

z = (x + y) ** 2
out = z.mean()

out.backward()
print(x.grad)
