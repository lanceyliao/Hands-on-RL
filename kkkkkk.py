import torch
from torch import nn
import numpy as np

network = nn.Linear(2, 2, True).requires_grad_(True)
cnt = 0
param = np.array([[1.0, 2.0], [6.0, 3.0]])
biast = np.array([1.0, 2.0])
inputs = np.array([2.0, 2.0]).reshape((1, -1))
for i in network.parameters():
    if cnt==0:
        i.data = torch.from_numpy(param.T)
        i.retain_grad = True
    else:
        i.data = torch.from_numpy(biast)
        i.retain_grad = True
    cnt += 1
inputs = torch.tensor(inputs, requires_grad=True)
output = network(inputs)
sum = torch.sum(output)
# kk = sum.backward(retain_graph=True)
# grad_params = 0
# grad_bias   = 0
# cnt = 0
# for i in network.parameters():
#     if cnt==0:
#         grad_params = i.grad
#     else:
#         grad_bias = i.grad
#     cnt += 1
# inputs.retain_grad()
# k = inputs.grad
kl_grad = torch.autograd.grad(sum, network.parameters(), create_graph=True)
kl_grad_vector = torch.cat([grad.reshape(-1) for grad in kl_grad])
vector = torch.ones_like(kl_grad_vector)
kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
grad2 = torch.autograd.grad(kl_grad_vector_product, network.parameters())

inputs = inputs.detach().numpy()
out = np.matmul(inputs, param)
out = out + biast.reshape((1, -1))
delta__ = np.ones_like(inputs)
params_delta = np.matmul(delta__.T, inputs).T
bias_delta = np.sum(delta__, axis = 0)

k = 0

# k = torch.tensor([6.0, 20.0, 200.0], requires_grad=True)
# vector = torch.tensor([2.0, 1.0, 1.0])

# y = torch.sum(k**2 + 2)

# d0 = torch.autograd.grad(y, k, create_graph=True)
# d0 = torch.concat([i.reshape(-1) for i in d0])

# ret = torch.dot(d0, vector)
# # ret = torch.sum(d0)

# z = torch.autograd.grad(ret, k)

z