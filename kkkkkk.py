import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from scipy.stats import truncnorm

# m = nn.Softplus()
# input = torch.randint(100, (100,)).float()
# output = m(input)
# output

# import os
# from PIL import Image
# from pillow_heif import register_heif_opener
# register_heif_opener()
# inpath = r'F:\photo'
# for i in os.listdir(inpath):
#     pth = os.path.join(inpath, i)
#     if '.MOV' in pth or '.AAE' in pth:
#         os.remove(pth)

#     img = Image.open(pth)
#     img.save(pth.replace(".HEIC", ".png"))
#     img.save(pth.replace(".HEIC", ".jpg"))

# import cv2

# pth = r'C:\Users\10696\Downloads\2023-09-27_202059.png'
# ave = r'C:\Users\10696\Downloads\2023-09-27_202059.jpg'
# def warp(height, width, star_points, src_img):
#     # 目标尺寸
#     # height = 35
#     # width = 110

#     pts1 = np.float32([[30, 0], [width - 60, 0], [0, height-30], [width - 100, height - 30]])
#     pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
#     M = cv2.getPerspectiveTransform(pts2, pts1)
#     dst = cv2.warpPerspective(src_img, M, (width, height), borderValue=(128, 128, 128))
#     return dst

# img = cv2.imread(pth)
# h, w, c = img.shape
# ret = warp(h, w, (36, 30, w - 30, h - 30), img)
# cv2.imwrite(ave, ret)

# np.tile
# X = truncnorm(-2, 2, loc=np.zeros_like(6), scale=np.ones_like(6))
# X.rvs()

# torch.bmm()

# train_index = np.vstack([
#                 np.random.permutation(6)
#                 for _ in range(10)
#             ])
# train_index

import torch
from torch import nn

# torch.randn(6, 6)

# import cv2
# import os
# from PIL import Image
# from pillow_heif import register_heif_opener
# register_heif_opener()
# inpath = r'F:\photo'
# outputpath = r'F:\video.avi'
# fps = 0.1
# sizes = (6031+1-2000, 3026-2) #w, h
# videowriter = cv2.VideoWriter(outputpath, cv2.VideoWriter_fourcc(*'XVID'), fps, sizes)
# kee = []
# for i in os.listdir(inpath):
#     pth = os.path.join(inpath, i)
#     if '.png' not in pth or r'毕业照' in pth:
#         continue
#     img = Image.open(pth).convert("RGB")
#     img = np.array(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     h, w, c = img.shape
#     if w!=6031+1-2000:
#         img = cv2.resize(img, sizes)
#         # continue
#     videowriter.write(img)
# videowriter.release()

# # network = nn.Linear(2, 2, True).requires_grad_(True)
# # network = nn.Softmax(dim=1).requires_grad_(True)

# class PolicyNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(PolicyNet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         # x = F.relu(self.fc1(x))
#         x = self.fc1(x)
#         return F.softmax(self.fc2(x), dim=1)  ## 返回该状态下，选择的动作的概率
#         # return self.fc2(x)  ## 返回该状态下，选择的动作的概率

# network = PolicyNet(2, 2, 2)
# cnt = 0
# param = np.array([[1.0, 2.0], [6.0, 3.0]])
# biast = np.array([1.0, 2.0])
# inp = np.array([2.0, 2.0]).reshape((1, -1))
# for i in network.parameters():
#     if cnt%2==0:
#         i.data = torch.from_numpy(param.T)
#         i.retain_grad = True
#     else:
#         i.data = torch.from_numpy(biast)
#         i.retain_grad = True
#     cnt += 1
# inputs = torch.tensor(inp, requires_grad=True).requires_grad_(True)
# output = network(inputs).requires_grad_(True)
# sum = torch.sum(output)
# # kk = sum.backward(retain_graph=True)
# # grad_params = 0
# # grad_bias   = 0
# # cnt = 0
# # for i in network.parameters():
# #     if cnt==0:
# #         grad_params = i.grad
# #     else:
# #         grad_bias = i.grad
# #     cnt += 1
# # inputs.retain_grad()
# # k = inputs.grad
# kl_grad = torch.autograd.grad(sum, inputs, create_graph=True)
# kl_grad_vector = torch.cat([grad.reshape(-1) for grad in kl_grad])
# vector = torch.ones_like(kl_grad_vector).requires_grad_(True)
# kl_grad_vector_product = torch.dot(kl_grad_vector, vector).requires_grad_(True)
# grad2 = torch.autograd.grad(kl_grad_vector_product, inputs, retain_graph=True)
# grad2 = torch.autograd.grad(kl_grad_vector_product, network.parameters(), retain_graph=True)

# inputs = inputs.detach().numpy()
# out = np.matmul(inputs, param)
# out = out + biast.reshape((1, -1))
# delta__ = np.ones_like(inputs)
# params_delta = np.matmul(delta__.T, inputs).T
# bias_delta = np.sum(delta__, axis = 0)

# k = 0

# # k = torch.tensor([6.0, 20.0, 200.0], requires_grad=True)
# # vector = torch.tensor([2.0, 1.0, 1.0])

# # y = torch.sum(k**2 + 2)

# # d0 = torch.autograd.grad(y, k, create_graph=True)
# # d0 = torch.concat([i.reshape(-1) for i in d0])

# # ret = torch.dot(d0, vector)
# # # ret = torch.sum(d0)

# # z = torch.autograd.grad(ret, k)

# z