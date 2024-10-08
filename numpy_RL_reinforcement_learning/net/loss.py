# https://zhuanlan.zhihu.com/p/642025009
import numpy as np
import torch
from torch import nn

def torch_compare_mean_square(predict:np.ndarray, label:np.ndarray):
    loss = nn.MSELoss(reduce='mean').requires_grad_(True)
    input = torch.tensor(predict, requires_grad=True)
    target = torch.tensor(label)
    output = loss(input, target)
    kk = output.backward()
    input.retain_grad()
    k = input.grad
    return output, k

def torch_compare_cross_entropy(predict:np.ndarray, label:np.ndarray):
    loss = nn.CrossEntropyLoss(reduce='mean').requires_grad_(True)
    input = torch.tensor(predict, requires_grad=True)
    target = torch.tensor(label)
    output = loss(input, target)
    kk = output.backward()
    input.retain_grad()
    k = input.grad
    return output, k

def torch_compare_binary_cross_entropy_logit_loss(predict:np.ndarray, label:np.ndarray):
    loss = nn.BCEWithLogitsLoss(reduce='mean').requires_grad_(True)
    input = torch.tensor(predict, requires_grad=True)
    target = torch.tensor(label)
    output = loss(input, target)
    kk = output.backward()
    input.retain_grad()
    k = input.grad
    return output, k

def torch_compare_binary_cross_entropy_loss(predict:np.ndarray, label:np.ndarray):
    loss = nn.BCELoss(reduce='mean').requires_grad_(True)
    input = torch.tensor(predict, requires_grad=True)
    target = torch.tensor(label)
    output = loss(input, target)
    kk = output.backward()
    input.retain_grad()
    k = input.grad
    return output, k
    
def cross_entropy_loss(predict:np.ndarray, label:np.ndarray):
    p_shift = predict - np.max(predict, axis = -1)[..., np.newaxis]   # avoid too large in exp 
    softmax = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
    loss = -np.sum(label * np.log(softmax + 1e-10)) / predict.shape[0] #avoid log(0)
    partial = (softmax - label) / predict.shape[0]
    return loss, partial, softmax

def binary_cross_entropy_logit_loss(predict:np.ndarray, label:np.ndarray):
    sigmoid = 1 / (1+np.exp(-predict))
    loss = -np.sum((label * np.log(sigmoid + 1e-10)) + ((1 - label) * np.log((1 - sigmoid) + 1e-10))) / predict.size #avoid log(0)
    partial = (sigmoid - label) / predict.size
    return loss, partial, sigmoid

def binary_cross_entropy_loss(sigmoid:np.ndarray, label:np.ndarray):
    loss = -np.sum((label * np.log(sigmoid + 1e-10)) + ((1 - label) * np.log((1 - sigmoid) + 1e-10))) / sigmoid.size #avoid log(0)
    zeros_mask = label==0
    partial = np.zeros_like(sigmoid)
    partial[~zeros_mask] = -1 / sigmoid[~zeros_mask]
    partial[zeros_mask] = 1 / (1 - sigmoid[zeros_mask])
    partial = partial / sigmoid.size
    return loss, partial, sigmoid

def mean_square_loss(predict:np.ndarray, label:np.ndarray):
    loss = np.sum(np.square(predict - label)) / predict.size
    partial = 2 * (predict - label) / predict.size
    return loss, partial

def mean_square_two_derive_loss(predict:np.ndarray, label:np.ndarray):
    loss = np.sum(np.square(predict - label)) / predict.size
    partial0 = 2 * (predict - label) / predict.size
    partial1 = 2 * (label - predict) / label.size
    return loss, partial0, partial1

if __name__=="__main__":
    predict = np.random.rand(3, 20)
    label = np.zeros((3, 20))
    label[np.arange(3), np.arange(3)] = 1

    loss_cross, partial_cross, softmax = cross_entropy_loss(predict.copy(), label)
    loss_torch_c, partial_torch_c = torch_compare_cross_entropy(predict.copy(), label)
    assert abs(loss_cross - loss_torch_c.item()) < 1e-8, (loss_cross, loss_torch_c.item())
    assert np.sum(np.abs(partial_cross - partial_torch_c.cpu().numpy())) < 1e-8, np.sum(np.abs(partial_cross - partial_torch_c.cpu().numpy()))
    
    loss_m, partial_m = mean_square_loss(predict.copy(), label)
    loss_torch_m, partial_torch_m = torch_compare_mean_square(predict.copy(), label)
    assert abs(loss_m - loss_torch_m.item()) < 1e-8, (loss_m, loss_torch_m.item())
    assert np.sum(np.abs(partial_m - partial_torch_m.cpu().numpy())) < 1e-8, np.sum(np.abs(partial_m - partial_torch_m.cpu().numpy()))

    loss_m, partial_m, sigmoid = binary_cross_entropy_logit_loss(predict.copy(), label)
    loss_torch_m, partial_torch_m = torch_compare_binary_cross_entropy_logit_loss(predict.copy(), label)
    assert abs(loss_m - loss_torch_m.item()) < 1e-8, (loss_m, loss_torch_m.item())
    assert np.sum(np.abs(partial_m - partial_torch_m.cpu().numpy())) < 1e-8, np.sum(np.abs(partial_m - partial_torch_m.cpu().numpy()))
    
    sigmoid = 1 / (1+np.exp(-predict))
    loss_m, partial_m, sigmoid = binary_cross_entropy_loss(sigmoid, label)
    loss_torch_m, partial_torch_m = torch_compare_binary_cross_entropy_loss(sigmoid, label)
    assert abs(loss_m - loss_torch_m.item()) < 1e-8, (loss_m, loss_torch_m.item())
    assert np.sum(np.abs(partial_m - partial_torch_m.cpu().numpy())) < 1e-8, np.sum(np.abs(partial_m - partial_torch_m.cpu().numpy()))
    i = 1