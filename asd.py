from utils.activations import softmax, softmax_derivative
import numpy as np
import torch

# x = torch.zeros((1, 3))
# x[0, 1] = float('-inf')
# x[0, 2] = float('-inf')


# y = torch.zeros((1, 3))
# y[0, 0] = 1
# print(softmax(x))
# print(softmax_derivative(x))

# print(torch.bmm(softmax_derivative(x), y.unsqueeze(2)).squeeze(2))


# n = np.array([[-31.2927,  44.8574, 232.7094, 124.1920, -66.5104,  44.0910,  37.0332,
#           88.6114, 150.5141, -25.9089]])

# t = torch.tensor(n)


# x_max = torch.max(t, dim=1, keepdim=True)[0]
# exp_x = torch.exp(t-x_max)
# res = exp_x / torch.sum(exp_x, dim=1, keepdim=True)
# print(res)


a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
a = torch.nn.functional.one_hot(a, num_classes=10)
print(a)
print(torch.argmax(a, dim=1))