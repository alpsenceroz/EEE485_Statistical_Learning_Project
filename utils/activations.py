import torch

# ReLU
def relu(x):
    return torch.maximum(torch.tensor(0), x)

def relu_derivative(x):
    return torch.sum(torch.where(x > 0, torch.tensor(1), torch.tensor(0)), dim=0).unsqueeze(1)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Tanh
def tanh(x):
    return torch.tanh(x)

def tanh_derivative(x):
    return 1 - torch.tanh(x) ** 2

# Softmax
def softmax(x):
    # TODO: fix the nan value error caused by the exp function
    # TODO: can i improve the backpropagation calculation speed by storing the output
    # Subtract max for numerical stability
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

def softmax_derivative(x):
    return torch.sum(softmax(x) * (1 - softmax(x)), dim=0).unsqueeze(1)


# activation objects
class Activation:
    def __init__(self, activation, derivative):
        self.input = None
        self.output = None
        self.activation = activation
        self.derivative = derivative
    
    def forward(self, x):
        self.input = x
        self.output = self.activation(x)
        return self.output
    
    def backward(self, dE_dX, lr):
        print(dE_dX.shape, self.derivative(self.input).shape)
        return self.derivative(self.input) * dE_dX # element-wise multiplication
    
    def __call__(self, x):
        return self.forward(x)

ReLU = Activation(relu, relu_derivative)
Sigmoid = Activation(sigmoid, sigmoid_derivative)
Softmax = Activation(softmax, softmax_derivative)
Tanh = Activation(tanh, tanh_derivative)


# class ReLU(Activation):
#     def __init__(self):
#         super().__init__(relu, relu_derivative)

# class Sigmoid(Activation):
#     def __init__(self):
#         super().__init__(sigmoid, sigmoid_derivative)

# class Softmax(Activation):
#     def __init__(self):
#         super().__init__(softmax, softmax_derivative)

# class Tanh(Activation):
#     def __init__(self):
#         super().__init__(tanh, tanh_derivative)



