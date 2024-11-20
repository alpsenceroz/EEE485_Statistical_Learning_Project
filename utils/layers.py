import torch


class Linear:
    def __init__(self, in_features, out_features):
        self.weights = torch.randn(out_features, in_features)
        self.bias = torch.randn(out_features)
        self.bias = torch.unsqueeze(self.bias, 1)
        self.input = None
        self.weights.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        self.input = x
        return torch.matmul(x, self.weights.T) + self.bias.T
    
    def __call__(self, x):
        return self.forward(x)
    
    def to(self, device):
        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)

    def backward(self, dL_dV, lr):
        print(dL_dV.shape, self.weights.shape, self.bias.shape, self.input.shape)
        dL_dX = (dL_dV.T @ self.weights).T
        print(dL_dX.shape)
        print(self.input.unsqueeze(2).expand(self.input.shape[0], self.input.shape[1], self.weights.shape[0]).shape)
        dL_dW = torch.sum(self.input.unsqueeze(2).expand(self.input.shape[0], self.input.shape[1], self.weights.shape[0]), dim=0).T * dL_dV
        print(dL_dW.shape)
        dL_dW = torch.sum(dL_dW, axis=0)
        dL_dB = dL_dV
        self.weights -= lr * dL_dW
        self.bias -= lr * dL_dB
        return dL_dX # TODO: should I return dL_dW, dL_dB?