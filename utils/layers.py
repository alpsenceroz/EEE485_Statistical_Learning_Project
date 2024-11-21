import torch


class Linear:
    def __init__(self, in_features, out_features):
        """
        input = n x dl
        weights = dl x dh
        bias = 1 x dh
        output = n x dh
        """
        # self.weights = torch.randn(out_features, in_features)
        self.weights = torch.randn(in_features, out_features)
        self.bias = torch.randn(out_features)
        # self.bias = torch.unsqueeze(self.bias, 1)
        self.bias = torch.unsqueeze(self.bias, 0)
        self.input = None
        
        self.weights.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        self.input = x
        return x @ self.weights + self.bias
    
    def __call__(self, x):
        return self.forward(x)
    
    def to(self, device):
        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)

    def backward(self, dL_dV, lr):
        """
        input = n x dl
        dL_dV = n x dh
        self.weights = dl x dh
        self.bias = 1 x dh
        self.input = n x dl
        """
        # print(dL_dV.shape, self.weights.shape, self.bias.shape, self.input.shape)
        # dL_dX = (dL_dV.T @ self.weights).T
        # print(dL_dX.shape)
        # print(self.input.unsqueeze(2).expand(self.input.shape[0], self.input.shape[1], self.weights.shape[0]).shape)
        # dL_dW = torch.sum(self.input.unsqueeze(2).expand(self.input.shape[0], self.input.shape[1], self.weights.shape[0]), dim=0).T * dL_dV
        # print(dL_dW.shape)
        # dL_dW = torch.sum(dL_dW, axis=0)
        # dL_dB = dL_dV
        dL_dX = dL_dV @ self.weights.T # (n x dh) @ (dh x dl) = n x dl
        dL_dW = self.input.T @ dL_dV # (dl x n) @ (n x dh) = dl x dh
        dL_dB = torch.sum(dL_dV, dim=0, keepdim=True) # (n x dh) -> 1 x dh
        
        self.weights -= lr * dL_dW
        self.bias -= lr * dL_dB
        return dL_dX # TODO: should I return dL_dW, dL_dB?