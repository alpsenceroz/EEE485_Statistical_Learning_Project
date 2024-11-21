
import torch

def mse(y_pred, y_true):
    return torch.mean((y_true - y_pred) ** 2)

def mse_derivative(y_pred, y_true):
    return torch.mean(2 * (y_pred - y_true), dim=1, keepdim=True)

def cross_entropy(y_pred, y_true):
    # print("y_pred", y_pred.shape, y_pred)
    # correct_probs = torch.gather(y_pred, 1, y_true.unsqueeze(1)).squeeze()
    y_pred_clipped = torch.clamp(y_pred, min=1e-10)
    res = res = -torch.mean(torch.sum(y_true * torch.log(y_pred_clipped), dim=1))

    return res

def cross_entropy_derivative(y_pred, y_true):
    # correct_probs = torch.gather(y_pred, 1, y_true.unsqueeze(1)).squeeze()
    # correct_probs = torch.clamp(correct_probs, min=1e-10)
    y_pred_clipped = torch.clamp(y_pred, min=1e-10)

    # print("dE_do", torch.mean(-y_true / correct_probs).shape, torch.mean(-y_true / correct_probs))
    return -(y_true / y_pred_clipped) / y_pred.shape[0]
# def soft_cross_entropy(z, y_true):
#     """
#     input: n x dl
#     output: n x dl
#     """
    
#     # softmax
#     x_max = torch.max(z, dim=1, keepdim=True)[0]
#     exp_x = torch.exp(z - x_max)
#     softmax_out = exp_x / torch.sum(exp_x, dim=1, keepdim=True)
    
#     # cross entropy
#     correct_probs = torch.gather(softmax_out, 1, y_true.unsqueeze(1)).squeeze()
#     correct_probs = torch.clamp(correct_probs, min=1e-10)
#     return -torch.mean(torch.log(correct_probs))

# def soft_cross_entropy_derivative(y_pred, y_true):
#     """
#     input: n x dl
#     output: n x dl
#     """
#     correct_probs = torch.gather(y_pred, 1, y_true.unsqueeze(1)).squeeze()
#     correct_probs = torch.clamp(correct_probs, min=1e-10)
#     # print("dE_do", torch.mean(-y_true / correct_probs).shape, torch.mean(-y_true / correct_probs))
#     return torch.mean(-y_true / correct_probs)
    

class Loss:
    def __init__(self, loss_function, derivative):
        self.loss_function = loss_function
        self.derivative = derivative
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return self.loss_function(y_pred, y_true)
    
    def __call__(self, y_pred, y_true):
        if self.forward(y_pred, y_true) == 0:
            pass
        return self.forward(y_pred, y_true)
    
    def backward(self):
        return self.derivative(self.y_pred, self.y_true)

# MSE = Loss(mse, mse_derivative)
# CrossEntropy = Loss(cross_entropy, cross_entropy_derivative)

class MSE(Loss):
    def __init__(self):
        super().__init__(mse, mse_derivative)
        
class CrossEntropy(Loss):
    def __init__(self):
        super().__init__(cross_entropy, cross_entropy_derivative)

