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
        # print (self.input.unsqueeze(2).expand(self.input.shape[0], self.input.shape[1], self.weights.shape[0]).shape)
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

# https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
def pad(X, padding=0):
    # if len(X.shape) == 2: # H x W
    row, cols = X.shape[-2], X.shape[-1]
    padded_matrix = torch.zeros((*X.shape[:-2], row+2*padding, cols+2*padding), dtype=X.dtype)
    padded_matrix[..., padding:padding+row, padding:padding+cols] = X
    return padded_matrix
    # elif len(X.shape) == 4: # B x C x H x W
    #     row, cols = X.shape[-2], X.shape[-1]
    #     padded_matrix = torch.zeros((X.shape[0], X.shape[1], row+2*padding, cols+2*padding), dtype=X.dtype)
    #     padded_matrix[:, :, padding:padding+row, padding:padding+cols] = X
    #     return padded_matrix
    # elif len(X.shape) == 3: #C x H x W
    #     row, cols = X.shape[-2], X.shape[-1]
    #     padded_matrix = torch.zeros((X.shape[0], row+2*padding, cols+2*padding), dtype=X.dtype)
    #     padded_matrix[:, padding:padding+row, padding:padding+cols] = X
    #     return padded_matrix


def dilate(X, dilation_size):
    # Get the shape of the input matrix
    
    rows, cols = X.shape[-2], X.shape[-1]
    
    # Calculate the new shape after dilation
    new_rows = rows + (rows - 1) * dilation_size
    new_cols = cols + (cols - 1) * dilation_size
    
    # Create a larger tensor filled with zeros
    dilated_matrix = torch.zeros((X.shape[0], new_rows, new_cols), dtype=X.dtype)
    
    # Fill in the original values at appropriate positions
    dilated_matrix[:, ::dilation_size + 1, ::dilation_size + 1] = X
    
    
    return dilated_matrix

def corr_2d(X:torch.Tensor, K:torch.Tensor, corr_type="valid", stride=1):
    # TODO: is it proper to include stride for this method?
    if corr_type == "valid":
        X_pad= X.clone()
    elif corr_type == "same":
        X_pad = pad(X, K.shape[-1]//2)
    elif corr_type == "full":
        X_pad = pad(X, K.shape[-1]-1)
        
        
    
    out_rows = (X_pad.shape[-2] - K.shape[-2]) // stride + 1
    out_cols = (X_pad.shape[-1] - K.shape[-1]) // stride + 1
    
    out = torch.zeros((out_rows, out_cols), dtype=X.dtype)
    
    for j in range(out_cols):
        for i in range(out_rows):
            window = X_pad[..., i*stride:i*stride+K.shape[-2], j*stride:j*stride+K.shape[-1]]
            out[i, j] = (window * K).sum()
    
    return out
    
    
def conv_2d(X, K, conv_type="valid"):
    K_rotated = torch.flip(K, [-2,-1]) # rotate the kernel 180 degrees
    return corr_2d(X, K_rotated, conv_type)

    
class Conv:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.kernels = torch.randn(out_channels, in_channels, kernel_size, kernel_size) # C_out x C_in x K X K
        # Bias should be broadcastable to output feature maps
        self.biases = torch.randn(out_channels, 1, 1) # C_out x 1 x 1
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.input = None
        self.kernels.requires_grad = False
        self.biases.requires_grad = False

    def forward(self, X): # B x C x H x W
        self.input = X
        
        self.padded_input = X
        
        if self.padding != 0:
            self.padded_input = pad(X, self.padding)
        
        outputs = []
        # Takes some time
        for X_i_padded in self.padded_input:
            X_outputs = []
            
            for kernel, bias in zip(self.kernels, self.biases):
                correlated = corr_2d(X=X_i_padded, K=kernel, stride=self.stride)
                bias_added = correlated + bias
                X_outputs.append(bias_added)
            
            X_output = torch.stack(X_outputs)
            outputs.append(X_output)
            
        output = torch.stack(outputs)
        
        return output
        
        
    def __call__(self, x):
        return self.forward(x)


    # https://www.youtube.com/watch?v=Lakz2MoHy6o
    def backward(self, dL_dV:torch.Tensor, lr):
        dL_dV_K = dL_dV.clone()
        dL_dV_X = dL_dV.clone()
        
        dL_dK = torch.zeros_like(self.kernels)
        dL_dX = torch.zeros(self.padded_input.shape[1:]) # exclude the batch dim  -> C x H x W
        
        if self.stride != 1:
            dL_dV_K = dilate(dL_dV, self.stride-1)
            dL_dV_X = dL_dV_K.clone()
            
        dL_db = torch.sum(dL_dV, dim=(1,2))
        
        # dL_dK
        for k, X_i in enumerate(self.padded_input):
            for i, K_i in enumerate(self.kernels):
                for j, K_i_j in enumerate(K_i): # channel
                    dL_dK[i, j] += conv_2d(X_i[j], dL_dV_K[i])
                

        #dL_dX
        for i, K_i in enumerate(self.kernels):
            for j, K_i_j in enumerate(K_i): # channel
                    dL_dX[j] += corr_2d(dL_dV_X[j], K_i_j, corr_type='full')
                    
                    
        
        # updates
        self.kernels -= lr * dL_dK
        self.biases -= lr * dL_db.unsqueeze(-1).unsqueeze(-1)
        
        if self.padding != 0:
            dL_dX = dL_dX[...,  self.padding: -self.padding, self.padding: -self.padding]
            
        return dL_dX
                     
        
        
class Pooling():
    def __init__(self, kernel_size:int, stride:int=None, pooling_type:str='max') -> torch.Tensor:
        """_summary_

        Args:
            kernel_size (int): _description_
            stride (int, optional): _description_. Defaults to None.
            pooling_type (str, optional): max or avg Defaults to 'max'.

        Returns:
            torch.Tensor: _description_
        """
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.pooling_type = pooling_type
    
    def forward(self, X): # X: B x C x H x W
        out_rows = (X.shape[2] - self.kernel_size) // self.stride + 1
        out_cols = (X.shape[3] - self.kernel_size) // self.stride + 1
        
        out = torch.zeros((X.shape[0], X.shape[1], out_rows, out_cols), dtype=X.dtype)
        
        self.backprop_value = torch.zeros_like(X[0])

        
        for j in range(out_rows):
            for i in range(out_cols):
                
                window = X[:, :, i*self.stride:i*self.stride+ self.kernel_size, j*self.stride:j*self.stride+ self.kernel_size]
                
                if self.pooling_type == 'max':
                    out[:, :, i, j] = torch.amax(window, dim=(2, 3))  # This gives us B x C
                    
                    self.backprop_value[:, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+ self.kernel_size]+=torch.sum(window == out[:, :, i, j].unsqueeze(-1).unsqueeze(-1), dim=0)
                    pass
                    
                elif self.pooling_type == 'avg':
                    out[:, :, i, j] = torch.mean(window, dim=(2,3))
                    self.backprop_value[:, i*self.stride:i*self.stride+ self.kernel_size, j*self.stride:j*self.stride+ self.kernel_size] += 1/(self.kernel_size ** 2)
                    
        return out
    def __call__(self, X):
        return self.forward(X)
    
    def backward(self, dL_dV, lr):
        return dL_dV * self.backprop_value
        
        
        
class Flatten():

    def forward(self, X): # X: B x C x H X W
        self.input_shape = X.shape
        return X.view(self.input_shape[0], -1) #TODO: should i clone this?

    def __call__(self, X):
        return self.forward(X)
    
    def backward(self, dL_dV: torch.Tensor):
        return dL_dV.view(self.input_shape[1:])
        
    
            
    

