import pickle

from utils.activations import Activation
from utils.layers import Linear


class SequentialModel:
    def __init__(self, layers: list[Linear | Activation]):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def to(self, device):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.to(device)
    
    def backward(self, dL_do, lr):
        dL_dV = dL_do
        for layer in reversed(self.layers):
            print("dL_dV.shape", dL_dV.shape)
            dL_dV = layer.backward(dL_dV, lr)
            
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.layers = tmp_dict["layers"]

    
    

## loss functions