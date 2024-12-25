from utils.activations import ReLU, Sigmoid, Tanh, Softmax, Activation
from utils.layers import Linear
from utils.sequential import SequentialModel



layers = [
    Linear(28 * 28, 512), ReLU(), 
    Linear(512, 256), ReLU(),
    Linear(256, 128), ReLU(),
    Linear(128, 10), Softmax()
    ]

MLP = SequentialModel(layers)
