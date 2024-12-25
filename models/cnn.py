from utils.activations import ReLU, Sigmoid, Tanh, Softmax, Activation
from utils.layers import Linear, Conv, Pooling, Flatten
from utils.sequential import SequentialModel




# layers = [
#     Pooling(pooling_type='avg', kernel_size=2), # 1 x 28 x 28
#     # Conv(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0), 
#     # Tanh(),
#     # Pooling(pooling_type='avg', kernel_size=2),
#     Conv(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0),
#     Tanh(),
#     Pooling(pooling_type='avg', kernel_size=2),
#     Conv(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
#     Tanh(),
#     Flatten(),
#     Linear(in_features=120, out_features=84),
#     Tanh(),
#     Linear(in_features=84, out_features=10),
#     Softmax()    
# ]


layers = [
    Conv(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
    Tanh(),
    Pooling(pooling_type='avg', kernel_size=2), # 1 x 28 x 28
    # Conv(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0), 
    # Tanh(),
    # Pooling(pooling_type='avg', kernel_size=2),
    Conv(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
    Tanh(),
    Pooling(pooling_type='avg', kernel_size=2),
    Conv(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
    Tanh(),
    Flatten(),
    Linear(in_features=120, out_features=84),
    Tanh(),
    Linear(in_features=84, out_features=10),
    Softmax()    
]

LeNet5 = SequentialModel(layers)