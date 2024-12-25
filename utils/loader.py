from torchvision.datasets import FashionMNIST
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Subset
import random

"""
FashionMNIST  contains 28x28 grayscale images of 10 categories.
Training set has 60,000 images and test set has 10,000 images.
Categories:
0: T-shirt
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle Boot
"""


def get_datasets(integer_valued=False, flatten=False):#, one_hot=False, num_classes=None):
    # transforms list
    transforms = [T.ToTensor()]
    
    # apply flags
    if integer_valued:
        transforms.extend([
            lambda x: x*255,
            lambda x: x.long(),
        ])
    
    if flatten:
        transforms.append(lambda x: x.view(-1))
    
    # if one_hot and num_classes is None:
    #     raise ValueError("num_classes must be specified when one_hot is True")
    # if one_hot:
    #     transforms.extend([
    #         lambda x: x.long(),
    #         lambda x: F.one_hot(x, num_classes=num_classes)])
    
    # create datasets
    full_train_data = FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=T.Compose(transforms)
    )
    test_data = FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=T.Compose(transforms)
    )
    
    
    # Bucket images by class
    class_buckets = {i: [] for i in range(10)}  # One bucket for each class
    for idx, (_, label) in enumerate(full_train_data):
        class_buckets[label].append(idx)

    # Shuffle each bucket
    # Set a random seed for reproducibility
    RANDOM_SEED = 42 # for reproductibility
    random.seed(RANDOM_SEED)
    for bucket in class_buckets.values():
        random.shuffle(bucket)

    # Sample 1000 images per class for the validation set
    val_indices = []
    for class_label, indices in class_buckets.items():
        val_indices.extend(indices[:1000])  # Take the first 1000 images per class

    # Create validation and remaining training datasets
    train_indices = list(set(range(len(full_train_data))) - set(val_indices))

    train_data = Subset(full_train_data, train_indices)
    val_data = Subset(full_train_data, val_indices)
    

    return train_data, val_data, test_data, full_train_data
# def get_dataloaders(dataset, batch_size=64):
   
    
#     # initialize the dataloaders
#     train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#     return train_dl, test_dl

# import matplotlib.pyplot as plt

# labels_map={
#     0: 'T-shirt',
#     1: 'Trouser',
#     2: 'Pullover',
#     3: 'Dress',
#     4: 'Coat',
#     5: 'Sandal',
#     6: 'Shirt',
#     7: 'Sneaker',
#     8: 'Bag',
#     9: 'Ankle Boot',
# }

# figure = plt.figure(figsize = (10,10))
# cols, rows = 3, 3

# for i in range (1, cols*rows + 1):
#     sample_idx = torch.randint(len(train_data), size = (1,)).item()
#     image, label = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis('off')
#     plt.imshow(image.squeeze(), cmap='gray')
# plt.show()