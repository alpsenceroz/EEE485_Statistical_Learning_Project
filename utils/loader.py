from torchvision.datasets import FashionMNIST
import torchvision.transforms as T
import torch.nn.functional as F

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
    train_data = FashionMNIST(
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
    
    return train_data, test_data
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