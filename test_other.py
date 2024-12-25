from utils.layers import Conv, Pooling, Flatten, pad, dilate, conv_2d, corr_2d
import torch





input = torch.randn(1, 1, 2, 2)

# pool = Pooling(kernel_size=4, pooling_type='avg')

# out = pool(input)
# print(input)
# print(out)
# print(pool.backprop_value)

# flatten = Flatten()

# out = flatten(input)
# print(out)
# print(out.shape)

# out2 = flatten.backward(out[0])

# print(out2)
# print(out2.shape)

# print(input[0] == out2)


input = torch.randn(2, 2)
print(input)
out = pad(input, padding=2)
print(out)
print(out.shape)



# input = torch.randn(2, 2)
# print(input)
# out = dilate(input, dilation_size=2)
# print(out)
# print(out.shape)



# input = torch.randn(2,2,2, 2)
# input = torch.randn(2, 2)

# print(input)

# out = corr_2d(input, input)

# # out = conv(input)
# print(out)
# print(out.shape)



# input = torch.randn(2,2,4,4)

# print(input)

# conv = Conv(in_channels=2, out_channels=5, kernel_size=2, stride=2, padding=2)


# out = conv(input)
# # out = conv(input)
# print(out)
# print(out.shape)

# dv = out[0]
# print(dv.shape)

# out2 = conv.backward(dv, 1)
# print(out2)
# print(out2.shape)