import torch
import torch.nn as nn
import numpy as np

lossfn = nn.CrossEntropyLoss()

wt = torch.Tensor([1/5, 4/5])

lossfnw = nn.CrossEntropyLoss(weight=wt)




# TODO NEXT:

# create file lossfunctions
# implement 2d_cross_entropy_loss
# typecast
# dimension check


# try another loss fn


# d more dimensions ...

# [Batchsize x nClasses x d...]
# [3 x 2]
input = np.array([[0, 100],[0, 100],[100, 0], [100, 0], [100, 0]])
# input = np.expand_dims(input, 0)


# [Batchsize]
# [3 x d...]
# classes with label 0, 1
target = np.array([1, 0, 0, 0, 0])
# add Batchsize dimension
# target = np.expand_dims(target, 0)

input = torch.from_numpy(input).float()
target = torch.from_numpy(target).long()


# input = torch.randn(10, 120).float()
# target = torch.FloatTensor(10).uniform_(0, 120).long()



# target = 0.0
loss = lossfn(input, target)
print('loss', loss)

# with weight
lossw = lossfnw(input, target)
print('loss w weight', lossw)
