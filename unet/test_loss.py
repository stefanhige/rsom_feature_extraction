import torch
import torch.nn as nn
import numpy as np




# TEST NEW LOSSFUNCTIONS

# output from CNN has shape
# [Batchsize x nClasses x d...]



# nn.CrossEntropyLoss()
# wants shape [Batchsize x d..]
# with entries 0, 1, 2 ... for classes


# create fake inputs

pred = np.zeros((1, 2, 10, 1))

# generate  for class 1:
pred_ideal = pred.copy()
pred_ideal[:, 1, 5, :] = 1

# INSERT MODIFICATION (nonideal)
pred_ideal[:, 1, 0:1,:] = 1


pred_ideal[:, 0, :, :] = np.logical_not(pred_ideal[:, 1, :, :])
pred_ideal *= 5



# print(pred_ideal)


label = np.zeros((1, 10, 1))
label[:, 5:8, :] = 1

nonz_idx = np.nonzero(label[0,:,0])

print(nonz_idx)
print(label)

# replace beginning with reverse function
label[0,:nonz_idx[0],0] = np.flip(fnc(nonz_idx[0]))

# replace end with function
label[0,nonz_idx[-1],0] = fnc(label_size = nonz_idx[-1])



# weight vector linear ascending
weight = np.array([6,5,4,3,2,1,2,3,4,5])
weight = np.transpose([weight])
weight =np.expand_dims(weight,0)

# print('weight')
# print(weight)
# print(weight.shape)

# convert to torch
pred_ideal = torch.from_numpy(pred_ideal).float()
label = torch.from_numpy(label).long()
weight = torch.from_numpy(weight).float()

lossfn1 = nn.CrossEntropyLoss(reduction='none')
loss1 = lossfn1(pred_ideal, label)
print('Standard cross entropy loss:', torch.sum(loss1))
print(loss1.shape)
print('weighted loss:', torch.sum(loss1*weight) )

# lsfn = nn.LogSoftmax()
# pred_ideal_logs = lsfn(pred_ideal)

# print(pred_ideal_logs)


# lossfn2 = nn.NLLLoss()
# loss2 = lossfn2(pred_ideal_logs, label)
# print('LogSoftmax + NLLLoss()', loss2.item())

# lossfn = nn.CrossEntropyLoss()

# wt = torch.Tensor([1/5, 4/5])

# lossfnw = nn.CrossEntropyLoss(weight=wt)


# d more dimensions ...

# [Batchsize x nClasses x d...]
# [3 x 2]
# input = np.array([[0, 100],[0, 100],[100, 0], [100, 0], [100, 0]])
# input = np.expand_dims(input, 0)


# [Batchsize]
# [3 x d...]
# classes with label 0, 1
# target = np.array([1, 0, 0, 0, 0])
# add Batchsize dimension
# target = np.expand_dims(target, 0)

# input = torch.from_numpy(input).float()
# target = torch.from_numpy(target).long()


# input = torch.randn(10, 120).float()
# target = torch.FloatTensor(10).uniform_(0, 120).long()



# target = 0.0
# loss = lossfn(input, target)
# print('loss', loss)

# with weight
# lossw = lossfnw(input, target)
# print('loss w weight', lossw)
