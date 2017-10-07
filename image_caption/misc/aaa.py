import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable


x = [[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]]
x = [[[1,2,3,4,5,6,7,8,9,10]]]
x = np.array(x).astype("float32")
x = torch.from_numpy(x)
x = Variable(x)
print(x)
print(F.adaptive_avg_pool2d(x, [10,10]))