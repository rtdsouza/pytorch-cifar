from torchswarm_gpu.particle import RotatedEMParticle
from torchswarm_gpu.rempso import RotatedEMParicleSwarmOptimizer
import torch
from nn_utils import *
from keras.utils import to_categorical
#test

input = torch.randn(125, 10, requires_grad=True)

target = torch.empty(125, dtype=torch.long).random_(10)
input_2 = torch.Tensor(to_categorical(target))
print(input.data)
print(input_2.data)
print(target.data)
print("----------")
criterion = torch.nn.CrossEntropyLoss()

print(criterion(input, target))
print(criterion(input_2, target))



lis = torch.LongTensor([1,0,2,3])
print(RotatedEMParticle(4, 0.1, 0.8, 0.9, 1, lis).position)
print(RotatedEMParticle(4, 0.1, 0.8, 0.9, 1,  lis).pbest_position)

p = RotatedEMParicleSwarmOptimizer(125, 10, 10, target)
p.optimize(CELoss(target))
for i in range(2):
    c1r1, c2r2, gbest = p.run_one_iter(verbosity=False)
    print(c1r1, c2r2, gbest)

print(criterion(gbest, target))
