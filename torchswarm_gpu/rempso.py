import torch 
import time
from torchswarm_gpu.particle import RotatedEMParticle, RotatedEMParticleWithBounds
from torchswarm_gpu.pso import ParticleSwarmOptimizer
if torch.cuda.is_available():  
  dev = "cuda:0" 
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:  
  dev = "cpu"  
device = torch.device(dev) 
class RotatedEMParticleSwarmOptimizer(ParticleSwarmOptimizer):
    def __init__(self,dimensions = 4, swarm_size=100,classes=1, true_y=None, options=None):
        if (options == None):
            options = [2,2,0.1,100]
        self.swarm_size = swarm_size
        self.c1 = options[0]
        self.c2 = options[1]
        self.beta = options[2]
        self.max_iterations = options[3]
        self.swarm = []
        self.true_y = true_y
        self.gbest_position = None
        self.gbest_value = torch.Tensor([float("inf")])

        for i in range(swarm_size):
            self.swarm.append(RotatedEMParticle(dimensions, self.beta, self.c1, self.c2, classes, self.true_y))

class RotatedEMParicleSwarmOptimizerWithBounds:
    def __init__(self,dimensions = 4, swarm_size=100,classes=1, bounds=None, options=None):
        if (options == None):
            options = [0.9,0.8,0.1,100]
        self.swarm_size = swarm_size
        self.c1 = options[0]
        self.c2 = options[1]
        self.beta = options[2]
        self.max_iterations = options[3]
        self.swarm = []
        self.gbest_position = None
        self.gbest_value = torch.Tensor([float("inf")])

        for i in range(swarm_size):
            self.swarm.append(RotatedEMParticleWithBounds(dimensions, self.beta, self.c1, self.c2, classes, bounds))