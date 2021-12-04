import torch 
from torchswarm_gpu.particle import NoOpParticle
from torchswarm_gpu.pso import ParticleSwarmOptimizer
class BaselineOptimizer(ParticleSwarmOptimizer):
    def __init__(self,dimensions = 4, swarm_size=100,classes=1, true_y=None, options=None):
        if (options == None):
            options = [0.9,0.8,0.5,100]
        self.swarm_size = swarm_size
        self.c1 = options[0]
        self.c2 = options[1]
        self.beta = options[2]
        self.max_iterations = options[3]
        self.swarm = []
        self.gbest_position = None
        self.gbest_value = torch.Tensor([float("inf")])
        self.gbest_particle = None

        for i in range(swarm_size):
            self.swarm.append(NoOpParticle(dimensions, self.beta, self.c1, self.c2, classes, true_y))