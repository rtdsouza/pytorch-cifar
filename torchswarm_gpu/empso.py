import torch 
import time
from torchswarm_gpu.particle import Particle

class EMParticleSwarmOptimizer:
    def __init__(self,dimensions = 4, swarm_size=100,classes=1, options=None):
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
            self.swarm.append(Particle(dimensions, self.beta, self.c1, self.c2, classes))
    
    def optimize(self, function):
        self.fitness_function = function

    def _evaluate_gbest(self):
        #--- Set PBest
        for particle in self.swarm:
            fitness_cadidate = self.fitness_function.evaluate(particle.position)
            # print("========: ", fitness_cadidate, particle.pbest_value)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position.clone()
            # print("========: ",particle.pbest_value)
        #--- Set GBest
        for particle in self.swarm:
            best_fitness_cadidate = self.fitness_function.evaluate(particle.position)
            if(self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position.clone()
                self.gbest_particle = particle

    def run(self,verbosity = True,return_cr = False,return_positions=True):
        #--- Run 
        positions = []
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            self._evaluate_gbest()
            positions.append(self.gbest_position.clone().numpy())
            c1r1s = []
            c2r2s = []
            #--- For Each Particle Update Velocity
            for particle in self.swarm:
                positions.append(particle.position.clone().numpy())
                c1r1,c2r2 = particle.update_velocity(self.gbest_position)
                c1r1s.append(c1r1)
                c2r2s.append(c2r2)
                particle.move()
            # for particle in self.swarm:
            #     print(particle)
            # print(self.gbest_position.numpy())
            toc = time.monotonic()
            if (verbosity == True):
                print('Iteration {:.0f} >> global best fitness {:.8f}  | iteration time {:.3f}'
                .format(iteration + 1,self.gbest_value,toc-tic))
            if(iteration+1 == self.max_iterations):
                print(self.gbest_position)
        if(return_positions):
            return positions
        elif(return_cr):
            return sum(c1r1s)/len(c1r1s),sum(c2r2s)/len(c2r2s)