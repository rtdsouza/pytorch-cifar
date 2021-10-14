import torch 
import time
from torchswarm_gpu.particle import HMCParticle
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 
class HMCParticleSwarmOptimizer:
    def __init__(self,dimensions = 4, swarm_size=100,classes=1, true_y=None, step_size=0.001, num_steps=100, options=None):
        if (options == None):
            options = [2,2,100]
        self.swarm_size = swarm_size
        self.c1 = options[0]
        self.c2 = options[1]
        self.max_iterations = options[2]
        self.swarm = []
        self.true_y = true_y
        self.gbest_position = None
        self.gbest_value = torch.Tensor([float("inf")]).to(device)
        self.step_size = step_size
        self.num_steps = num_steps

        for i in range(swarm_size):
            mass_matrix = torch.diag(torch.rand(classes))
            particle = HMCParticle(dimensions, self.c1, self.c2, classes, mass_matrix)
            particle.set_ref_to_optimizer(self)
            self.swarm.append(particle)
    
    def optimize(self, function):
        self.fitness_function = function
        for particle in self.swarm:
            particle.set_fitness_function(function)

    def run(self,verbosity = True):
        if self.fitness_function is None:
            print('Fitness function not specified')
            return
        #--- Run 
        positions = []
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            #--- Set PBest
            if self.gbest_position is not None:
                self.gbest_value = self.fitness_function.evaluate(self.gbest_position)
            for particle in self.swarm:
                fitness_candidate = self.fitness_function.evaluate(particle.position)
                # print("========: ", fitness_candidate, particle.pbest_value)
                if(particle.pbest_value > fitness_candidate):
                    particle.pbest_value = fitness_candidate
                    particle.pbest_position = particle.position.clone()

                if(fitness_candidate < self.gbest_value):
                    self.gbest_position = particle.position.clone()
                    self.gbest_value = fitness_candidate
                    self.gbest_mass_matrix = particle.mass_matrix
                    self.gbest_velocity = particle.velocity
            
            #--- For Each Particle Update Velocity
            positions.append(self.gbest_position.clone())
            for particle in self.swarm:
                positions.append(particle.position.clone())
                particle.move(self.num_steps,self.step_size)
                
            toc = time.monotonic()
            if (verbosity == True):
                print('Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}'
                .format(iteration + 1,self.gbest_value,toc-tic))
            if(iteration+1 == self.max_iterations):
                print(self.gbest_position)
        return positions