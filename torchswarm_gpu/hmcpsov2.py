import torch 
import time
from torchswarm_gpu.particle import HMCParticle
from torchswarm_gpu.empso import EMParticleSwarmOptimizer
if torch.cuda.is_available():  
  dev = "cuda:0" 
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:  
  dev = "cpu"  
device = torch.device(dev) 
class HMCParticleSwarmOptimizer:
    def __init__(self,dimensions = 4, swarm_size=100,classes=1, true_y=None, step_size=0.001, num_steps=100, options=None):
        if (options == None):
            options = [2,2,100]
        if(swarm_size < 2):
            raise ValueError('Swarm size includes HMC particle \
                and cannot be less than 2.')
        self.swarm_size = swarm_size
        self.c1 = options[0]
        self.c2 = options[1]
        self.max_iterations = options[2]
        self.true_y = true_y
        self.gbest_position = None
        self.gbest_value = torch.Tensor([float("inf")])
        self.step_size = step_size
        self.num_steps = num_steps
        mass_matrix = torch.diag(torch.rand(dimensions))
        self.hmc_particle = HMCParticle(dimensions, self.c1, self.c2, classes, mass_matrix)
        self.hmc_particle.set_ref_to_optimizer(self)
        self.em_swarm = EMParticleSwarmOptimizer(dimensions,self.swarm_size-1,classes,true_y)
        self.em_swarm.max_iterations = 1
        self.gbest_mass_matrix = mass_matrix
        self._shape = self.hmc_particle.position.shape
    
    def optimize(self, function):
        self.fitness_function = function
        self.em_swarm.optimize(function)
        self.hmc_particle.set_fitness_function(function)

    def run(self,verbosity = True,return_cr=False,return_positions=True):
        if self.fitness_function is None:
            print('Fitness function not specified')
            return
        #--- Run 
        positions = []
        c1r1s = []
        c2r2s = []
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            #--- Set PBest
            if self.gbest_position is not None:
                self.gbest_value = self.fitness_function.evaluate(self.gbest_position)

            if(self.em_swarm.gbest_position is None):
                self.em_swarm._evaluate_gbest()
            
            hmc_fitness = self.fitness_function.evaluate(self.hmc_particle.position)

            if hmc_fitness >= self.em_swarm.gbest_value:
                if(self.em_swarm.gbest_value < self.gbest_value or \
                    self.gbest_position is None):
                    self.gbest_position = self.em_swarm.gbest_position.clone().reshape(self._shape)
                    self.gbest_value = self.em_swarm.gbest_value
                self.gbest_particle = self.em_swarm.gbest_particle
            else:
                if(hmc_fitness < self.gbest_value or \
                    self.gbest_position is None):
                    self.gbest_position = self.hmc_particle.position.clone().reshape(self._shape)
                    self.gbest_value = hmc_fitness
                self.gbest_particle = self.hmc_particle

            self.em_swarm.gbest_value = self.gbest_value
            self.em_swarm.gbest_position = self.gbest_position.clone()

            positions.append(self.gbest_position.clone())
            for particle in self.em_swarm.swarm:
                positions.append(particle.position.clone().reshape(self._shape))
            positions.append(self.hmc_particle.position.clone().reshape(self._shape))

            self.gbest_velocity = self.gbest_particle.velocity.clone().reshape(self._shape)
            c1r1,c2r2 = self.em_swarm.run(verbosity=False,return_cr=True,return_positions=False)
            c1r1s.append(c1r1)
            c2r2s.append(c2r2)
            self.hmc_particle.move(self.num_steps,self.step_size)

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

    def run_one_iter(self,verbosity=True):
        tic = time.monotonic()
        old_iterations = self.max_iterations
        self.max_iterations = 1
        result = self.run(verbosity=False,return_cr=True,return_positions=False)
        self.max_iterations = old_iterations
        toc = time.monotonic()
        if (verbosity == True):
            print(' >> global best fitness {:.8f}  | iteration time {:.3f}'
            .format(self.gbest_value,toc-tic))
        return result + (self.gbest_position,)
