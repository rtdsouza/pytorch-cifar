import torch
import numpy as np
from torchswarm_gpu.utils.rpso import *
from tensorflow.keras.utils import to_categorical
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

class Particle:
    def __init__(self, dimensions, w, c1, c2, classes):
        self.dimensions = dimensions
        self.position = torch.randn(dimensions, classes).to(device)
        self.velocity = torch.zeros((dimensions, classes)).to(device)
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")]).to(device)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __str__(self):
        return ('Particle >> pbest {:.3f}  | pbest_position {}'
                .format(self.pbest_value.item(),self.pbest_position))
        
    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        for i in range(0, self.dimensions):
            # print(self.velocity[i], (self.pbest_position[i]), .(gbest_position[i] - self.position[i]))
            self.velocity[i] = self.w * self.velocity[i] \
                                + self.c1 * r1 * (self.pbest_position[i] - self.position[i]) \
                                + self.c2 * r2 * (gbest_position[i] - self.position[i])

            # print(self.velocity[i])
        return ((self.c1*r1).item(), (self.c2*r2).item())
    
    def move(self):
        self.position += self.velocity

class RotatedParticle(Particle):
    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        a_matrix = get_rotation_matrix(self.dimensions, np.pi/5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix 
        self.velocity = self.w * self.velocity \
                        + torch.matmul((a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix).float().to(device),(self.pbest_position - self.position).float().to(device)) \
                        + torch.matmul((a_inverse_matrix * get_phi_matrix(self.dimensions, self.c2, r2) * a_matrix).float().to(device), (gbest_position - self.position).float().to(device))
        return ((self.c1*r1).item(), (self.c2*r2).item())

class EMParticle(Particle):
    def __init__(self, dimensions, beta, c1, c2, classes):
        super().__init__(dimensions, 0, c1, c2, classes)
        self.momentum = torch.zeros((dimensions, 1))
        self.beta = beta

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        momentum_t = self.beta*self.momentum + (1 - self.beta)*self.velocity
        self.velocity = momentum_t \
                            + self.c1 * r1 * (self.pbest_position - self.position) \
                            + self.c2 * r2 * (gbest_position - self.position)
        self.momentum = momentum_t
        return ((self.c1*r1).item(), (self.c2*r2).item())

class RotatedEMParticle(Particle):
    def __init__(self, dimensions, beta, c1, c2, classes, true_y):
        super().__init__(dimensions, 0, c1, c2, classes)
        # print(to_categorical(true_y.cpu().detach().numpy()))
        if(true_y is not None):
            self.position = initialize_position(true_y, dimensions, classes).to(device)
        else:
            self.position = torch.randn((dimensions,classes)).to(device)
        self.pbest_position = self.position
        self.momentum = torch.zeros((dimensions, 1)).to(device)
        self.beta = beta


    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        momentum_t = self.beta*self.momentum + (1 - self.beta)*self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi/5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix 
        self.velocity = momentum_t \
                        + torch.matmul((a_inverse_matrix @ get_phi_matrix(self.dimensions, self.c1, r1) @ a_matrix).float().to(device),(self.pbest_position - self.position).float().to(device)) \
                        + torch.matmul((a_inverse_matrix @ get_phi_matrix(self.dimensions, self.c2, r2) @ a_matrix).float().to(device), (gbest_position - self.position).float().to(device))

        return ((self.c1*r1).item(), (self.c2*r2).item())
    def move(self):
        for i in range(0, self.dimensions):
            # print("Before Update: ",self.position[i])
            self.position[i] = self.position[i] + self.velocity[i]
            # print("After Update: ",self.position[i], self.velocity[i])



class RotatedEMParticleWithBounds(Particle):
    def __init__(self, dimensions, beta, c1, c2, classes, bounds):
        super().__init__(dimensions, 0, c1, c2, classes)
        self.position = (bounds[0]-bounds[1])*torch.rand(dimensions, classes) + bounds[1]
        self.momentum = torch.zeros((dimensions, 1))
        self.beta = beta

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        momentum_t = self.beta*self.momentum + (1 - self.beta)*self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi/5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix 
        self.velocity = momentum_t \
                        + torch.matmul((a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix).float().to(device),(self.pbest_position - self.position).float().to(device)) \
                        + torch.matmul((a_inverse_matrix * get_phi_matrix(self.dimensions, self.c2, r2) * a_matrix).float().to(device), (gbest_position - self.position).float().to(device))

        return ((self.c1*r1).item(), (self.c2*r2).item())
    def move(self):
        for i in range(0, self.dimensions):
            # print("Before Update: ",self.position[i])
            self.position[i] = self.position[i] + self.velocity[i]
            # print("After Update: ",self.position[i], self.velocity[i])
        self.position = torch.clamp(self.position,-50,50)

class HMCParticleWithGradients(Particle):
    def __init__(self, dimensions, c1, c2, classes, mass_matrix=None, energy_function=None, energy_grad=None, beta=0):
        super().__init__(dimensions, 1, c1, c2, classes)
        if(mass_matrix is not None):
            self.mass_matrix = mass_matrix
        else:
            self.mass_matrix = torch.diag(torch.ones(classes))
        self.M_inv = torch.linalg.inv(self.mass_matrix)
        self.energy = energy_function
        self.classes = classes
        self.energy_grad = energy_grad
        self.position = torch.randn(classes).to(device)
        self.pbest_position = self.position.clone()
        self.velocity = torch.randn(classes).to(device)
        self.beta = beta

    def evaluate_grad(self):
        return self.energy.evaluate_grad(self.position)

    def set_fitness_function(self, fitness_function):
        self.energy = fitness_function

    def kinetic_energy(self, velocity):
        return velocity @ self.M_inv @ velocity

    def leapfrog(self, L=100, step_size=0.001):
        M_inv = self.M_inv
        proposed_velocity = self.velocity.clone()
        proposed_position = self.position.clone()
        proposed_velocity -= 0.5 * step_size * self.evaluate_grad()
        for i in range(L):
            proposed_position += step_size * M_inv @ self.velocity
            if(hasattr(self,'optimizer')):
                fitness_candidate = self.energy.evaluate(proposed_position)
                if(fitness_candidate < self.optimizer.gbest_value):
                    self.optimizer.gbest_value = fitness_candidate
                    self.optimizer.gbest_position = proposed_position.clone()
            if(i != L-1):
                proposed_velocity -= step_size * self.evaluate_grad()
        proposed_velocity -= 0.5 * step_size * self.evaluate_grad()
        proposed_velocity *= -1
        proposed_velocity = proposed_velocity.float()
        return proposed_position, proposed_velocity

    def mh_step(self, proposed_position, proposed_velocity):
        if self.energy is None:
            print('Fitness function not specified')
            return
        original_energy = self.energy.evaluate(self.position) \
            + self.kinetic_energy(self.velocity)
        proposed_energy = self.energy.evaluate(proposed_position) \
            + self.kinetic_energy(proposed_velocity)
        acceptance_prob = min(1.0, torch.exp(
            original_energy - proposed_energy).item())
        if(torch.rand(1) < acceptance_prob):
            self.position = proposed_position
            self.velocity = proposed_velocity

    def move(self, num_steps=100, step_size=0.001):
        self.velocity = torch.randn(self.classes)
        proposal = self.leapfrog(num_steps,step_size)
        self.mh_step(*proposal)

class HMCParticle(HMCParticleWithGradients):
    def evaluate_grad(self):
        gbest_position = self.optimizer.gbest_position.clone().reshape(self.position.shape)
        return (self.c1 * torch.rand(1) \
                * (self.mass_matrix @ (self.pbest_position - self.position)) \
                + self.c2 * torch.rand(1) \
                * (self.mass_matrix @ (gbest_position - self.position))) / self.eta

    def set_ref_to_optimizer(self,optimizer):
        self.optimizer = optimizer

    def move(self, num_steps=100, step_size=0.001):
        if not hasattr(self,'optimizer'):
            print('Please use set_ref_to_optimizer() to pass the optimizer ref to the particle')
            return

        velocity_distribution = torch.distributions.MultivariateNormal(
            self.optimizer.gbest_velocity,
            self.optimizer.gbest_mass_matrix
        )
        self.eta = step_size
        old_v = self.velocity.clone()
        new_v = velocity_distribution.sample()
        self.velocity = self.beta*old_v + (1-self.beta)*new_v
        proposal = self.leapfrog(num_steps,step_size)
        self.mh_step(*proposal)

def initialize_position(true_y, dimensions, classes):
    const = -4
    position = torch.tensor([[const]*classes]*dimensions)
    for i in range(dimensions):
        position[i][true_y[i]] = 1
    return position + torch.rand(dimensions, classes)
