import autograd.numpy as np

class TestFunction:
  def evaluate(self,x):
    raise NotImplementedError('Function has not been implemented')
  def evaluate_grad(self,x):
    x = x.clone().numpy()
    if self.grad is None:
      self.grad = autograd.grad(self.evaluate)
    return self.grad(x)

class Ackley(TestFunction):
  def __init__(self):
    self.grad = None
  def evaluate(self,x):
    x = np.array(x).reshape((2,))
    val = -20*np.exp(-0.2*np.sqrt(np.sum(0.5*x**2,axis=0))) \
              -np.exp(np.sum(0.5*np.cos(2*np.pi*x),axis=0)) + 20 + np.e
    return val

class Rosenbrock(TestFunction):
  def __init__(self,a,b):
    self.a = a
    self.b = b
    self.grad = None
  def evaluate(self,x):
    x,y = np.array(x).reshape((2,))
    val = (self.a-x)**2 + self.b*(y-x**2)**2
    return val

class Beale(TestFunction):
  def evaluate(self,x):
    x = np.array(x).reshape((2,))
    val = (1.5 - x[0] + x[0]*x[1])**2 + \
          (2.25 - x[0] + x[0]*x[1]**2)**2 + \
          (2.625 - x[0] + x[0]*x[1]**3)**2
    return val

class Goldstein_Price(TestFunction):
  def evaluate(self,x):
    x,y = np.array(x).reshape((2,))
    return (1+np.square(x+y+1)*(19 - 14*x + 3*np.square(x) - 14*y + 6*x*y + 3*np.square(y)))*\
    (30 + np.square(2*x - 3*y)*(18 - 32*x + 12*np.square(x) + 48*y - 36*x*y + 27*np.square(y)))

class Booth(TestFunction):
  def evaluate(self,x):
    x,y = np.array(x).reshape((2,))
    return np.square(x + 2*y - 7) + np.square (2*x + y - 5)

class Bukin(TestFunction):
  def evaluate(self,x):
    x,y = np.array(x).reshape((2,))
    return 100*np.sqrt(np.abs(y - 0.01*np.square(x))) + 0.01*np.abs(x + 10)

class Matyas(TestFunction):
  def evaluate(self,x):
    x,y = np.array(x).reshape((2,))
    return 0.26*(x**2+y**2) - 0.48*x*y

class Levi(TestFunction):
  def evaluate(self,x):
    x,y = np.array(x).reshape((2,))
    return np.square(np.sin(3*np.pi*x)) + np.square(x-1)*(1+np.square(np.sin(3*np.pi*y))) + \
            np.square(y-1)*(1+np.square(np.sin(2*np.pi*y)))

class Himmelblau(TestFunction):
  def evaluate(self,x):
    x,y = np.array(x).reshape((2,))
    return np.square(np.square(x) + y - 11) + np.square(x + np.square(y) - 7)

class Threehumpcamel(TestFunction):
  def evaluate(self,x):
    x,y = np.array(x).reshape((2,))
    return 2*np.square(x) - 1.05*np.power(x,4) + np.power(x,6)/6 + x*y + np.square(y)

class Easom(TestFunction):
  def evaluate(self,x):
    x,y = np.array(x).reshape((2,))
    return -(np.cos(x)*np.cos(y)*np.exp(-(np.square(x-np.pi) + np.square(y-np.pi))))

class Crossintray(TestFunction):
  def evalute(self,x):
    x,y = np.array(x).reshape((2,))
    return -0.0001*np.power(1+np.abs(
            np.sin(x)*np.sin(y)*np.exp(np.abs(
            100 - np.sqrt(np.square(x) + np.square(y))/np.pi
            ) + 1)
            ), 0.1)

class Sqrt(TestFunction):
  def evaluate(self,x):
    x = np.array(x).reshape((1,))[0]
    return np.sqrt(np.abs(x))

class Mod(TestFunction):
  def evaluate(self,x):
    x = np.array(x).reshape((1,))[0]
    return np.abs(x)

class DiscontinuousSine(TestFunction):
  def evaluate(self,x):
    x = np.array(x).reshape((1,))[0]
    if(x <= 0):
      return 0
    else:
      return np.pow(x,1/3) + np.sin(x)

class DiscontinuousPoly(TestFunction):
  def evaluate(self,x):
    x = np.array(x).reshape((1,))[0]
    if(x <= 0):
      return np.square(x)
    else:
      return x