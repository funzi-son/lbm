import numpy as np

x = np.random.uniform(-1,2,size=(10))
print(x)
pos = (x>1)*1
neg = 1-(x>0)*1

nx = pos + (1-pos)*x
nx = (1-neg)*nx

print(nx)
