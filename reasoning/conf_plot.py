import numpy as np
import matplotlib.pyplot as plt

x = np.array(list(np.arange(-1,1,0.01)))

cs = [0.1,0.5,1,5,10]
for c in cs:
    y = -np.log(1+np.exp(c*x))
    plt.plot(x,y,label="c="+str(c))

plt.xlabel("x")
plt.ylabel("-log(1+exp(c*x))")
plt.legend()
plt.show()

