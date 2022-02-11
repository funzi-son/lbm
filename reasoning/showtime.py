import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os
from collections import OrderedDict
import ntpath

sns.set()


def load_result(d):
    completeness = None
    f = os.path.join(d,"time.txt")
    t = np.loadtxt(f)
    return  np.mean(t), np.std(t)

ms = []
ss = []
ll = [3,4,5,6,7,8,9,10]
kk = [2**(20+l) for l in ll]
for i in ll: 
    mean,std = load_result("./results_pool/20_%d_0.500_5.000"%(i))
    ms.append(mean)
    ss.append(std)

ms = np.array(ms)
ss = np.array(ss)

#plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('N (M=20)')
ax1.set_ylabel('Time (sec)', color=color)
ax1.plot(ll, ms, 'r-')
ax1.fill_between(ll, ms - ss, ms + ss, color='r', alpha=0.2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
ax2.plot(ll, kk, 'b-')
color = 'tab:blue'
ax2.set_ylabel('Search space size', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout() 
#plt.fill_between(ll, ms - ss, ms + ss, color='g', alpha=0.2)


#plt.legend()
#plt.ylabel("Time (sec)")
#plt.xlabel("N (M=20)")
plt.show()

