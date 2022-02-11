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
    fs = glob.glob(os.path.join(d,"*.txt"))
    snum = []
    count = 0
    allrs = None
    mx = 0
    for f in fs:
        if "time" in ntpath.basename(f):
            continue
        print(f)
        rs = np.loadtxt(f)
        snum = np.append(snum,rs[:,-1])        

        if mx<rs[-1,-1]:
            mx = rs[-1,-1]
            print(f)

        if rs[-1,1]<1:
            print(f)
            raise ValueError("Not converged")

        if np.sum(rs[:,2])>0:
            print(f)
            raise ValueError("Error")
    
        v= rs[:,1][:,np.newaxis]
        if allrs is None:
            allrs = v
        else:
            allrs = np.append(allrs,np.zeros((allrs.shape[0],1)),axis=1)
            v = np.append(np.zeros((v.shape[0],allrs.shape[1]-1)),v,axis=1)
            allrs = np.append(allrs,v,axis=0)

    snum  = np.array(snum)
    inds = np.argsort(snum)
    snum = snum[inds]

    allrs = allrs[inds,:]
    for i in range(allrs.shape[0]):
        allrs[i,:] = np.amax(allrs[:i+1,:],axis=0)

    means = np.mean(allrs,axis=1)
    stds  = np.std(allrs,axis=1)

    return snum, means, stds


snums = []
means = []
stds  = []

import glob
ds = glob.glob("./results_pool/*")
Ms = []
Ns = []
for d in ds:
    mns = d[d.rfind("/")+1:].split("_")
    Ms.append(mns[0])
    Ns.append(mns[1])

    snum_,mean_,std_ = load_result(d)
    snums.append(snum_)
    means.append(mean_)
    stds.append(std_)

plt.figure(num=None, figsize=(5, 4), dpi=80, facecolor='w', edgecolor='k')
#plt.xlim([0,200000000])
    
for (snum_,mean_,std_,M,N) in zip(snums,means,stds,Ms,Ns):
    plt.plot(snum_, mean_, label='M=%s, N=%s'%(M,N))
    plt.fill_between(snum_, mean_ - std_, mean_ + std_, alpha=0.2)


plt.legend(loc=4)
plt.xlabel("Samples")
plt.ylabel("Completeness")
plt.show()

