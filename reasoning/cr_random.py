import numpy as np
import itertools
import sys
import os
import time

M = 20
N = 3
epsilon = 0.5
CVAL = 5
POOL_SIZE=500
SAMPLE_SIZE = 10000000


if len(sys.argv)==3:
    M = int(sys.argv[1])
    N = int(sys.argv[2])

def eval(samples,labels):
    err = 0
    completeness = 0
    for x in samples:
        truth = np.sum(np.sum(np.abs(labels - x),axis=1)==0)
        if truth==1:
            completeness+=1
        else:
            err +=1
    completeness = completeness/((2**N)-1)
    return completeness, err


labels = None
comb = list(itertools.product([0, 1], repeat=N))
for c in comb:
    l = np.zeros((1,M+N))
    l[0,:M] = 1
    if np.sum(c) >0:
        l[0,M:] = c
        if labels is None:
            labels = l
        else:
            labels = np.append(labels,l,axis=0)

#print(labels)
elapse_time = []
for tr in range(100):
    total_samples = 0
    startt = time.time()
    
    i=0
    while True:
        samples = np.random.randint(2,size=(100,M+N))
        samples = np.unique(samples,axis=0)
        print(samples.shape)
        input("")
        total_samples+= samples.shape[0]

        accs = np.empty((0,4))

        completeness,err = eval(samples,labels)
        accs = np.append(accs,[[i+1,completeness,err,total_samples]],axis=0)
        print((i+1,completeness,err,total_samples))
        if completeness==1:
            break
        i+=1  

    elapse_time.append(time.time()-startt)
    flname = "./results_rand/%d_%d_%.3f_%.3f"%(M,N)

    if not os.path.isdir(flname):
        os.makedirs(flname)
    rsfname = os.path.join(flname,"%d.txt"%(tr))
    np.savetxt(rsfname,accs)
    np.savetxt(os.path.join(flname,"time.txt"),elapse_time)
