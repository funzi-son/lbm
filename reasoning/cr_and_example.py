import numpy as np
import itertools
import sys
import os

M = 20
N = 3
epsilon = 0.5
CVAL = 5
SAMPLE_SIZE = 10000000


eval_list = [5,10,100,1000,100000,
             1000000,2000000,3000000,4000000,5000000,
             6000000,7000000,8000000,9000000,
             10000000,20000000,30000000,50000000,
             60000000,70000000,80000000,90000000,
             100000000,200000000,300000000,400000000,500000000,
             600000000,700000000,800000000]

if len(sys.argv)==6:
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    epsilon = float(sys.argv[3])
    CVAL = float(sys.argv[4])
    SAMPLE_SIZE = int(sys.argv[5])

print((M,N,epsilon,CVAL))

def free_en(W,hb,x):
    inp = np.matmul(x,W) + hb
    #print(inp)
    fen = np.sum(np.log(1+np.exp(inp)),axis=1)
    return fen
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

W = None
hb = []
for n_ in range(N-1):
    w = np.zeros((1,M+N))
    w[0,:M] = 1
    w[0,M:M + N - n_]  = -1
    w[0,M+N-n_-1] = 1
    if W is None:
        W = w
    else:
        W = np.append(W,w,axis=0)

    b = -(M+1) + epsilon 
    hb.append(b)

w = np.zeros((1,M+N))
w[0,:M+1] = 1    
W = np.append(W,w,axis=0)
b = -(M+1) + epsilon 
hb.append(b)

hb = np.array(hb)*CVAL
W = np.transpose(W)*CVAL
#
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

for tr in range(100):
    x = np.ones((1,M+N))*0.5

    samples = np.empty((0,M+N))
    accs = np.empty((0,3))
    for i in range(SAMPLE_SIZE):
        hinp = np.matmul(x,W) + hb
        h    = 1/(1+np.exp(-hinp))
        h    = (h>np.random.uniform(0,1,h.shape))*1
    
        x = np.matmul(h,np.transpose(W))
        x = 1/(1+np.exp(-x))
        #x = (x>0.5)*1
        x  = (x>np.random.uniform(0,1,x.shape))*1

        fen = free_en(W,hb,x)
    
        #truth = np.sum(np.sum(np.abs(labels - x),axis=1)==0)
    
        if fen>=np.log(1+np.exp(epsilon*CVAL)):
            samples = np.append(samples,x,axis=0)
            if samples.shape[0]>1:
                samples = np.unique(samples,axis=0)
            
        if i+1 in eval_list:
            completeness,err = eval(samples,labels)
            accs = np.append(accs,[[i+1,completeness,err]],axis=0)
            print((i+1,completeness,err))

    flname = "./results/%d_%d_%.3f_%.3f"%(M,N,epsilon,CVAL)
    if not os.path.isdir(flname):
        os.makedirs(flname)
    flname = os.path.join(flname,"%d.txt"%(tr))
    np.savetxt(flname,accs)
