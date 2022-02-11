import numpy as np
from utils.funcs import logistic
from ilp.birbm import evaluate
from ilp.bc_data_utils import get_rules

class EBCILP_SEARCH(object):
    def __init__(self,conf,dataset):
        self.dataset = dataset
        self.visNum  = dataset.ftr_dim
        self.conf    = conf
        self.crules = dataset.theory()
        
    def build_model(self):
        if self.conf.num_rules!=1:
            self.crules = get_rules(self.crules,self.conf.num_rules,self.conf.order_by)
        
        self.W_     = np.transpose(np.array(self.crules))
        self.hidNum = self.W_.shape[1]
        self.c      = np.array([self.conf.initial_c]*self.hidNum) 
        self.vb     = np.zeros((self.visNum))
        if hasattr(self.conf,"epsilon"):
            epsilon = self.conf.epsilon
        else:
            epsilon = 0.5            
        self.hb_    = -np.sum((self.W_>0)*1.0,axis=0)+epsilon

    def train(self): 
        self.W  = self.W_*self.c
        self.hb =  self.hb_*self.c                                        
        trn_dat,trn_inds   = self.dataset.train_dat()
        trn_acc_gen,trn_acc_dis = evaluate(self,trn_dat,trn_inds)    
                
        vld_dat,vld_inds   = self.dataset.valid_dat()
        vld_acc_gen,vld_acc_dis = evaluate(self,vld_dat,vld_inds)
            
        return trn_acc_gen,trn_acc_dis,vld_acc_gen,vld_acc_dis

    """    
    def evaluate(self,data,inds):
        ygen = []
        ydis = []
        y    = []
        for x,inx in zip(data,inds):
            # reconstruction
            xo = np.copy(x)
            y.append(x[inx])
            xo[inx] = 0.5
            h = logistic(np.matmul(xo,self.W) + self.hb)
            #h= h>np.random.uniform(size=hpos.shape)
            x_  = np.matmul(h,np.transpose(self.W)) + self.vb
            ygen.append(x_[inx]>0)
            # conditional - TODO
            xo[inx] = 0
            e0 = np.sum(np.log(1+np.exp(np.matmul(xo,self.W)+self.hb)))
            xo[inx] = 1
            e1 = np.sum(np.log(1+np.exp(np.matmul(xo,self.W)+self.hb)))+self.vb[inx]
            # print(e1,e0)
            ydis.append(e1>=e0)
        acc_gen =  np.mean([yi==ygeni for yi,ygeni in zip(y,ygen)])
        acc_dis =  np.mean([yi==ydisi for yi,ydisi in zip(y,ydis)])
        
        return acc_gen, acc_dis
    """   
    def run(self):
        self.build_model()
        all_accs=self.train()
        return all_accs
