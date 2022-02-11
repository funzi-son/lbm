import random
import numpy as np
from ilp.birbm import evaluate
from ilp.bc_data_utils import *

class RBM():
    W_ = None
    hb_ = None
    W = None
    vb  = None
    hb  = None


class EBCILP_GPS(object): 
    def __init__(self,conf,dataset,init_c,init_e=0.5):
        self.conf = conf
        self.dataset = dataset
        self.init_c = init_c
        self.init_e = init_e
        
    def run(self):
        if self.conf.search_by=="train":
            dats,inds = self.dataset.train_dat()
        else:
            dats,inds   = self.dataset.valid_dat()
            
        crules      = self.dataset.theory()
        if self.conf.num_rules!=1:
            crules = get_rules(crules,self.conf.num_rules,self.conf.order_by)
                               
        rbm         = RBM()
        rbm.W_      = np .transpose(np.array(crules))
        rnum        = rbm.W_.shape[1]
        rbm.hb_     = -np.sum((rbm.W_>0)*1.0,axis=0)+self.init_e
        rbm.vb      = np.zeros((rbm.W_.shape[0]))

        # initialize cs
        cs = np.array([self.init_c]*rnum)
        rbm.W  = rbm.W_*cs
        rbm.hb = rbm.hb_*cs
        # get low bound accuracy
        init_acc,_ = evaluate(rbm,dats,inds)
        acc = best_acc = init_acc
        print("initial_c = %f , initial_acc= %f" % (self.init_c,init_acc))
        while True:
            for inx in range(len(cs)):
                c = best_c = cs[inx]
                for delta in np.arange(self.conf.lbound,self.conf.hbound,self.conf.search_step):
                # Construct RBM
                    cs[inx] = c+delta
                    if cs[inx]>0: #confidence values are non-negative
                        continue
                    
                    rbm.W  = rbm.W_*cs
                    rbm.hb = rbm.hb_*cs
    
                    # Prediction
                    acc_gen,acc_dis = evaluate(rbm,dats,inds)
                    if acc_gen > acc:
                        acc = acc_gen
                        best_c = cs[inx]
                cs[inx] = best_c
            if acc >best_acc:
                best_acc = acc
                print("better confidence found: %f"% best_acc)
            else:
                break

        if self.conf.search_by=="train":
            rbm.W  = rbm.W_*cs
            rbm.hb = rbm.hb_*cs
            edats,einds = self.dataset.valid_dat()
            eval_acc,_ = evaluate(rbm,edats,einds)

            
            # Just for double check, comment the below lines for faster run
            val_acc,_ = evaluate(rbm,dats,inds)
            if val_acc != best_acc:
                raise ValueError("Check the bove code")
        else:
            eval_acc = best_acc

        return best_acc,eval_acc
