import random
import numpy as np
from ilp.birbm import evaluate
from scipy.stats import bernoulli
from deap import creator,base, tools, algorithms
from bitstring import BitArray, InterpretError

class RBM():
    W_ = None
    hb_ = None
    W = None
    vb  = None
    hb  = None

rbm = RBM()

dats = None
inds = None
bits_per_cval = 0
bits_per_eval = 0
clbound = 0
chbound = 0
elbound = 0
ehbount = 0
cvals_num = 0
class EBCILP_GP(object):
    def __init__(self,conf,dataset):
        global rbm
        global dats
        global inds
        global bits_per_eval
        global bits_per_cval
        
        self.conf = conf
        
        vld_dat,vld_inds   = dataset.valid_dat()
        dats      = vld_dat
        inds      = vld_inds
        crules    = dataset.theory()
        rbm.W_    = np .transpose(np.array(crules))
        rbm.hb_ = -np.sum((rbm.W_>0)*1.0,axis=0)+0.5
        rbm.vb    = np.zeros((rbm.W_.shape[0]))

        clbound  = conf.clbound
        chbound  = conf.chbound
        elbound  = conf.elbound
        ehbound  = conf.ehbound

        bits_per_cval = conf.bits_per_cval
        bits_per_eval = conf.bits_per_eval
        
    def run(self):
        global rbm
        global cvals_num
        global bits_per_eval
        global bits_per_cval

        
        cvals_num = rbm.W_.shape[1]
        gene_length = cvals_num*bits_per_cval
        print(gene_length)
        if self.conf.use_epsilon:
            gene_length += cvals_num*bits_per_eval
        
        creator.create("FitnessMax", base.Fitness,weights=(1.0,))
        creator.create("Individual", list,fitness=creator.FitnessMax)

        ### --- what is this?
        toolbox = base.Toolbox()
        toolbox.register("binary",bernoulli.rvs,0.5)
        toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.binary,n=gene_length)
        toolbox.register("population",tools.initRepeat,list,toolbox.individual)

        ## --- what is this too?
        toolbox.register("mate",tools.cxOrdered)
        toolbox.register("mutate",tools.mutShuffleIndexes,indpb=0.6) # What does this 0.6 mean?
        toolbox.register("select",tools.selRoulette)
        toolbox.register("evaluate",crilp_gp_eval)

        ###
        population = toolbox.population(n=self.conf.population_size)

        
        r = algorithms.eaSimple(population,
                                toolbox,
                                cxpb=0.4,
                                mutpb=0.1,
                                ngen=self.conf.num_generations,
                                verbose=True)
        best = tools.selBest(population,k=1)

        rs = crilp_gp_eval(best)
        
        return rs
    
def crilp_gp_eval(ga_ind_sol):
    global conf
    global rbm
    global bits_per_cval
    global bits_per_eval
    global clbound
    global chbound
    global cvals_num
    global dats
    global ins

    cs = ind_sol_2_confidences(ga_ind_sol,
                               bits_per_cval,
                               cvals_num,
                               clbound,
                               chbound)

    # Construct RBM
    rbm.W  = rbm.W_*cs
    rbm.hb = rbm.hb_*cs
    
    # Prediction
    acc_gen,acc_dis = evaluate(rbm,dats,inds)
    
    #print(acc_gen)
    return acc_gen,
    
def ind_sol_2_confidences(ga_ind_sol,
                          bits_per_cval,
                          cvals_num,
                          clbound,
                          chbound):
    cs = np.zeros((cvals_num,))
    for i in range(cvals_num):
        inx = i*bits_per_cval
        try:
            barray = BitArray(ga_ind_sol[inx:inx+bits_per_cval])        
            cs[i] = barray.uint
        except InterpretError:
            #print(len(ga_ind_sol))
            #print(barray)
            print((inx,bits_per_cval))
            #input("")
            
            
    return to_value_range(cs,clbound,chbound,bits_per_cval)    

def ind_sol_2_epsilons(ga_ind_sol,
                       bits_per_eval,
                       evals_num,
                       esize,
                       start_offset):
    
    es = np.array((1,evals_num))
    for i in range(evals_num):
        inx = i*bits_per_val + start_offset
        barray = BitArray(ga_ind_sol[inx:inx+bits_per_eval])
        es[i] =barray.uint
    return to_value_range(es,0,1,bits_per_eval)    
                
def to_value_range(unnormalised,lbound,hbound,bit_per_eval):
    h = 2**bit_per_eval-1
    return unnormalised*(hbound-lbound)/h + lbound
    


    
