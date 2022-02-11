import sys
import os
import numpy as np

from ebcilp_learn import EBCILP_LEARN
from dna   import DNA
from bc_dat  import BC_DAT

prune   = False

if prune:
        EXP_DIR = "./results/ilp_shortest_rules/"
else:
        EXP_DIR = "./resultss/ilp/"

dataset = sys.argv[1] #datalist  = ["muta","krk","uwcse_i1","alzAmi","alzSco","alzCho","alzTox","alzAmi"]
lr     = float(sys.argv[2])
alpha  = float(sys.argv[3])
beta   = float(sys.argv[4])
num_rules = float(sys.argv[5])
order_by  = "shortest"
opt="sgd"

class Config():
    MAX_ITER = 300
    ES_LIMIT = 50
    initial_c = .1
    confidence_level = 1
    opt = "sgd"
    
    num_rules = 1
    order_by  = "shortest"
    
    verbose=True

input_type = "unipolar"

def run(trial,dataset,lr,alpha,beta,opt,num_rules,order_by):
    conf = Config()
    log_file = EXP_DIR + "/"+ dataset + "/EBCILP_LEARN/opt"+opt+"_level"+str(conf.confidence_level)+"_"+str(num_rules)+"_by"+order_by
    
    if not os.path.isdir(log_file):
        os.makedirs(log_file)
        
    log_file = log_file + "/clevel"+str(conf.confidence_level)+"_lr"+str(lr)+"_alpha"+str(alpha)+"_beta"+str(beta) + "_trial"+ str(trial) + ".txt"
    if os.path.isfile(log_file):
        print("exist"+log_file)
        return

    conf.num_rules = num_rules
    conf.order_by  = order_by
    conf.lr        = lr
    conf.alpha     = alpha
    conf.beta      = beta
    
    dataset = BC_DAT(dataset,prune=prune)
    accs = None

    for fold in range(dataset.fold_num):
        print("Run fold: %d"%fold)
        dataset.load_fold(fold,input_type=input_type)
        model = EBCILP_LEARN(conf,dataset)
        acc = model.run()
        if accs is None:
                accs = np.array([acc])
        else:
                accs = np.append(accs,[acc],axis=0)
    print(np.mean(accs,axis=0))
    accs = np.append(accs,[np.mean(accs,axis=0)],axis=0)
    np.savetxt(log_file,accs,delimiter=",")

def main(trial):
    run(trial,dataset,lr,alpha,beta,opt,num_rules,order_by)

               
if __name__ == "__main__":
    for trial in range(1):
        main(trial)
    
