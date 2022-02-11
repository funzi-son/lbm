import sys, glob, os
import numpy as np
from pathlib import Path
import argparse

from cnf import DIMACSReader
from cnf2rbm import RBMSAT
from gibbs import *
from femin import *

parser = argparse.ArgumentParser()
parser.add_argument('--dnf', action='store', type=str, help='path to dnf file')
parser.add_argument('--search',action='store',type=str,help='search type')
parser.add_argument('--optimizer',action='store',type=str,help='optimizer, only use with optimization approachs')
args = parser.parse_args()

if __name__=="__main__":
    fname = os.path.basename(args.dnf)
    search = "gibbs"
    optimizer = args.optimizer
    problem_dir_name = os.path.dirname(args.dnf)
    rs_dir = problem_dir_name.replace("data","results/"+search)
    if not os.path.exists(rs_dir):
        os.makedirs(rs_dir)

    log_file = os.path.join(rs_dir,fname)
    log_dir  = log_file + "_log"
    dnf_r = DIMACSReader(args.dnf)
    rbmsat = RBMSAT(dnf_r)
    if search=="gibbs":
        reasoner = Gibbs(rbmsat,log_dir)
    elif search=="gibbsrinit":
        reasoner = GibbsRandInit(rbmsat,log_dir)
    elif search=="gibbsannealing":
        reasoner = GibbsSimulatedAnnealing(rbmsat,log_dir)
    elif search=="herding":
        reasoner = Herding(rbmsat,log_dir)
    elif search=="femin":
        reasoner = FEMin(rbmsat,log_dir,optimizer=optimizer)
    elif search=="feminnn":
        reasoner = FEMinNN(rbmsat,log_dir,optimizer=optimizer)
    else:
        raise ValueError("Error")
    is_sat = reasoner.run()
    
    logger = open(log_file,"w")
    logger.write(str(dnf_r.is_sat) + " " + str(is_sat))
    logger.close()
        
