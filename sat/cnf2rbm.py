import glob
import numpy as np
from cnf import DIMACSReader

def clause2sdnf(c,n_vars,conf_val,epsilon):
    sdnf = []
    c = np.array(c)
    W= None
    hb = np.array([])
    xb = np.array([])
    for i in range(len(c)):
        # variable elimination
        cclause = np.append(c[i],-1*c[i+1:])
        w = np.zeros((1,n_vars))
        s = np.sign(cclause)
        w[0,np.abs(cclause)-1] = s
        if W is None:
            W = w
        else:
            W = np.append(W,w,axis=0)
        xb = np.append(xb,[0])
        hb = np.append(hb,[-np.sum((s+1)/2)+epsilon])

        sdnf.append((cclause))

    params = {"W":W*conf_val, "hb":hb*conf_val, "xb":xb}

    return sdnf,params

def free_en(W,hb,x):
    inp = np.matmul(x,W) + hb
    #print(inp)
    fen = -np.sum(np.log(1+np.exp(inp)),axis=1)
    return fen

class RBMSAT():
    def __init__(self,cnf_reader,conf_val=3,epsilon=0.5):
        self.conf_val = conf_val
        self.epsilon = epsilon
        self.cnf_reader = cnf_reader
        self.build_model()

    def build_model(self):
        self.W  = None
        self.xb = None
        self.hb = None
        while self.cnf_reader.has_next():
            clause = self.cnf_reader.next_clause()
            sdnf,params = clause2sdnf(clause,self.cnf_reader.n_vars,self.conf_val,self.epsilon)
            if self.W is None:
                self.W  = params["W"]
                self.xb = params["xb"]
                self.hb = params["hb"]
            else:
                self.W  = np.append(self.W,params["W"],axis=0)
                self.xb = np.append(self.xb,params["xb"])
                self.hb = np.append(self.hb,params["hb"])

        self.W = np.transpose(self.W)
        #print(self.W.shape)
        #print(self.xb.shape)
        #print(self.hb.shape)
        
    def has_sat(self,x):
        hi = np.matmul(x,self.W) + self.hb
        sat_clause = (hi>0)*1.0
        total_sat_clause = np.sum(sat_clause,axis=1)
        numsat = np.max(total_sat_clause)

        assert numsat<= self.cnf_reader.n_clauses, "Number of satisfied clauses cannot larger than total clauses"
        has_sat = (numsat==self.cnf_reader.n_clauses)
        
        all_e_rank= -np.sum(hi*sat_clause,axis=1)
        e_rank = np.min(all_e_rank)
        fen = np.min(free_en(self.W,self.hb,x))
        total_sat_clause = np.max(total_sat_clause)
    
        return has_sat,e_rank,fen,total_sat_clause,all_e_rank
        
if __name__=="__main__":
    fs = glob.glob("/Users/sntran/WORK/projects/deepsymbolic/code/sat/neurosat/dimacs/test/sr5/grp1/*.*")
    for fn in fs:
        n = DIMACSReader(fn)
        rbmsat = RBMSAT(n)
        input("")
