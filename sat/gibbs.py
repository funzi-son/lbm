import numpy as np
import os
import time

class Reasoner():
    def __init__(self,rbm,log_dir,batch_size=10000):
        self.rbm = rbm
        self.log_dir = log_dir
        self.batch_size = batch_size
        #self.trackers = {"erank":[],
        #                 "satcnum":[],
        #                 "freeen":[]}

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = None

    def log(self,iter,elapsed_time,e_rank,total_sat_clause,n_clauses,fen):
        line = str(iter) + "," \
            + str(elapsed_time) + "," \
            + str(e_rank) + "," \
            + str(total_sat_clause) + ","\
            + str(n_clauses) + "," \
            + str(fen) + "\n"

        self.log_file.write(line)
        
class Gibbs(Reasoner):
    def __init__(self,rbm,log_dir,batch_size = 10000):
        super().__init__(rbm,log_dir,batch_size=batch_size)
        

    def run(self,store=False):
        x = 0.5*np.ones((self.batch_size,self.rbm.W.shape[0]))
        iter = 0
        start_time = time.time()
        while True:
            h = np.matmul(x,self.rbm.W) + self.rbm.hb
            h = 1/(1+np.exp(-h))
            h = (h>np.random.uniform(0,1,h.shape))*1
            
            xn = np.matmul(h,np.transpose(self.rbm.W))
            xn = 1/(1+np.exp(-xn))
            xn = (xn>np.random.uniform(0,1,xn.shape))*1
            
            
            has_sat,e_rank,fen,total_sat_clause,_ = self.rbm.has_sat(xn) 

            x = xn
            elapsed_time = time.time()-start_time
            if iter%10000==0:
                print("[iter %d time %d] erank %.5f satcnum %d/%d freeen %.5f"%(iter,elapsed_time,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen))
                if self.log_file is not None:
                    self.log_file.close()                    
                lfpath = os.path.join(self.log_dir,str(iter)+".csv")
                self.log_file  = open(lfpath,"a")
                
                if store and samples.shape[0]>0:
                    spath = os.path.join(self.log_dir,str(iter)+".pkl.gzs")
                    
            
            self.log(iter,elapsed_time,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen)
            iter+=1
            if has_sat:
                return True
            #else:
            #    self.trackers["erank"].append(e_rank)
            #    self.trackers["satcnum"].append(total_sat_clause)
            #    self.trackers["freeen"].append(fen)

            #
            #print("Yay, getting here ...")
            #input("")

#######################################################################
class GibbsRandInit(Reasoner):
    def __init__(self,rbm,log_dir,batch_size = 10000):
        super().__init__(rbm,log_dir,batch_size=batch_size)
        
    def run(self):
        x = np.random.randint(2,size=(self.batch_size,self.rbm.W.shape[0]))
        iter = 0
        start_time = time.time()
        while True:
            h = np.matmul(x,self.rbm.W) + self.rbm.hb
            h = 1/(1+np.exp(-h))
            h = (h>np.random.uniform(0,1,h.shape))*1
            
            xn = np.matmul(h,np.transpose(self.rbm.W))
            xn = 1/(1+np.exp(-xn))
            xn = (xn>np.random.uniform(0,1,xn.shape))*1
            
            
            has_sat,e_rank,fen,total_sat_clause,_ = self.rbm.has_sat(xn) 

            x = xn
            elapsed_time = time.time()-start_time
            if iter%10000==0:
                print("[iter %d time %d] erank %.5f satcnum %d/%d freeen %.5f"%(iter,elapsed_time,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen))
                if self.log_file is not None:
                    self.log_file.close()                    
                lfpath = os.path.join(self.log_dir,str(iter)+".csv")
                self.log_file  = open(lfpath,"a")
            
            self.log(iter,elapsed_time,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen)
            iter+=1
            if has_sat:
                return True
#######################################################################        
class GibbsSimulatedAnnealing(Reasoner):
    def __init__(self,rbm,log_dir,batch_size = 10000):
       super().__init__(rbm,log_dir,batch_size=batch_size)

    def run(self):
        print("Run Gibbs sampling with simulated annealing")
        x = 0.5*np.ones((self.batch_size,self.rbm.W.shape[0]))
        T = 100
        iter = 0
        prv_e_rank = None
        start_time = time.time()
        while True:
            h = np.matmul(x,self.rbm.W) + self.rbm.hb
            h = 1/(1+np.exp(-h))
            h = (h>np.random.uniform(0,1,h.shape))*1

            
            xn = np.matmul(h,np.transpose(self.rbm.W))
            xn = 1/(1+np.exp(-xn))
            xn = (xn>np.random.uniform(0,1,xn.shape))*1
                    
            has_sat,e_rank,fen,total_sat_clause,all_e_rank = self.rbm.has_sat(xn) 
            
            T = T*0.9
            elapsed_time = time.time()-start_time
            if iter%10000==0:
                print("[iter %d time %d] erank %.5f satcnum %d/%d freeen %.5f"%(iter,elapsed_time,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen))
                if self.log_file is not None:
                    self.log_file.close()                    
                lfpath = os.path.join(self.log_dir,str(iter)+".csv")
                self.log_file  = open(lfpath,"a")
            
            self.log(iter,elapsed_time,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen)
            iter+=1

            if has_sat:
                return True
            else:
                if prv_e_rank is None:
                    prv_e_rank = all_e_rank
                    continue
                #print(np.append([all_e_rank],[prv_e_rank],axis=0))
                accept = all_e_rank<prv_e_rank
                acc_inds = np.where(accept)[0]  
                #print(acc_inds)
                if T>=0.000001:# Don't jump if temperature is too small
                    jump = np.exp(-(all_e_rank - prv_e_rank)/T)
                    if not np.isinf(jump).any() and not np.isnan(jump).any():
                        jump = jump > np.random.uniform(0,1,all_e_rank.shape)
                        acc_02_inds = np.where(np.logical_and(np.logical_not(accept),jump))[0]
                        acc_inds = np.append(acc_inds,acc_02_inds,axis=0)
                #print(acc_inds)
                x[acc_inds,:] = xn[acc_inds,:] 
                prv_e_rank[acc_inds] = all_e_rank[acc_inds]
                #print(prv_e_rank)
                #input("")
            #print("[iter %d] erank %.5f satcnum %d/%d freeen %.5f"%(iter,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen))
            #input("")
