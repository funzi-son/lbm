import os
import numpy as np

HOME = os.path.expanduser('~')
DAT_DIR = HOME+"/WORK/Data/ILP/BC"
class DNA(object):
    def __init__(self):
        self.data     = np.loadtxt(DAT_DIR+"/dna.csv",delimiter=",")
        self.sample_inx  = 0

    def load_fold(self,fold_id,input_type="unipolar"): # leave one out
        ix = list(range(self.data.shape[0]))
        ix.remove(fold_id)
        self.trn_dat = self.data[ix,:]
        self.trn_ids = [0]*self.trn_dat.shape[0]
        self.vld_dat = self.data[[fold_id],:]
        self.vld_ids = [0]
        if input_type=="bipolar":
            self.trn_dat = 2*self.trn_dat-1
            self.vld_dat = 2*self.vld_dat-1

    @property
    def fold_num(self):
        return self.data.shape[0]
    
    def theory(self):
        # Return the theory in form of confidence rules
        crules = None

        return crules
    
    def next_sample(self):
        if self.sample_inx >= self.trn_dat.shape[0]:
            self.sample_inx = 0
            return None,None

        x   = self.trn_dat[[self.sample_inx],:]
        self.sample_inx +=1
        inx = 0
        return x,inx
        
    def valid_dat(self):
        return self.vld_dat,self.vld_ids
    
    def train_dat(self):    
        return self.trn_dat,self.trn_ids

    @property
    def ftr_dim(self):
        return self.trn_dat.shape[1]
