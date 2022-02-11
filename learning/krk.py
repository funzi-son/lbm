import os
import numpy as np
from ilp.bc_data_utils import *

HOME = os.path.expanduser('~')
DAT_DIR = HOME+"/WORK/Data/ILP/BC"

class BC_DAT(object):
    def __init__(self,name="krk"):
        self.data_path = DAT_DIR + name
        self.target_names,self.input_names = meta_data(self.data_path)
        if len(self.target_names)==2 and self.target_names[1][0]=="~":
            tmp = self.target_names[0]
            self.target_names[0] = self.target_names[1]
            self.target_names[1] = tmp
            
        self.sample_inx  = 0

    def load_fold(self,fold_id,input_type="unipolar"):
        # Tenfold (fold_id = 0,1, ...9)
        dim = 1+len(self.target_names)
        self.trn_dat = data2mat([self.data_path+"/trainingpositive"+str(fold_id+1)+".data",
                                 self.data_path+"/trainingnegative"+str(fold_id+1)+".data"],
                                self.target_names,self.input_names)
        self.trn_ids = [0]*self.trn_dat.shape[0]
        self.vld_dat = data2mat([self.data_path+"/testingpositive"+str(fold_id+1)+".data",
                                 self.data_path+"/testingnegative"+str(fold_id+1)+".data"],
                                self.target_names,self.input_names)

        self.vld_ids = [0]*self.vld_dat.shape[0]

    @property
    def fold_num(self):
        return 10
    
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

if __name__ =="__main__":
    dataset = KRK()
    for fid in range(10):
        dataset.load_fold(fid)
