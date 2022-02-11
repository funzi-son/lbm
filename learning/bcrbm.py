from funcs import *
class BC_RBM(object):
    def __init__(self,conf,dataset):
        self.dataset  = dataset
        self.visNum = dataset.ftr_dim
        self.hidNum = conf.hidNum
        self.conf   = conf
        
    def build_model(self):
        # Initialise weights
        print("TODO")
    def train(self):
        print("TODO")
    
    def evaluate(self,inp,inds):
        print("TODO")
    
    def run(self):
        """ Encode """
        self.build_model()
        """ Train  """
        vld_acc = self.train()
        """ Test """
        return vld_acc
        

