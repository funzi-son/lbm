import os
import numpy as np

HOME = os.path.expanduser('~')
DAT_DIR = HOME+"/WORK/Data/ILP/BC"
class ILP_DAT(object):
    def __init__(self,fold=1,name="krk"):
        trn_pos_file= DAT_DIR + "/"+name + "/trainingpositive"+str(fold)+".data"
        trn_neg_file= DAT_DIR + "/"+name + "/trainingnegative"+str(fold)+".data"
        vld_pos_file= DAT_DIR + "/"+name + "/testingpositive"+str(fold)+".data"
        vld_neg_file= DAT_DIR + "/"+name + "/testingnegative"+str(fold)+".data"        

        preds,head_inds = meta([trn_pos_file,trn_neg_file,vld_pos_file,vld_neg_file])
        self.cf_rules = hornbc2crules([trn_pos_file,trn_neg_file],preds)
        self.trn_dat,self.trn_head_inds  = hornbc2data([trn_pos_file,trn_neg_file],preds)
        self.vld_dat,self.vld_head_inds  = hornbc2data([vld_pos_file,vld_neg_file],preds)

        self.end_inx = 0

    def theory(self):
        return self.cf_rules
    
    def next(self,batch_size=0):
        if batch_size == 0:
            batch_size = self.trn_dat.shape[0]
            
        self.start_inx = self.end_inx
        self.end_inx   = min(self.start_inx + batch_size,self.trn_dat.shape[0])
        if self.start_inx >= self.trn_dat.shape[0]:
            #print("All batches done")
            self.end_inx = 0
            return None,None

        batch_x    = self.trn_dat[self.start_inx:self.end_inx,:]
        batch_inds = self.trn_head_inds[self.start_inx:self.end_inx]
        
        return batch_x,batch_inds
        
    def valid_dat(self):
        return self.vld_dat,self.vld_head_inds
    
    def train_dat(self):
        return self.trn_dat,self.trn_head_inds

    @property
    def ftr_dim(self):
        return self.trn_dat.shape[1]

def meta(files):
    preds = []
    head_inds = []
    for f in files:
        print("reading meta from %s" % f)
        r = open(f)
        while True:
            line = r.readline()
            if not line:
                break
            #print(line)
            strs = line.split(":")
            head = str.replace(strs[0],"~","")
            if head not in preds:
                head_inds.append(len(preds))
                preds.append(head)

            #print(strs)
            strs = strs[1].split(",")
            for s in strs:
                s = str.replace(s,"~","")
                s = str.replace(s,"-","")
                if s not in preds:
                    preds.append(s)
            
    return preds,head_inds

def hornbc2crules(files,preds):
    return None

def hornbc2data(files,preds):
    data = None
    vlen = len(preds)
    head_inds = []
    for f in files:
        print("convert hornbc to data reading %s" % f)
        r = open(f)
        while True:
            line = r.readline()
            if not line:
                break
            vec = np.zeros((1,vlen))
            strs = line.split(":")
            head = strs[0]
            if head[0]=="~":
                inx = preds.index(head[1:])
                vec[0,inx] = -1
            else:
                inx = preds.index(head)
                vec[0,inx] = 1

            head_inds.append(inx)
            #print(strs)
            strs = strs[1].split(",")
            for s in strs:
                s = str.replace(s,"-","")
                if s[0]=="~":
                    vec[0,preds.index(s[1:])] = -1
                else:
                    vec[0,preds.index(s)] = 1
            
            if data is None:
                data = vec
            else:
                data = np.append(data,vec,axis=0)
    return data, head_inds
if __name__=="__main__":
    indices = []
    for fold in range(1,11):
        ilp_dat  = ILP_DAT(fold=fold,name="alzCho")
        while True:
            x,ids = ilp_dat.next(batch_size=1)
            if x is None:        
                break
            indices.append(ids)
    print(np.unique(indices))
