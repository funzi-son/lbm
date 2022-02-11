import numpy as np
from funcs import logistic
from birbm import evaluate
from bc_data_utils import get_rules

class EBCILP_LEARN(object):
    def __init__(self,conf,dataset):
        self.dataset = dataset
        self.visNum  = dataset.ftr_dim
        self.conf    = conf
        self.crules = dataset.theory()
        
    def build_model(self):
        if self.conf.num_rules!=1:
            self.crules = get_rules(self.crules,self.conf.num_rules,self.conf.order_by)
            
        self.W_     = np .transpose(np.array(self.crules))
        self.hidNum = self.W_.shape[1]        
    

        if self.conf.confidence_level==1:
            #self.c      = self.conf.initial_c*np.random.rand(self.hidNum)
            self.c      = self.conf.initial_c*np.ones(self.hidNum)
        else:
            self.c      = self.conf.initial_c
            
        self.vb     = np.zeros((self.visNum))
        self.hb_    =  -np.sum((self.W_>0)*1.0,axis=0)+0.5

        self.W  = self.W_*self.c
        self.hb =  self.hb_*self.c
        #trn_dat,trn_inds   = self.dataset.train_dat()
        #trn_acc_gen,trn_acc_dis = evaluate(self,trn_dat,trn_inds)
        #print(trn_acc_gen)
        #input("")

    def train(self):
        lr = self.conf.lr
        epoch = count = rec_err = 0
        a_cgrad  =  a_vbgrad = 0
        max_vld_acc_gen = max_vld_acc_dis = 0
        es_vld_acc_gen  = es_vld_acc_dis = 0
        gen_es_count =dis_es_count  = 0
        while True:
            x,inx = self.dataset.next_sample()
            if x is not None:
                count+=1
                self.W  = self.W_*self.c
                self.hb =  self.hb_*self.c
                
                hi   = np.matmul(x,self.W_) + self.hb_

                hpos = logistic(hi*self.c)
                # Generative gradient
                gen_c_grad  = 0
                gen_vb_grad = 0
                
                if self.conf.alpha!=0:
                    hpos_s = hpos>np.random.uniform(size=hpos.shape)
                    #hpos_s = hpos
                    vneg   = logistic(np.matmul(hpos_s,np.transpose(self.W)) + self.vb)
                    vneg_s = vneg>np.random.uniform(size=vneg.shape)
                    #vneg_s = vneg
                    hnegi  = np.matmul(vneg_s,self.W_) + self.hb_
                    hneg   = logistic(hnegi*self.c)

                    rec_err += np.sum(np.power(x-vneg,2))
    
                    gen_c_grad = hi*hpos - hnegi*hneg
                    if self.conf.confidence_level==0: # confidence of whole program
                        c_prog_grad = np.sum(gen_c_grad)
                        gen_c_grad.fill(c_prog_grad)                    
                    #gen_vb_grad = np.mean(x - vneg_s,axis=0)

                # Discriminative gradient
                dis_c_grad  = 0
                dis_vb_grad = 0
                
                if self.conf.beta!=0:
                    xtemp = np.copy(x)
                    xtemp[:,inx] = 1
                    hi_1 = np.matmul(xtemp,self.W_) + self.hb_
                    xtemp[:,inx] = 0
                    hi_0 = np.matmul(xtemp,self.W_) + self.hb_

                    s1=self.vb[inx] + np.sum(np.log(1+np.exp(hi_1*self.c)))
                    s0=np.sum(np.log(1+np.exp(hi_0*self.c)))
                    mx = max(s1,s0)
                    s1 = s1-mx
                    s0 = s0-mx
                    
                    p1_x = np.exp(s1)/(np.exp(s1)+np.exp(s0))
                    if np.isnan(p1_x).any():
                        #epoch = self.conf.MAX_ITER
                        continue
                    p0_x = 1- p1_x

                    h_1 = logistic(hi_1*self.c)
                    h_0 = logistic(hi_0*self.c)

                    dis_vb_grad = np.array([0]*self.visNum)
                    dis_vb_grad[inx] = x[0,inx] - p1_x
                    
                    dis_c_grad  = hi*hpos - (p1_x*h_1*hi_1 + p0_x*h_0*hi_0)
                   
                    if self.conf.confidence_level==0: # confidence of whole program
                        c_prog_grad_dis = np.sum(dis_c_grad)
                        #print(c_prog_grad_dis)
                        dis_c_grad.fill(c_prog_grad_dis)
                
                c_grad  =  self.conf.alpha*gen_c_grad  + self.conf.beta*dis_c_grad
                vb_grad =  self.conf.alpha*gen_vb_grad + self.conf.beta*dis_vb_grad
                #hb_grad =  self.conf.alpha*gen_hb_grad + self.conf.beta*dis_hb_grad

                # update
                if self.conf.opt == "sgd":
                    self.c  += lr*np.squeeze(c_grad)
                else:
                    a_cgrad +=  c_grad                 
            else:
                if self.conf.opt != "sgd":
                    self.c  += lr*np.squeeze(a_cgrad/count)
                #self.vb += lr*np.squeeze(np.mean(a_vbgrad,axis=0))
                a_cgrad  =  a_vbgrad = 0

                self.W  = self.W_*self.c
                self.hb = self.hb_*self.c
                #print((np.std(self.c),np.min(self.c),np.max(self.c)))
                trn_dat,trn_inds   = self.dataset.train_dat()
                trn_acc_gen,trn_acc_dis = evaluate(self,trn_dat,trn_inds)
                vld_dat,vld_inds   = self.dataset.valid_dat()
                vld_acc_gen,vld_acc_dis = evaluate(self,vld_dat,vld_inds)
                if self.conf.verbose and epoch%100==0:
                    print("[Epoch %d] rec=%.5f trn_gen=%.5f vld_gen=%.5f trn_dis=%.5f vld_dis=%.5f"%(epoch,rec_err/count,trn_acc_gen,vld_acc_gen,trn_acc_dis,vld_acc_dis))
                count=rec_err = 0 
                epoch +=1
				
                if max_vld_acc_gen < vld_acc_gen:
                    max_vld_acc_gen = vld_acc_gen
                    gen_es_count = 0
                elif max_vld_acc_gen>vld_acc_gen:
                    gen_es_count+=1
                    if gen_es_count>=self.conf.ES_LIMIT:
                        es_vld_acc_gen = max_vld_acc_gen

                if max_vld_acc_dis < vld_acc_dis:
                    max_vld_acc_dis = vld_acc_dis
                    dis_es_count = 0
                elif max_vld_acc_dis>vld_acc_dis:
                    dis_es_count+=1
                    if dis_es_count>=self.conf.ES_LIMIT:
                        es_vld_acc_dis = max_vld_acc_dis
				
				
                if epoch>self.conf.MAX_ITER:
                    break
                
        
        if es_vld_acc_gen == 0:
            es_vld_acc_gen = max_vld_acc_gen
        if es_vld_acc_dis == 0:
            es_vld_acc_dis = max_vld_acc_dis
            
        return [max_vld_acc_gen,max_vld_acc_dis,es_vld_acc_gen,es_vld_acc_dis,vld_acc_gen,vld_acc_dis]


    def run(self):
        self.build_model()
        vld_accs=self.train()
        return vld_accs
