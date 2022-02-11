import numpy as np
from utils.funcs import logistic
from ilp.birbm import evaluate
from ilp.bc_data_utils import get_rules

class EBCILP_INIT_LEARN(object):
    def __init__(self,conf,dataset):
        self.dataset = dataset
        self.visNum  = dataset.ftr_dim
        self.conf    = conf
        self.crules = dataset.theory()
        
    def build_model(self):
        # Knoweldge part
        if self.conf.num_rules!=1:
            self.crules = get_rules(self.crules,self.conf.num_rules,self.conf.order_by)
            
        self.W     = np .transpose(np.array(self.crules))
        self.hidNum = self.W.shape[1]
        self.hb =  -np.sum((self.W>0)*1.0,axis=0)+0.5
        # normalisation
        self.W  = self.W/(np.sqrt(self.hidNum-1)*np.max(np.abs(self.W)))
        # add randomness
        self.W  += (np.random.rand(self.W.shape[0],self.W.shape[1])*0.02 -0.01)
        #(y-(-0.01))/(0.01 - (-0.01)) = (x-0)/(1-0)
        self.vb =  np.zeros((self.visNum))
        
    def train(self):
        lr = self.conf.lr
        epoch = count = rec_err = 0
        a_wgrad = a_hbgrad = a_vbgrad = 0
        max_vld_acc_gen = max_vld_acc_dis = 0
        gen_es_count =dis_es_count  = 0
        es_vld_acc_gen = es_vld_acc_dis = 0
        
        while True:
            x,inx = self.dataset.next_sample()
            if x is not None:
                count+=1
                hi   = np.matmul(x,self.W) + self.hb
                hpos = logistic(hi)
                # Generative gradient
                gen_w_grad  = 0
                gen_vb_grad = 0
                gen_hb_grad = 0
                
                if self.conf.alpha!=0:
                    hpos_s = hpos>np.random.uniform(size=hpos.shape)
                    vneg   = logistic(np.matmul(hpos_s,np.transpose(self.W)) + self.vb)
                    vneg_s = vneg>np.random.uniform(size=vneg.shape)
                    hneg   = logistic(np.matmul(vneg_s,self.W) + self.hb)

                    rec_err += np.sum(np.power(x-vneg,2))
                    gen_w_grad = (np.matmul(np.transpose(x),hpos) - np.matmul(np.transpose(vneg_s),hneg))
                    gen_vb_grad = np.mean(x - vneg_s,axis=0)
                    gen_hb_grad = np.mean(hpos - hneg,axis=0)


                # Discriminative gradient
                dis_w_grad  = 0
                dis_vb_grad = 0
                dis_hb_grad = 0
                if self.conf.beta!=0:
                    xtemp = np.copy(x)
                    xtemp[:,inx] = 1
                    hi_1 = np.matmul(xtemp,self.W) + self.hb
                    xtemp[:,inx] = 0
                    hi_0 = np.matmul(xtemp,self.W) + self.hb

                    s1=self.vb[inx] + np.sum(np.log(1+np.exp(hi_1)))
                    s0=np.sum(np.log(1+np.exp(hi_0)))
                    mx = max(s1,s0)
                    s1 = s1-mx
                    s0 = s0-mx
                    
                    p1_x = np.exp(s1)/(np.exp(s1)+np.exp(s0))
                    if np.isnan(p1_x).any():
                        #epoch = self.conf.MAX_ITER
                        continue
                    p0_x = 1- p1_x
                    h_1 = logistic(hi_1)
                    h_0 = logistic(hi_0)
                    
                    dis_vb_grad = np.array([0]*self.visNum)
                    dis_vb_grad[inx] = x[0,inx] - p1_x
                    dis_hb_grad = hpos - (h_1*p1_x + h_0*p0_x)
                    dis_w_grad  = np.matmul(np.transpose(x),dis_hb_grad)
                    dis_hb_grad = np.mean(dis_hb_grad,axis=0)
                    
                    dis_w_grad[inx,:] = (hpos*x[0,inx] - h_1*p1_x)[0]                    
                    
                w_grad  =  self.conf.alpha*gen_w_grad  + self.conf.beta*dis_w_grad
                vb_grad =  self.conf.alpha*gen_vb_grad + self.conf.beta*dis_vb_grad
                hb_grad =  self.conf.alpha*gen_hb_grad + self.conf.beta*dis_hb_grad

                # update
                if self.conf.opt == "sgd":
                    self.W  += lr*w_grad
                    self.vb += lr*vb_grad
                    self.hb += lr*hb_grad
                else:
                    a_wgrad  += w_grad
                    a_hbgrad += hb_grad
                    a_vbgrad += vb_grad
	
          
            else:
                if self.conf.opt != "sgd":
                    self.vb  += lr*a_vbgrad/count
                    self.hb  += lr*a_hbgrad/count
                    self.W   += lr*a_wgrad/count
                    
                a_wgrad = a_hbgrad = a_vbgrad = 0
				
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
                        es_acc_gen = max_vld_acc_gen
                        
                    if max_vld_acc_dis < vld_acc_dis:
                        max_vld_acc_dis = vld_acc_dis
                        dis_es_count = 0
                    elif max_vld_acc_dis>vld_acc_dis:
                        dis_es_count+=1
                        if dis_es_count>=self.conf.ES_LIMIT:
                            es_acc_dis = max_vld_acc_dis
				
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
