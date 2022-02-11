import numpy as np
from utils.funcs import logistic
from ilp.birbm import evaluate
from ilp.bc_data_utils import get_rules

class EBCILP_HYBRID_LEARN(object):
    def __init__(self,conf,dataset):
        self.dataset = dataset
        self.visNum  = dataset.ftr_dim
        self.conf    = conf
        self.crules = dataset.theory()
        
    def build_model(self):
        # Knoweldge part
        if self.conf.num_rules!=1:
            self.crules = get_rules(self.crules,self.conf.num_rules,self.conf.order_by)
            
        self.W_     = np .transpose(np.array(self.crules))
        self.hidNum = self.W_.shape[1]

        if self.conf.confidence_level==1:
            #self.c      = self.conf.initial_c*np.random.rand(self.hidNum)
            self.c = self.conf.initial_c*np.ones(self.hidNum)
        else:
            self.c = self.conf.initial_c
            
        self.hb_ =  -np.sum((self.W_>0)*1.0,axis=0)+0.5

        # Supplement part
        self.hidNumx = self.conf.hidNum
        self.Wx  =  np.random.normal(scale=1/max(self.visNum,self.hidNumx),size=(self.visNum,self.hidNumx))
        self.hbx = np.zeros((self.hidNumx))        

        # encoded part
        self.W  = np.concatenate((self.W_*self.c,self.Wx),axis=1)
        self.hb = np.concatenate((self.hb_*self.c,self.hbx))
        
        self.vb  =  np.zeros((self.visNum))
        
        
    def train(self):
        lr = self.conf.lr
        epoch = count = rec_err = 0
        a_cgrad = a_wgrad = a_hbgrad = a_vbgrad = 0
        max_vld_acc_gen = max_vld_acc_dis = 0
        gen_es_count =dis_es_count  = 0
        es_vld_acc_gen = es_vld_acc_dis = 0
        
        while True:
            x,inx = self.dataset.next_sample()
            if x is not None:
                count+=1
                self.W  = np.concatenate((self.W_*self.c,self.Wx),axis=1)
                self.hb = np.concatenate((self.hb_*self.c,self.hbx))
        
                hi     = np.matmul(x,self.W_) + self.hb_
                hpos_c = logistic(hi*self.c)
                hpos_w = logistic(np.matmul(x,self.Wx)+self.hbx)
                # Generative gradient
                gen_c_grad  = 0
                gen_w_grad  = 0
                gen_hb_grad = 0
                gen_vb_grad = 0                
                
                if self.conf.alpha!=0:
                    hpos_cs = hpos_c>np.random.uniform(size=hpos_c.shape)
                    hpos_ws = hpos_w>np.random.uniform(size=hpos_w.shape)

                    hpos_s  = np.concatenate((hpos_cs,hpos_ws),axis=1)
                    vneg    = logistic(np.matmul(hpos_s,np.transpose(self.W)) + self.vb)
                    vneg_s  = vneg>np.random.uniform(size=vneg.shape)
                    
                    hnegi   = np.matmul(vneg_s,self.W_) + self.hb_
                    hneg_c  = logistic(hnegi*self.c)
                    hneg_w  = logistic(np.matmul(vneg_s,self.Wx)+self.hbx)

                    rec_err += np.sum(np.power(x-vneg,2))

                    gen_c_grad = hi*hpos_c - hnegi*hneg_c
                    gen_w_grad = np.matmul(np.transpose(x),hpos_w) - np.matmul(np.transpose(vneg_s),hneg_w)
                    if self.conf.confidence_level==0: # confidence of whole program
                        c_prog_grad = np.sum(gen_c_grad)
                        gen_c_grad.fill(c_prog_grad)
                    
                    gen_vb_grad = np.mean(x - vneg_s,axis=0)
                # Discriminative gradient
                dis_c_grad  = 0
                dis_w_grad  = 0
                dis_hb_grad = 0
                dis_vb_grad = 0
                
                if self.conf.beta!=0:
                    xtemp = np.copy(x)
                    xtemp[:,inx] = 1
                    hi_c1 = np.matmul(xtemp,self.W_) + self.hb_
                    hi_w1 = np.matmul(xtemp,self.Wx) + self.hbx
                    xtemp[:,inx] = 0
                    hi_c0 = np.matmul(xtemp,self.W_) + self.hb_
                    hi_w0 = np.matmul(xtemp,self.Wx) + self.hbx

                    hi_1 = np.concatenate((hi_c1,hi_w1),axis=1)
                    hi_0 = np.concatenate((hi_c0,hi_w0),axis=1)
                    
                    s1 = self.vb[inx] + np.sum(np.log(1+np.exp(hi_1)))
                    s0 = np.sum(np.log(1+np.exp(hi_0)))
                    mx = max(s1,s0)
                    s1 = s1-mx
                    s0 = s0-mx
                    
                    p1_x = np.exp(s1)/(np.exp(s1)+np.exp(s0))
                    if np.isnan(p1_x).any():
                        #epoch = self.conf.MAX_ITER
                       continue
                    p0_x = 1- p1_x


                    h_c1 = logistic(hi_c1*self.c)
                    h_c0 = logistic(hi_c0*self.c)

                    h_w1 = logistic(hi_w1)
                    h_w0 = logistic(hi_w0)
                    
                    dis_vb_grad = np.array([0]*self.visNum)
                    dis_vb_grad[inx] = x[0,inx] - p1_x
                    
                    dis_c_grad  = hi*hpos_c - (p1_x*h_c1*hi_c1 + p0_x*h_c0*hi_c0)
                    dis_hb_grad = hpos_w - (h_w1*p1_x+h_w0*p0_x)
                    dis_w_grad  = np.matmul(np.transpose(x),dis_hb_grad)
                    dis_hb_grad = np.mean(dis_hb_grad,axis=0)
 
                    if self.conf.confidence_level==0: # confidence of whole program
                        c_prog_grad_dis = np.sum(dis_c_grad)
                        dis_c_grad.fill(c_prog_grad_dis)
                
                c_grad  = self.conf.alpha*gen_c_grad  + self.conf.beta*dis_c_grad
                w_grad  = self.conf.alpha*gen_w_grad  + self.conf.beta*dis_w_grad
                hb_grad = self.conf.alpha*gen_hb_grad + self.conf.beta*dis_hb_grad
                vb_grad = self.conf.alpha*gen_vb_grad + self.conf.beta*dis_vb_grad

                # update
                if self.conf.opt == "sgd":
                    self.c   += lr*np.squeeze(c_grad)
                    self.Wx  += lr*w_grad
                    self.hbx += lr*hb_grad
                    self.vb  += lr*vb_grad
                else:
                    a_cgrad  += c_grad
                    a_wgrad  += w_grad
                    a_hbgrad += hb_grad
                    a_vbgrad += vb_grad
            else:
                if self.conf.opt != "sgd":
                    self.c    += lr*np.squeeze(a_cgrad/count)#np.mean(a_cgrad,axis=0))
                    self.vb   += lr*a_vbgrad/count
                    self.hbx  += lr*a_hbgrad/count
                    self.Wx   += lr*a_wgrad/count
                    
                a_wgrad = a_hbgrad = a_vbgrad = 0

                self.W  = np.concatenate((self.W_*self.c,self.Wx),axis=1)
                self.hb = np.concatenate((self.hb_*self.c,self.hbx))
                
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

