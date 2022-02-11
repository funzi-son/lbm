import numpy as np
from funcs import logistic

class BiRBM(object):
    def __init__(self,conf,dataset):
        self.dataset = dataset
        self.visNum  = dataset.ftr_dim
        self.hidNum  = conf.hidNum
        self.conf    = conf

    def build_model(self):
        self.W  = np.random.normal(scale=1/max(self.visNum,self.hidNum),size=(self.visNum,self.hidNum))
        self.vb = np.zeros((self.visNum))
        self.hb = np.zeros((self.hidNum))

    def train(self):
        lr = self.conf.lr
        epoch = count = rec_err = 0
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
                    #print((hpos*x[0,inx] - h_1*p1_x)[0])
                    #print(dis_w_grad[inx,:])
                    #print(dis_w_grad[1,:])
                    dis_w_grad[inx,:] = (hpos*x[0,inx] - h_1*p1_x)[0]
                    #print(dis_w_grad[inx,:])
                    #print(dis_w_grad[1,:])
                    #input("")
                    
                w_grad  =  self.conf.alpha*gen_w_grad  + self.conf.beta*dis_w_grad
                vb_grad =  self.conf.alpha*gen_vb_grad + self.conf.beta*dis_vb_grad
                hb_grad =  self.conf.alpha*gen_hb_grad + self.conf.beta*dis_hb_grad

                
                self.W  += lr*w_grad
                self.vb += lr*vb_grad
                self.hb += lr*hb_grad
                                
            else:
                #if epoch>self.conf.MAX_ITER/2:
                #    lr = lr/(1+0.01)
                trn_dat,trn_inds   = self.dataset.train_dat()
                trn_acc_gen,trn_acc_dis = evaluate(self,trn_dat,trn_inds)
                if self.conf.verbose and epoch%100==0:
                    print("[Epoch %d] rec=%.5f trn_gen=%.5f trn_dis=%.5f"%(epoch,rec_err/count,trn_acc_gen,trn_acc_dis))
                count=rec_err = 0                
                epoch +=1

                if epoch>self.conf.MAX_ITER:
                    #if trn_acc_dis<0.9:
                    #    self.conf.MAX_ITER += 100
                    #    lr = self.conf.lr
                    #else:
                    break
                
        vld_dat,vld_inds   = self.dataset.valid_dat()
        vld_acc_gen,vld_acc_dis = evaluate(self,vld_dat,vld_inds)
            
        return vld_acc_gen,vld_acc_dis

    def run(self):
        self.build_model()
        vld_accs = self.train()
        return vld_accs

def evaluate(rbm,data,inds):
    ygen = []
    ydis = []
    y    = []
    for x,inx in zip(data,inds):
        # reconstruction
        xo = np.copy(x)
        y.append(x[inx])
        xo[inx] = 0.5
        h = logistic(np.matmul(xo,rbm.W) + rbm.hb)
        #print(h.shape)
        #input("")
        #h= h>np.random.uniform(size=hpos.shape)
        x_  = np.matmul(h,np.transpose(rbm.W)) + rbm.vb
        ygen.append(x_[inx]>0)
        
        # conditional - TODO
        xo[inx] = 0
        e0 = np.sum(np.log(1+np.exp(np.matmul(xo,rbm.W)+rbm.hb)))
        xo[inx] = 1
        e1 = np.sum(np.log(1+np.exp(np.matmul(xo,rbm.W)+rbm.hb)))+rbm.vb[inx]
        # print(e1,e0)
        ydis.append(e1>=e0)
        
    acc_gen =  np.mean([yi==ygeni for yi,ygeni in zip(y,ygen)])
    acc_dis =  np.mean([yi==ydisi for yi,ydisi in zip(y,ydis)])
    
    return acc_gen, acc_dis
   
