from funcs import *
class BipolarRBM(object):
    def __init__(self,conf,dataset):
        self.dataset  = dataset
        self.visNum = dataset.ftr_dim
        self.hidNum = conf.hidNum
        self.conf   = conf
        
    def build_model(self):
        # Initialise weights
        self.W  = np.random.normal(scale=1/max(self.visNum,self.hidNum),size=(self.visNum,self.hidNum))  
        self.vb = np.zeros((self.visNum))
        self.hb = np.zeros((self.hidNum))

    def train(self):
        lr = self.conf.lr
        epoch=1
        rec_err = 0
        gen_err = 0
        c = True
        while True:
            x,inx = self.dataset.next(batch_size=1)
            if x is not None:
                mask = np.where(x==0)[1]
                sNum = x.shape[0]
                hi   = np.matmul(x,self.W) + self.hb
                hpos = logistic(hi)
                # Generative gradient
                gen_w_grad  = 0
                gen_vb_grad = 0
                gen_hb_grad = 0

                if self.conf.alpha!=0:
                    hpos_s = hpos>np.random.uniform(size=hpos.shape)
                    vneg   = logistic(2*(np.matmul(hpos_s,np.transpose(self.W)) + self.vb))
                    vneg_s = (vneg>np.random.uniform(size=vneg.shape))*2.0-1
                    vneg_s[0,mask] = 0
                    hneg   = logistic(np.matmul(vneg_s,self.W) + self.hb)

                    gen_w_grad  = (np.matmul(np.transpose(x),hpos) - np.matmul(np.transpose(vneg_s),hneg))/sNum
                    gen_vb_grad = np.mean(x - vneg_s,axis=0)
                    gen_hb_grad = np.mean(hpos - hneg,axis=0)

                if c:                    
                    print(x)
                    print(vneg)
                    print((x[0,inx],vneg_s[0,inx]))
                    c=False
                rec_err += np.mean(np.absolute(vneg_s - x))
                # Discriminative gradient
                dis_w_grad  = 0
                dis_vb_grad = 0
                dis_hb_grad = 0
                """
                if self.conf.beta!=0:
                    x_ = np.copy(x)
                    x_[inx] = -1*x_[inx]
                    if x[inx]==0:
                        h_0 = hi
                        h_1 = np.matmul(x_,self.W) + self.hb
                    
                        hj_0 = hpos                                        
                        hj_1 = logistic(h_1)
                    
                    else:
                        h_1 = hi
                        h_0 = np.matmul(x_,self.W) + self.hb
                    
                        hj_1 = hpos
                        hj_0 = logistic(h_0)
                        
                    p  =  condprob(h_0,h_1,self.vb[inx],inx,1)
                    p_ = 1-p

                    dis_hb_grad = (hpos - (p*pj_1 +  p_*pj_0))
                    dis_w_grad  = np.matmul(np.transpose(x),dis_hb_grad)
                    dis_w_grad[inx,:] = x[inx]*hpos - p*hj_1
                    dis_vb_grad = np.zeros((self.visNum))
                    dis_vb_grad[inx] = x[inx] - (p if x[inx]==1 else p_)
                """
                # Sparsity
                spr_w_grad  = 0
                spr_vb_grad = 0
                spr_hb_grad = 0
                if self.conf.gamma !=0:
                    print("TODO")
                # Optimise                        
                w_grad =  self.conf.alpha*gen_w_grad + self.conf.beta*dis_w_grad + self.conf.gamma*spr_w_grad
                w_grad[mask,:] = 0
                vb_grad =  self.conf.alpha*gen_vb_grad + self.conf.beta*dis_vb_grad + self.conf.gamma*spr_vb_grad
                vb_grad[mask] = 0
                hb_grad =  self.conf.alpha*gen_hb_grad + self.conf.beta*dis_hb_grad + self.conf.gamma*spr_hb_grad
                # Try adagrad later on
                self.W  += lr*w_grad
                self.vb += lr*vb_grad
                self.hb += lr*hb_grad
            else:
                print(rec_err)
                c=True
                rec_err = 0
                trn_dat,trn_head_inds = self.dataset.train_dat()
                
                trn_dis_acc,trn_gen_acc = self.evaluate(trn_dat,trn_head_inds)
                #vld_dat,vld_head_inds = self.dataset.valid_dat()
                vld_acc = 0#self.evaluate(vld_dat,vld_head_inds)

                print("[Epoch %d] trn_acc=%.5f| vld_acc=%.5f" % (epoch,trn_dis_acc,trn_gen_acc))
                # Reduce learning rate & (TODO: early stop)
                lr = lr/1.001 # Check this later
                epoch+=1
                max_vld_acc = vld_acc
                if epoch > self.conf.MAX_ITER:
                    print(trn_acc)
                    break
        return max_vld_acc
    
    def evaluate(self,inp,inds):
        dis_acc = 0
        gen_acc = 0
        for x_,inx in zip(inp,inds):
            # compute dis_acc
            x = np.copy(x_)[np.newaxis,:]
            y=x[0,inx]
            """
            x[0,inx] = -1
            h_false = np.matmul(x,self.W) + self.hb 
            x[0,inx] = 1
            h_true = np.matmul(x,self.W) + self.hb
            p_xix_rest = condprob(h_false,h_true,self.vb[inx],inx,y)
            if p_xix_rest>0.5:
                dis_acc+=1
            """
            # compute gen_acc
            #x[0,inx] = 1
            print(x)
            hid = logistic(np.matmul(x,self.W)+self.hb)
            hid = (hid > np.random.uniform(size=hid.shape))
            l=logistic(2*(np.matmul(hid,np.transpose(self.W))+self.vb))
            l = (l>np.random.uniform(size=l.shape))*2.0-1
            l = l[0,inx]
            print((y,l))
            input("")
            break
            if y==l:
                gen_acc += 1
            
        dis_acc = dis_acc/inp.shape[0]
        #input("")
        return dis_acc,gen_acc
    
    def run(self):
        """ Encode """
        self.build_model()
        """ Train  """
        vld_acc = self.train()
        """ Test """
        return vld_acc
        

