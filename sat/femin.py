import time, os

from gibbs import Reasoner
import tensorflow as tf


class FEMin(Reasoner):
    def __init__(self,rbm,log_dir,batch_size = 1,optimizer="adam",lr=0.01):
        super().__init__(rbm,log_dir,batch_size=batch_size)

        if optimizer=="adam":
            self.optimizer = tf.train.AdamOptimizer(lr)
        elif optimizer=="sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(lr)
        elif optimizer=="rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer(lr)
        else:
            raise ValueError("optimization is not set!!!")

    def run(self):
        with tf.Graph().as_default():
            W  = tf.convert_to_tensor(self.rbm.W,dtype=tf.float32)
            hb = tf.convert_to_tensor(self.rbm.hb,dtype=tf.float32)
            xb = tf.convert_to_tensor(self.rbm.xb,dtype=tf.float32)
            print("Run optimization: %s"%(self.optimizer))
            x = tf.get_variable("x",[1,self.rbm.cnf_reader.n_vars],initializer=tf.random_uniform_initializer(minval=0,maxval=1))  
            hi = tf.matmul(x,W)+hb
            fe = -(tf.reduce_sum(xb) + tf.reduce_sum(tf.log(1+tf.exp(hi)),axis=1))
        
            loss = fe
            
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss,tvars)
            opt_func = self.optimizer.apply_gradients(zip(grads,tvars))
            
            init     = tf.global_variables_initializer()
            saver    = tf.train.Saver()

            total_sat_clause = 0
            iter = 0
            session = tf.Session()
            start_time = time.time()
            while total_sat_clause<self.rbm.cnf_reader.n_clauses:
                session.run(init)
                _,err = session.run([opt_func,loss])
                pos = tf.cast(tf.math.greater(x,1),tf.float32)
                print(pos.get_shape())
                neg = 1-tf.cast(tf.math.less(x,0),tf.float32)
                print(neg.get_shape())
                print(err)
                x = pos + (1.0-pos)*x
                x = (1.0-neg)*x
                
                print(x.get_shape())

                xn = x.eval(session=session)
                #get xn (numpy version)
                has_sat,e_rank,fen,total_sat_clause,all_e_rank = self.rbm.has_sat(xn)

                elapsed_time = time.time()-start_time
                if iter%10==0:
                    print("[iter %d time %d] erank %.5f satcnum %d/%d freeen %.5f"%(iter,elapsed_time,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen))
                    if self.log_file is not None:
                        self.log_file.close()                    
                    lfpath = os.path.join(self.log_dir,str(iter)+".csv")
                    self.log_file  = open(lfpath,"a")
            
                self.log(iter,elapsed_time,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen)
                iter +=1
                
class FEMinNN(Reasoner):
    def __init__(self,rbm,log_dir,batch_size = 1,optimizer="adam",lr=0.1):
        super().__init__(rbm,log_dir,batch_size=batch_size)

        self.h1 = 200
        if optimizer=="adam":
            self.optimizer = tf.train.AdamOptimizer(lr)
        elif optimizer=="sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(lr)
        elif optimizer=="rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer(lr)
        else:
            raise ValueError("optimization is not set!!!")

    def run(self):
        with tf.Graph().as_default():
            W  = tf.convert_to_tensor(self.rbm.W,dtype=tf.float32)
            hb = tf.convert_to_tensor(self.rbm.hb,dtype=tf.float32)
            xb = tf.convert_to_tensor(self.rbm.xb,dtype=tf.float32)
            print("Run optimization: %s"%(self.optimizer))
            inp = tf.ones([1,1])
            fc1 = inp#tf.layers.dense(inp,self.h1,activation=tf.nn.relu)
            #fc1 = tf.nn.dropout(fc1,0.1)
            fc2  = tf.layers.dense(fc1, self.rbm.cnf_reader.n_vars,activation=tf.nn.sigmoid)
            x  = fc2#tf.nn.dropout(fc2,0.1)
            hi = tf.matmul(x,W)+hb
            #fe = -tf.reduce_mean(hi)
            fe = -(tf.reduce_sum(xb) + tf.reduce_sum(tf.log(1+tf.exp(hi)),axis=1))
        
            loss = fe
            
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss,tvars)
            opt_func = self.optimizer.apply_gradients(zip(grads,tvars))
            
            init     = tf.global_variables_initializer()
            saver    = tf.train.Saver()

            total_sat_clause = 0
            iter = 0
            session = tf.Session()
            start_time = time.time()
            while total_sat_clause<self.rbm.cnf_reader.n_clauses:
                session.run(init)
                _,err = session.run([opt_func,loss])
            
                #get xn (numpy version)
                xn = session.run(x)
                #print(xn >0.5)
                xn = 1*(xn>0.5)
                #input("")
                has_sat,e_rank,fen,total_sat_clause,all_e_rank = self.rbm.has_sat(xn)

                elapsed_time = time.time()-start_time
                if iter%10==0:
                    print("[iter %d time %d] erank %.5f satcnum %d/%d freeen %.5f"%(iter,elapsed_time,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen))
                    if self.log_file is not None:
                        self.log_file.close()                    
                    lfpath = os.path.join(self.log_dir,str(iter)+".csv")
                    self.log_file  = open(lfpath,"a")
            
                self.log(iter,elapsed_time,e_rank,total_sat_clause,self.rbm.cnf_reader.n_clauses,fen)
                iter +=1

