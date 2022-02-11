from gibbs import Reasoner

class Herding(Reasoner):
    def __init__(self,rbm,log_dir,batch_size = 10000):
        super().__init__(rbm,log_dir,batch_size=batch_size)
        
    def run(self):
        
        
