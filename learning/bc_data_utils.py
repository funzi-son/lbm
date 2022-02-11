import glob
import numpy as np

def meta_data(path):
    fs = glob.glob(path+"/*.data")
    target_names  = []
    literal_names = []
    for f in fs:
        with open(f) as fp:
            for line in fp:
                head,lstring = line.split(":")
                if head not in target_names:
                    target_names.append(head)
                
                literals = lstring[1:].replace(".\n","").split(",")
                for l in literals:                    
                    if l not in literal_names:
                        literal_names.append(l)
    return target_names,literal_names

def data2mat(filenames,target_names,input_names):
    dat = np.empty((0,len(input_names)+1))
    for f in filenames:
        with open(f) as fp:
            for line in fp:
                d = [0]*(len(input_names)+1)
                head,lstring = line.split(":")
                if target_names.index(head)==1:
                    d[0] = 1
                
                literals = lstring[1:].replace(".\n","").split(",")
                for l in literals:
                    d[input_names.index(l)+1] = 1
                dat = np.append(dat,[d],axis=0)
    return dat

def data2rules(filenames,target_names,input_names,shortest_rule=False):
    crules = np.empty((0,len(input_names)+1))
    for f in filenames:
        with open(f) as fp:
            for line in fp:
                d = [0]*(len(input_names)+1)
                head,lstring = line.split(":")

                if target_names.index(head)==0:
                    d[0] = -1                    
                if target_names.index(head)==1:
                    d[0] = 1
                
                literals = lstring[1:].replace(".\n","").split(",")
                for l in literals:
                    d[input_names.index(l)+1] = 1
                crules = np.append(crules,[d],axis=0)

    crules = np.unique(crules,axis=0)

    if shortest_rule:
        crules = prune_rules(crules)

    return crules

def prune_rules(crules):
    removed_list = []
    remain_list  = []
    lens = np.sum(crules[:,1:],axis=1)
    inds = np.argsort(lens)
    for i in range(len(inds)):
        sinx = inds[i]
        if sinx in removed_list:
            continue
        for j in range(i+1,len(inds)):
            linx = inds[j]            
            if linx not in removed_list and lens[linx] > lens[sinx] :
                if entail(crules[sinx,:],crules[linx,:]):
                    removed_list.append(linx)                    
        remain_list.append(sinx)
    ## Check
    #print(len(removed_list))
    #print(len(remain_list))
    #print(crules.shape)
    #match_sum = np.matmul(crules,np.transpose(crules))
    #self_sum  = np.sum(np.abs(crules),axis=1)
    #print(np.sum((match_sum - self_sum)==0,axis=1))
    #print(((match_sum - self_sum)==0)[6,:]*1)
    #elist = np.sum((np.transpose(match_sum) - self_sum)==0,axis=1)
    #print(np.sum(elist>1))
    #input(" ")
    return crules[remain_list,:]

def entail(r1,r2):
    if r1[0]!=r2[0]:
        return False
    return (np.sum(r1*r2)== np.sum(np.abs(r1)))

def get_rules(crules,percent,order_by):
    if order_by=="shortest":
        return shortest_rules(crules,percent)
    elif order_by=="longest":
        return longest_rules(crules,percent)
    else:
        raise ValueError("Order is wrong")
    
def shortest_rules(crules,percent):
    """
    get percent% shortest rule from crules
    """
    #raise ValueError("This has not been checked yet")
    #print(np.sort(np.sum(np.abs(crules),axis=1)))
    inds  = np.where(crules[:,0]==1)
    r     = crules[inds,1:][0]
    lens  = np.sum(np.abs(r),axis=1)
    num   = int(np.round(percent*len(lens)))
    s_pos = inds[0][np.argsort(lens)][:num]
    #print(np.sum(np.abs(crules[s_pos,1:]),axis=1))
    
    inds  = np.where(crules[:,0]==-1)
    r     = crules[inds,1:][0]
    lens  = np.sum(np.abs(r),axis=1)
    num   = int(np.round(percent*len(lens))) 
    s_neg = inds[0][np.argsort(lens)][:num]
    #print(np.sum(np.abs(crules[s_neg,1:]),axis=1))
    #input("")
    return crules[np.append(s_pos,s_neg),:]

def longest_rules(crules,percent):
    """
    get percent% longest rule from crules
    """
    #raise ValueError("This has not been checked yet")
    #print(np.sort(np.sum(np.abs(crules),axis=1)))
    inds  = np.where(crules[:,0]==1)
    r     = crules[inds,1:][0]
    lens  = np.sum(np.abs(r),axis=1)
    num   = int(np.round(percent*len(lens)))
    s_pos = inds[0][np.argsort(lens)][-num:]
    #print(np.sum(np.abs(crules[s_pos,1:]),axis=1))
    
    inds  = np.where(crules[:,0]==-1)
    r     = crules[inds,1:][0]
    lens  = np.sum(np.abs(r),axis=1)
    num   = int(np.round(percent*len(lens))) 
    s_neg = inds[0][np.argsort(lens)][-num:]
    #print(np.sum(np.abs(crules[s_neg,1:]),axis=1))
    #input("")
    return crules[np.append(s_pos,s_neg),:]
