# -*- coding: utf-8 -*-

import numpy as np
import os


def write_Ki(Nod_num,para,filename_Ki):
    para=np.squeeze(para)
    if para[0]<0:  
        para=np.exp(para)
    value_list={}   
    num_of_layer=para.shape[0]
    layer_points=Nod_num/num_of_layer
    for i in xrange(num_of_layer):
        for k in xrange(i*layer_points,(i+1)*layer_points):
            value_list[k]=para[i]   
            
    value_list_to_str=[]    
    for i,j in value_list.iteritems():
        var=' '.join([str(i),str(j),'\n']) 
        value_list_to_str.append(var)
    with open(filename_Ki,'w') as f:
        for line in value_list_to_str:
            f.write(line)
            pass
        f.write('#STOP')
    
def write_n(Nod_num,para,filename_n):
    para=np.squeeze(para)
    value_list={}   
    num_of_layer=para.shape[0]
    layer_points=Nod_num/num_of_layer
    for i in xrange(num_of_layer):
        for k in xrange(i*layer_points,(i+1)*layer_points):
            value_list[k]=para[i]     
    
    if len(value_list) != Nod_num: 
        for k in xrange(len(value_list),Nod_num):
            value_list[k]=para[-1]

    
    value_list_to_str=[]   
    for i,j in value_list.iteritems():
        var=' '.join([str(i),str(j),'\n'])  
        value_list_to_str.append(var)
    with open(filename_n,'w') as f:
        for line in value_list_to_str:
            f.write(line)
            pass
        f.write('#STOP')
        
        
if __name__=='__main__':
    import scipy.io as si
    import numpy as np
    Nod_num=402
    filename_KI='DrawDown_n.direct'
    para=si.loadmat('parameter.mat')
    para=para['parameter'].astype(float)
    para=np.squeeze(para)
    write_n(Nod_num,para,filename_KI)

    
    