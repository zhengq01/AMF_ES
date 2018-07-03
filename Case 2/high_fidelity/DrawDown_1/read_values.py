# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def read_values(time_index,filename_result):

    with open(filename_result,'r') as f:
        content=f.readlines()
    content=content[3:]
    for i in xrange(len(content)):
        content[i]=content[i].split()
    content=np.array(np.float64(content))
    
    # content[:,1]=-content[:,1]*3600*4.6*4.6
    content[:,1]=np.cumsum(-content[:,1]*3600*4.6*4.6)
    time_for_obs=map(lambda x:list(content[:,0]).index(x),time_index)
    volume=content[time_for_obs,1]
#    p_w=np.cumsum(-p_w*3600*4.6*4.6)
#    p_w=map(lambda x: x if x<0 else 0,p_w)
#    leachate_level=-np.array(p_w)/(1000*9.81)
    return volume
    
def read_obs_alltime(t,i,obs_Num,Nod_num,root_directory,filename_result,sub_dir):

#    gas_file='gas_{0}'.format(i)\
    gas_file='DrawDown_00'
    args_domain=os.path.join(root_directory,sub_dir,gas_file,filename_result)
    with open(args_domain,'r') as f:
        content=f.readlines()
    x=np.zeros((0,10))   
    for i in xrange(t+1):
        content1=content[227*i:227*i+Nod_num+3] 
        content1=content1[3:]              
        for i in xrange(len(content1)):
            content1[i]=content1[i].split()
        content1=pd.DataFrame(content1)

        content_1=content1.iloc[obs_Num,[0,1,2,3,4,5,6,7,8,9]]
        value=content_1.values
        
        x=np.vstack((x,value))

    x=x[len(obs_Num):]   
    x=np.array(x)
    x=np.float64(x)
    return x[:,4]   

if __name__=='__main__':
    import re
    current_directory=os.getcwd()
    time_all=np.loadtxt('true_obs_time.txt')
    filename_result='DrawDown_time_POINT1.tec'
    pred=read_values(time_all,filename_result)
    
    end=current_directory[-3:] 
    n=re.findall(r'\d',end)
    i=''.join(element for element in n)
    np.savetxt('data_all_time_{0}.txt'.format(i),pred)
     
    
    
