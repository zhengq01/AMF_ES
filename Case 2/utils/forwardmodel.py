# -*- coding: utf-8 -*-

# 输入参数，输出预测值
# para必须是可以直接代入ogs计算的形式，维度（input_dim,Ne）

import multiprocessing
import numpy as np
import os
from write_para import write_n
from read_values import read_values
from runexe import runexe



def forwardmodel(para,time_for_ogs_train,Nod_num,filename_n,filename_result,root_directory,sub_dir):

    Ne=para.shape[1]
    Nobs=len(time_for_ogs_train)

    for i in xrange(Ne):
        write_n(Nod_num,para[:,i],i,filename_n,root_directory,sub_dir)

    pool=multiprocessing.Pool(20)
    for i in xrange(Ne):
        if np.mod(i,10)==0:
            print 'this is {0} sample'.format(i)
        pool.apply_async(runexe,(i,root_directory,sub_dir))
    pool.close()
    pool.join()
        
    y_pred=np.zeros((Nobs,Ne))
    for i in xrange(Ne):
        y_pred[:,i]=read_values(time_for_ogs_train,i,filename_result,root_directory,sub_dir)
    return y_pred

if __name__=='__main__':
    n1_mean=0.4
    n2_mean=0.43
    n3_mean=0.52
    n4_mean=0.6
    n5_mean=0.65
    
    n_std=0.03
    num_of_layer=5
    Ne=30
    para_mean=np.array([n1_mean,n2_mean,n3_mean,n4_mean,n5_mean])
    input_dim=para_mean.shape[0]
    Fi=np.diag([n_std]*input_dim)
    
    m1=np.random.randn(input_dim,Ne)
    
    para_for_ogs=para_mean.reshape(-1,1)+np.dot(Fi,m1)
    
    time_for_ogs_train=np.loadtxt('../true_obs_time.txt')
    Nod_num=402
    sub_dir_low='low_fidelity'
    sub_dir_high='high_fidelity'
    filename_n='DrawDown_n.direct'
    filename_result='DrawDown_time_POINT1.tec'
    cur_directory=os.getcwd()
    root_directory=os.path.dirname(cur_directory)
    
    y_pred=forwardmodel(para_for_ogs,time_for_ogs_train,Nod_num,filename_n,filename_result,root_directory,sub_dir_high)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    