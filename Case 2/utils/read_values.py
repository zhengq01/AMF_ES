# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def read_values(time_index,i,filename_result,root_directory,sub_dir):
    
    ogs_file='DrawDown_{0}'.format(i)
    # ogs_file='DrawDown_00'
    args_obs=os.path.join(root_directory,sub_dir,ogs_file,filename_result)
    with open(args_obs,'r') as f:
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
    '''
    t表示取出1~t时刻的所有观测点的预测值
    '''
#    gas_file='gas_{0}'.format(i)\
    gas_file='DrawDown_00'
    args_domain=os.path.join(root_directory,sub_dir,gas_file,filename_result)
    with open(args_domain,'r') as f:
        content=f.readlines()
    x=np.zeros((0,10))   #准备提取tec文件中每个时间步输出的前5列数据（x,y,z,p1,p2）
    for i in xrange(t+1):
        content1=content[227*i:227*i+Nod_num+3]  #跳过第0个时间步，第一个时间步从第3284行开始，854会随网格划分格点及单元个数而改变
#        return content1
        content1=content1[3:]              #踢掉tec前面3行头数据
        for i in xrange(len(content1)):
            content1[i]=content1[i].split()
        content1=pd.DataFrame(content1)

        content_1=content1.iloc[obs_Num,[0,1,2,3,4,5,6,7,8,9]]
        value=content_1.values
        
        x=np.vstack((x,value))

    x=x[len(obs_Num):]   ##把0时刻的踢掉
    x=np.array(x)
    x=np.float64(x)
    return x[:,4]   

if __name__=='__main__':
    i=1
    cur_directory=os.getcwd()
    root_directory=os.path.dirname(cur_directory)
#    filename_result='DrawDown_time_POINT1.tec'
    time_all=np.loadtxt('../true_obs_time.txt')
    obs=np.loadtxt('../true_obs.txt')
#    leachate_level=read_values(time_all,i,filename_result,root_directory)
    
#    
    sub_dir='high_fidelity'
    t=50
    obs_Num=[88]
    Nod_num=402
    filename_result='DrawDown_time_POINT1.tec'
    pred=read_values(time_all,i,filename_result,root_directory,sub_dir)
    
    
#    leachate_level_true=np.array(list(np.loadtxt('../true_obs_for_train.txt'))+list(np.loadtxt('../true_obs_for_test.txt')))
    plt.plot(time_all,pred,'b',label='simulated')
    plt.scatter(time_all,obs,c='r',label='true')
    plt.legend()
    plt.show()
#   

    
    
    
    
