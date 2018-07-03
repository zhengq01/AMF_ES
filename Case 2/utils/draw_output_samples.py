# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from runexe import runexe
from write_para import write_Ki,write_n
from read_values import read_values

index=5
parY=np.loadtxt('../updated_para_{0}.txt'.format(index))  #得到（Nod_num_high,N_H）的parY
updated_para_ave=np.mean(parY,1)


num_of_layer=5
Ne=100
Nod_num=150


filename_Ki='DrawDown_Ki.direct'
filename_n='DrawDown_n.direct'
filename_result='DrawDown_time_POINT1.tec'
y_obs=np.loadtxt('../true_obs.txt')
Nobs=y_obs.shape[0]
time_for_ogs_train=np.loadtxt('../true_obs_time.txt')

cur_directory=os.getcwd()
root_directory=os.path.dirname(cur_directory)
sub_dir='DrawDown_files'

for i in xrange(Ne):
    write_n(Nod_num,parY[:,i],i,filename_n,root_directory,sub_dir)

pool=multiprocessing.Pool(20)
for i in xrange(Ne):
    if np.mod(i,10)==0:
            print 'this is {0} sample'.format(i)
    pool.apply_async(runexe,(i,root_directory,sub_dir))
pool.close()
pool.join()

y_pred_ensem=np.zeros((Nobs,Ne))
for i in xrange(Ne):
    y_pred_ensem[:,i]=read_values(time_for_ogs_train,i,filename_result,root_directory,sub_dir)
    
    
i_test=1
#write_Ki(Nod_num,updated_para_ave[:num_of_ki],i_test,filename_Ki,root_directory)
write_n(Nod_num,updated_para_ave,i_test,filename_n,root_directory,sub_dir)
runexe(i_test,root_directory,sub_dir)
y_hat=read_values(time_for_ogs_train,i_test,filename_result,root_directory,sub_dir)
#y_test=np.loadtxt('true_obs_for_test.txt')
error=np.sqrt(np.mean((y_hat-y_obs)**2))
print 'rmse_train:',error
#绘图 
for i in xrange(Ne):
    if i==Ne-1:
        plt.plot(time_for_ogs_train,y_pred_ensem[:,i],'g',alpha=0.2,label='predicted_sample')
    if np.any(y_pred_ensem[:,i]>10):
        continue
    plt.plot(time_for_ogs_train,y_pred_ensem[:,i],'g',alpha=0.2)
plt.plot(time_for_ogs_train,y_hat,'r',label='predicted_mean')
plt.scatter(time_for_ogs_train,y_obs,c='b',label='true_obs')

plt.legend(loc='best')
plt.show()
