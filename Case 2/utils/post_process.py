# -*- coding: utf-8 -*-

##查看迭代过程中的中间结果是否也有比较好的结果
##与主程序输出的图不同的是该程序输出的图最后测试阶段也有样本输出


import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from runexe import runexe
from write_para import write_Ki,write_n
from read_values import read_values
from sklearn.metrics import mean_squared_error


Ne=100
Nod_num=150
filename_Ki='DrawDown_Ki.direct'
filename_n='DrawDown_n.direct'
filename_result='DrawDown_time_POINT1.tec'
time_for_ogs_train=np.loadtxt('../true_obs_time.txt')
time_all=time_for_ogs_train
# time_for_ogs_test=np.loadtxt('../true_obs_time_stamp_for_test.txt')
# time_all=np.array(list(time_for_ogs_train)+list(time_for_ogs_test))
cur_directory=os.getcwd()
root_directory=os.path.dirname(cur_directory)

y_obs_train=np.loadtxt('../true_obs.txt')
y_obs_all=y_obs_train
# y_obs_test=np.loadtxt('../true_obs_for_test.txt')
# y_obs_all=np.array(list(y_obs_train)+list(y_obs_test))
num_of_obs_train=y_obs_train.shape[0]
num_of_obs=y_obs_train.shape[0]


##计算样本的输出
iter_index=15
para_ensem=np.loadtxt('../updated_para_{0}.txt'.format(iter_index))
para_mean=np.mean(para_ensem,axis=1)


for i in xrange(Ne):
    write_n(Nod_num,para_ensem[:,i],i,filename_n,root_directory)

	
pool=multiprocessing.Pool(20)
for i in xrange(Ne):
	pool.apply_async(runexe,(i,root_directory))
pool.close()
pool.join()

y_pred=np.zeros((num_of_obs,Ne))
for i in xrange(Ne):
	y_pred[:,i]=read_values(time_all,i,filename_result,root_directory)
	

##计算参数均值的输出
i_mean=1	
write_n(Nod_num,para_mean,i_mean,filename_n,root_directory)
runexe(i_mean,root_directory)
y_pred_mean=read_values(time_all,i_mean,filename_result,root_directory)	
	

#绘图	
for i in xrange(Ne):
    if i==Ne-1:
        plt.plot(time_all,y_pred[:,i],'g',alpha=0.2,label='predicted_sample')
    if np.any(y_pred[:,i]>10):
        continue
    plt.plot(time_all,y_pred[:,i],'g',alpha=0.2)
plt.plot(time_all,y_pred_mean,'r',label='predicted_mean')
plt.scatter(time_all,y_obs_all,c='b',label='true_obs')
rmse_train=np.sqrt(mean_squared_error(y_pred_mean[:num_of_obs_train],y_obs_all[:num_of_obs_train]))
# rmse_test=np.sqrt(mean_squared_error(y_pred_mean[num_of_obs_train:],y_obs_all[num_of_obs_train:]))
print 'para_mean:',para_mean
print 'rmse_train:',rmse_train
print 'rmse_test:',rmse_test
plt.xlabel('Time(s)')
plt.ylabel('Cumulated volume(m)')
plt.legend(loc='best')
plt.show()	
	
	
	
	