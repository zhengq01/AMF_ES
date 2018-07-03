# -*- coding: utf-8 -*-

##每个维度独立构造GP
import GPy
import numpy as np
from sklearn import preprocessing
import time

def train_multifidelity_GP(x_train_high,y_train_high,x_train_low,y_train_low):
    # x_train_high=para[0]
    # y_train_high=para[1]
    # x_train_low=para[2]
    # y_train_low=para[3]
    # if np.mod(i,10)==0:
        # print 'this is the {0} dimension'.format(i)
    input_dim=x_train_high.shape[1]
    start_1=time.time()
    '''Train level 1'''
    k1=GPy.kern.Matern52(input_dim,ARD=False)
    scaler=preprocessing.StandardScaler().fit(y_train_low.reshape(-1,1))
    y_train_low_scaled=scaler.transform(y_train_low.reshape(-1,1))
    model1 = GPy.models.GPRegression(X=x_train_low, Y=y_train_low_scaled, kernel=k1)
    
    model1[".*Gaussian_noise"] = model1.Y.var()*0.1
    model1[".*Gaussian_noise"].fix()
    
    model1.optimize(max_iters = 2000)
    
    mu1,v1=model1.predict(x_train_high)
    # print 'time_1:',time.time()-start_1

    start_2=time.time()
    '''Train level 2'''
    XX = np.hstack((x_train_high, mu1))
    
    # 新造的核还得再考虑，active_dimension
    k2 = GPy.kern.Matern52(1,active_dims=[input_dim])*GPy.kern.Matern52(input_dim,active_dims=np.arange(input_dim)) + GPy.kern.Matern52(input_dim,active_dims=np.arange(input_dim))
    y_train_high_scaled=scaler.transform(y_train_high.reshape(-1,1))
    model2 = GPy.models.GPRegression(X=XX, Y=y_train_high_scaled, kernel=k2)

    model2[".*Gaussian_noise"] = model2.Y.var()*0.01
    model2[".*Gaussian_noise"].fix()

    model2.optimize(max_iters = 2000)
    # print 'time_2:',time.time()-start_2
    return [model1,model2],scaler

def predict_multifidelity_GP(model_list,Nts,x_test_high,scaler):
    model1=model_list[0]
    model2=model_list[1]
    
    start_3=time.time()
    '''Predict at test points'''
    # sample f_1 at xtest
    nsamples = 10
    mu_1, C_1 = model1.predict(x_test_high, full_cov=True)
    Z = np.random.multivariate_normal(mu_1.flatten(),C_1,nsamples)
    # print 'time_3:',time.time()-start_3
    
    start_4=time.time()
    # push samples through f_2
    tmp_m = np.zeros((nsamples,Nts))
    tmp_v = np.zeros((nsamples,Nts))
    for j in range(0,nsamples):
        mu, v = model2.predict(np.hstack((x_test_high, Z[j,:][:,None]))) #predict得到的是Nts*1的结果
        tmp_m[j,:] = mu.flatten()
        tmp_v[j,:] = v.flatten()
    # print 'time_4:',time.time()-start_4
    # get posterior mean and variance
    mean = np.mean(tmp_m, axis = 0)[:,None]
    var = np.mean(tmp_v, axis = 0)[:,None]+ np.var(tmp_m, axis = 0)[:,None]
    var = np.abs(var)
    var=var.ravel()
    
    mean_rescaled=scaler.inverse_transform(mean.ravel())
    
#    error = np.linalg.norm(y_original_high[:,i] - mean_rescaled)/np.linalg.norm(y_original_high[:,i])
    return mean_rescaled,var



if __name__=='__main__':
    import scipy.io as si
    import os
    import matplotlib.pyplot as plt
    from forwardmodel import forwardmodel
    from KLexpansion import KLexpansion
    data_ensem=si.loadmat('../sample.mat')
    # Fi_high=np.loadtxt('../Fi_high.txt')
    # Fi_low=np.loadtxt('../Fi_low.txt')
    ki_mean=2.9e-12
    N_H=70
    N_L=120
    Nts=20
    filename_result='gas_domain_quad.tec'
    filename_KI='gas_Ki.direct'
    time_step=10
    output_dimension=16
    kl_term=25
    Lx=60.0
    Ly=60.0
    deltax=Lx/3.0
    deltay=Ly/3.0
    dx_high=1.
    dy_high=1.
    dx_low=3
    dy_low=3
    m_high=int(Ly/dy_high+1)
    n_high=int(Lx/dx_high+1)
    m_low=int(Ly/dy_low+1)
    n_low=int(Lx/dx_low+1)
    sigma=0.6
    Nobs=output_dimension*time_step
    sub_dir_low='low_fidelity'
    sub_dir_high='high_fidelity'
    obs_Num_high=[555,570,591,603,1287,1302,1323,1335,2568,2583,2604,2616,3117,3132,3153,3165]
    Nod_num_high=3721
    obs_Num_low=[65,70,77,81,149,154,161,165,296,301,308,312,359,364,371,375]
    Nod_num_low=441
    # Fi_high_=KLexpansion(m_high,n_high,Lx,Ly,deltax,deltay,sigma)
    # Fi_high=Fi_high_[:,:kl_term]
    Fi_high=np.loadtxt('../Fi_high.txt')
    # Fi_low_=KLexpansion(m_low,n_low,Lx,Ly,deltax,deltay,sigma)
    # Fi_low=Fi_low_[:,:kl_term]
    Fi_low=np.loadtxt('../Fi_low.txt')
    
    cur_directory=os.getcwd()
    cur_directory=os.path.dirname(cur_directory)
    

    x_train_high=np.random.randn(kl_term,N_H)
    x_train_high_for_ogs=np.log(ki_mean)+np.dot(Fi_high,x_train_high)
    x_train_high_for_ogs=np.exp(x_train_high_for_ogs)
    x_train_high=x_train_high.T
    x_train_low=np.random.randn(kl_term,N_L)
    x_train_low_for_ogs=np.log(ki_mean)+np.dot(Fi_low,x_train_low)
    x_train_low_for_ogs=np.exp(x_train_low_for_ogs)
    x_train_low=x_train_low.T
    x_test_high=np.random.randn(kl_term,Nts)
    x_test_high_for_ogs=np.log(ki_mean)+np.dot(Fi_high,x_test_high)
    x_test_high_for_ogs=np.exp(x_test_high_for_ogs)
    x_test_high=x_test_high.T
    
    y_train_high=forwardmodel(x_train_high_for_ogs,time_step,obs_Num_high,Nod_num_high,filename_KI,filename_result,cur_directory,sub_dir_high)
    y_train_low=forwardmodel(x_train_low_for_ogs,time_step,obs_Num_low,Nod_num_low,filename_KI,filename_result,cur_directory,sub_dir_low)
    y_original_high=forwardmodel(x_test_high_for_ogs,time_step,obs_Num_high,Nod_num_high,filename_KI,filename_result,cur_directory,sub_dir_high)
    y_train_high=y_train_high.T
    y_train_low=y_train_low.T 
    y_original_high=y_original_high.T
    
    
    
    model_ensem=[]
    scaler_ensem=[]
    y_pred=np.zeros((Nts,Nobs))
    var_pred=np.zeros(y_pred.shape)
    start=time.time()
    for i in range(Nobs):
        print 'this is {0} dimension'.format(i)
        model_list,scaler_for_test=train_multifidelity_GP(x_train_high,y_train_high[:,i],x_train_low,y_train_low[:,i])
        model_ensem.append(model_list)
        scaler_ensem.append(scaler_for_test)
        y_pred[:,i],var_pred[:,i]=predict_multifidelity_GP(model_list,Nts,x_test_high,scaler_for_test)
    
    elapsed_time=time.time()-start
    print 'elapsed_time:',elapsed_time
    
    y_original=y_original_high.reshape(-1,1)
    y_predict=y_pred.reshape(-1,1)    
    print 'corrcoef',np.corrcoef(y_original.ravel(),y_predict.ravel())[0,1]
    # print 'elpased time of multifidelity data',time.time()-start
    plt.scatter(y_original,y_predict)    
    plt.plot([y_original.min()-100,y_original.max()+100],[y_predict.min()-100,y_predict.max()+100],'r',lw=2)
    plt.xlabel('Original')
    plt.ylabel('Surrogate')
    plt.show()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

