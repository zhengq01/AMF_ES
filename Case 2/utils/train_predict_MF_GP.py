# -*- coding: utf-8 -*-

##每个维度独立构造GP
import GPy
import numpy as np
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt

def train_multifidelity_GP(x_train_high,y_train_high,x_train_low,y_train_low):
    # x_train_high=para[0]
    # y_train_high=para[1]
    # x_train_low=para[2]
    # y_train_low=para[3]
    # if np.mod(i,10)==0:
        # print 'this is the {0} dimension'.format(i)
    input_dim=x_train_high.shape[1]
#    start_1=time.time()
    '''Train level 1'''
    k1=GPy.kern.Matern52(input_dim,ARD=False)
    scaler=preprocessing.StandardScaler().fit(y_train_low.reshape(-1,1))
    y_train_low_scaled=scaler.transform(y_train_low.reshape(-1,1))
    model1 = GPy.models.GPRegression(X=x_train_low, Y=y_train_low_scaled, kernel=k1)
    
    model1[".*Gaussian_noise"] = model1.Y.var()*0.1
    model1[".*Gaussian_noise"].fix()
    
    model1.optimize(max_iters = 1000)
    
    mu1,v1=model1.predict(x_train_high)
    # print 'time_1:',time.time()-start_1

#    start_2=time.time()
    '''Train level 2'''
    XX = np.hstack((x_train_high, mu1))
    
    # 新造的核还得再考虑，active_dimension
    k2 = GPy.kern.Matern52(1,active_dims=[input_dim])*GPy.kern.Matern52(input_dim,active_dims=np.arange(input_dim)) + GPy.kern.Matern52(input_dim,active_dims=np.arange(input_dim))
    y_train_high_scaled=scaler.transform(y_train_high.reshape(-1,1))
    model2 = GPy.models.GPRegression(X=XX, Y=y_train_high_scaled, kernel=k2)

    model2[".*Gaussian_noise"] = model2.Y.var()*0.01
    model2[".*Gaussian_noise"].fix()

    model2.optimize(max_iters = 1000)
    # print 'time_2:',time.time()-start_2
    return [model1,model2],scaler

def predict_multifidelity_GP(model_list,Nts,x_test_high,scaler):
    model1=model_list[0]
    model2=model_list[1]
    
#    start_3=time.time()
    '''Predict at test points'''
    # sample f_1 at xtest
    nsamples = 10
    mu_1, C_1 = model1.predict(x_test_high, full_cov=True)
    Z = np.random.multivariate_normal(mu_1.flatten(),C_1,nsamples)
    # print 'time_3:',time.time()-start_3
    
#    start_4=time.time()
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
    x_train_high=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_JL\X_H.txt')
    x_train_low=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_JL\X_L.txt')
    x_test=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_JL\X_test.txt')
    y_train_high=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_JL\Y_H.txt')
    y_train_low=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_JL\Y_L.txt')
    y_test=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_JL\Y_test.txt')
    
    [x_train_high,x_train_low,x_test,y_train_high,y_train_low]=map(lambda x:x.T,[x_train_high,x_train_low,x_test,y_train_high,y_train_low])
    Nobs=y_train_high.shape[1]
    Nts=x_test.shape[0]
    y_pred=np.zeros_like(y_test)
    var_pred=np.zeros_like(y_test)
    start=time.time()
    for i in range(Nobs):
#        if np.mod(i,50)==0:
        print('this is {0} dimension'.format(i))
        model_list,scaler=train_multifidelity_GP(x_train_high,y_train_high[:,i],x_train_low,y_train_low[:,i])
        y_pred[i,:],var_pred[i,:]=predict_multifidelity_GP(model_list,Nts,x_test,scaler)
    
    print('elapsed time:',time.time()-start)
    np.savetxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_JL\y_pred.txt',y_pred)
    np.savetxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_JL\var_pred.txt',var_pred)
    coef=np.corrcoef(y_pred.ravel(),y_test.ravel())
    print('corr coef:',coef[0,1])
    plt.scatter(y_pred.ravel(),y_test.ravel())
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    

