# -*- coding: utf-8 -*-

##每个维度独立构造GP
import GPy
import numpy as np
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError

def train_multifidelity_GP(x_train_high,y_train_high,x_train_low,y_train_low,i):
    try:
        input_dim=x_train_high.shape[1]
        '''Train level 1'''
        k1=GPy.kern.Matern52(input_dim,ARD=False)
        scaler=preprocessing.StandardScaler().fit(y_train_low.reshape(-1,1))
        y_train_low_scaled=scaler.transform(y_train_low.reshape(-1,1))
        model1 = GPy.models.GPRegression(X=x_train_low, Y=y_train_low_scaled, kernel=k1)
        
        model1[".*Gaussian_noise"] = model1.Y.var()*0.1
        model1[".*Gaussian_noise"].fix()
        
        model1.optimize(max_iters = 10000)
        model1[".*Gaussian_noise"].unfix()
        model1[".*Gaussian_noise"].constrain_positive()
        
        model1.optimize_restarts(10, optimizer = "bfgs",  max_iters = 10000,verbose=False)
        
        mu1,v1=model1.predict(x_train_high)
        
        '''Train level 2'''
        XX = np.hstack((x_train_high, mu1))
        

        k2 = GPy.kern.Matern52(1,active_dims=[input_dim])*GPy.kern.Matern52(input_dim,active_dims=np.arange(input_dim)) + GPy.kern.Matern52(input_dim,active_dims=np.arange(input_dim))
        y_train_high_scaled=scaler.transform(y_train_high.reshape(-1,1))
        model2 = GPy.models.GPRegression(X=XX, Y=y_train_high_scaled, kernel=k2)
        
        model2[".*Gaussian_noise"] = model2.Y.var()*0.01
        model2[".*Gaussian_noise"].fix()
        
        model2.optimize(max_iters = 10000)
        model2[".*Gaussian_noise"].unfix()
        model2[".*Gaussian_noise"].constrain_positive()
        
        model2.optimize_restarts(10, optimizer = "bfgs",  max_iters = 10000,verbose=False)
    except LinAlgError:
        return [model1,model2],scaler
    return [model1,model2],scaler

def predict_multifidelity_GP(model_list,Nts,x_test_high,scaler):
    model1=model_list[0]
    model2=model_list[1]
    
    '''Predict at test points'''
    nsamples = 10
    mu_1, C_1 = model1.predict(x_test_high, full_cov=True)
    Z = np.random.multivariate_normal(mu_1.flatten(),C_1,nsamples)

    tmp_m = np.zeros((nsamples,Nts))
    tmp_v = np.zeros((nsamples,Nts))
    for j in range(0,nsamples):
        mu, v = model2.predict(np.hstack((x_test_high, Z[j,:][:,None]))) #predict得到的是Nts*1的结果
        tmp_m[j,:] = mu.flatten()
        tmp_v[j,:] = v.flatten()

    mean = np.mean(tmp_m, axis = 0)[:,None]
    var = np.mean(tmp_v, axis = 0)[:,None]+ np.var(tmp_m, axis = 0)[:,None]
    var = np.abs(var)
    var=var.ravel()
    
    mean_rescaled=scaler.inverse_transform(mean.ravel())
    
#    error = np.linalg.norm(y_original_high[:,i] - mean_rescaled)/np.linalg.norm(y_original_high[:,i])
    return mean_rescaled,var



if __name__=='__main__':
    x_train_high=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_zjj\ES_MDA_MF_GP_with_5_para\X_H.txt')
    x_train_low=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_zjj\ES_MDA_MF_GP_with_5_para\X_L.txt')
    x_test=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_zjj\ES_MDA_MF_GP_with_5_para\X_test.txt')
    y_train_high=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_zjj\ES_MDA_MF_GP_with_5_para\Y_H.txt')
    y_train_low=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_zjj\ES_MDA_MF_GP_with_5_para\Y_L.txt')
#    y_test=np.loadtxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_zjj\ES_MDA_MF_GP_with_5_para\Y_test.txt')
    
    [x_train_high,x_train_low,x_test,y_train_high,y_train_low]=map(lambda x:x.T,[x_train_high,x_train_low,x_test,y_train_high,y_train_low])
    Nobs=y_train_high.shape[1]
    Nts=x_test.shape[0]
    y_pred=np.zeros((Nobs,Nts))
    var_pred=np.zeros_like(y_pred)
    start=time.time()
    

    for i in range(Nobs):
#        if np.mod(i,50)==0:
#        print('this is {0} dimension'.format(i))
        model_list,scaler=train_multifidelity_GP(x_train_high,y_train_high[:,i],x_train_low,y_train_low[:,i],i)
        y_pred[i,:],var_pred[i,:]=predict_multifidelity_GP(model_list,Nts,x_test,scaler)
#        iter1+=1

    with open(r'F:\zq\multi_fidelity_GP\Hydrus_case_zjj\ES_MDA_MF_GP_with_5_para\indicator.txt','w') as f:
        if np.isnan(y_pred).sum()==0:
            f.write('0')
        else:
            f.write('1')
                
#    print('elapsed time:',time.time()-start)
    np.savetxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_zjj\ES_MDA_MF_GP_with_5_para\y_pred.txt',y_pred)
    np.savetxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_zjj\ES_MDA_MF_GP_with_5_para\var_pred.txt',var_pred)
#    coef=np.corrcoef(y_pred.ravel(),y_test.ravel())
#    print('corr coef:',coef[0,1])
#    plt.xlabel('predicted value')
#    plt.ylabel('true value')
#    plt.plot([1.1*np.min(y_pred),1.1*np.max(y_pred)],[1.1*np.min(y_test),1.1*np.max(y_test)],'r',lw=2)
#    plt.scatter(y_pred.ravel(),y_test.ravel())
#    plt.show()
    
    
    
    
    
   