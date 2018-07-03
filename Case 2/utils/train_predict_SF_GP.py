# -*- coding: utf-8 -*-

import numpy as np
import GPy
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt


def train_predict_SF_GP(x_train,y_train,x_test):
    '''
    输入必须是样本在0维，特征在1维
    '''
    input_dim=x_train.shape[1]


#    y_pred=np.zeros((Nts,Nobs))
#    var_pred=np.zeros(y_pred.shape)

#    for i in range(Nobs):
#        if np.mod(i,100)==0:
#        print('this is {0} dimension'.format(i))
    k1=GPy.kern.Matern52(input_dim,ARD=False)
    scaler=preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
    y_train_scaled=scaler.transform(y_train.reshape(-1,1))
    
#    Z = np.random.rand(12,input_dim)*12
    model1 = GPy.models.SparseGPRegression(X=x_train,Y=y_train_scaled,num_inducing=200,kernel=k1)

#    model1=GPy.models.GPRegression(X=x_train,Y=y_train_scaled,kernel=k1)
    
    model1[".*Gaussian_noise"] = model1.Y.var()*0.1
    model1[".*Gaussian_noise"].fix()
    model1.optimize('bfgs',max_iters=1000)
    print model1.log_likelihood()
#    model1[".*Gaussian_noise"].unfix()
#    model1[".*Gaussian_noise"].constrain_positive()
#    
#    model1.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000)
    
    mu1,v1=model1.predict(x_test)
    
    y_pred=scaler.inverse_transform(mu1.ravel())
    var_pred=v1.ravel()
    return [y_pred,var_pred]


    
if __name__=='__main__':
    x_train=np.loadtxt('X_H.txt')
    x_test=np.loadtxt('X_test.txt')
    y_train=np.loadtxt('Y_H.txt')
    y_test=np.loadtxt('Y_test.txt')
    
    [x_train,x_test,y_train,y_test]=map(lambda x:x.T,[x_train,x_test,y_train,y_test])
    start=time.time()
#    y_pred,var_pred=train_predict_SF_GP(x_train,y_train,x_test)

 
#    print('elapsed time:',time.time()-start)
#    [y_pred,var_pred]=map(lambda x:x.T,[y_pred,var_pred])
#    np.savetxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_JL\y_pred.txt',y_pred)
#    np.savetxt(r'F:\zq\multi_fidelity_GP\Hydrus_case_JL\var_pred.txt',var_pred)
#    coef=np.corrcoef(y_pred.ravel(),y_test.ravel())
#    print('corr coef:',coef[0,1])
#    plt.scatter(y_pred.ravel(),y_test.ravel())
#    plt.show()

    import multiprocessing
#    pool=multiprocessing.Pool(10)
#    result=[]
#    for i in xrange(y_train.shape[1]):
#        print i
#        result.append(pool.apply_async(train_predict_SF_GP,(x_train,y_train[:,i],x_test)))
#    pool.close()
#    pool.join()
    
    Nobs=y_train.shape[1]

    Nts=x_test.shape[0]
    y_pred=np.zeros((Nts,2))
    var_pred=np.zeros(y_pred.shape)
#    for i in xrange(y_train.shape[1]):
#        tmp=result[i].get()
#        y_pred[:,i]=tmp[0]
#        var_pred[:,i]=tmp[1]
        
    for i in xrange(2):
        print i
        y_pred[:,i],var_pred[:,i]=train_predict_SF_GP(x_train,y_train[:,i],x_test)   
        
    print('elapsed time:',time.time()-start)   
    coef=np.corrcoef(y_pred.ravel(),y_test[:,:2].ravel())
    print('corr coef:',coef[0,1])
    plt.scatter(y_pred.ravel(),y_test[:,:2].ravel())
    plt.show()   
        