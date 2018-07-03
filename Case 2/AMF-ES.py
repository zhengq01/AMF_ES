# -*- coding: utf-8 -*-
def ES_MDA_MF_GP():
    import numpy as np
    import matplotlib.pyplot as plt
    import os,sys,time
    import GPy
    import multiprocessing
    from sklearn import preprocessing
    from utils.runexe import runexe
    from utils.write_para import write_Ki,write_n
    from utils.read_values import read_values
    from utils.forwardmodel import forwardmodel
    from utils.train_predict_multifidelity_GP_parallel import train_multifidelity_GP,predict_multifidelity_GP

    # parameter setting
    n1_mean=0.4
    n2_mean=0.43
    n3_mean=0.52
    n4_mean=0.6
    n5_mean=0.65
    n_std=0.03
    num_of_layer=5
    Ne=50
    N_H=5
    N_L=45
    Nod_num_high=402
    Nod_num_low=102
    filename_Ki='DrawDown_Ki.direct'
    filename_n='DrawDown_n.direct'
    filename_result='DrawDown_time_POINT1.tec'
    time_for_ogs_train=np.loadtxt('true_obs_time.txt')
    root_directory=os.getcwd()
    sub_dir_high='high_fidelity'
    sub_dir_low='low_fidelity'
    new_points_num_high=1
    new_points_num_low=2
    new_points_num=3

    para_mean=np.array([n1_mean,n2_mean,n3_mean,n4_mean,n5_mean])
    input_dim=para_mean.shape[0]
    Fi=np.diag([n_std]*input_dim)


    # observations
    y_obs=np.loadtxt('true_obs.txt')
    Nobs=y_obs.shape[0]
    sd=0.05*y_obs
    yobs=y_obs+sd*np.random.randn(Nobs)


    # GP prior
    X_H=np.random.randn(input_dim,N_H)
    X_H_for_ogs=para_mean.reshape(-1,1)+np.dot(Fi,X_H)
    X_H=X_H.T
    Y_H=forwardmodel(X_H_for_ogs,time_for_ogs_train,Nod_num_high,filename_n,filename_result,root_directory,sub_dir_high)
    Y_H=Y_H.T

    X_L=np.random.randn(input_dim,N_L)
    X_L_for_ogs=para_mean.reshape(-1,1)+np.dot(Fi,X_L)
    X_L=X_L.T
    Y_L=forwardmodel(X_L_for_ogs,time_for_ogs_train,Nod_num_low,filename_n,filename_result,root_directory,sub_dir_low)
    Y_L=Y_L.T

    
    mpr=np.random.randn(input_dim,Ne)
    # mpr=np.loadtxt('x_ini.txt')
    m1=mpr
    N_iter=15
    dy_ensem=[]
    surrogate_acc=[]
    start=time.time()
    time_ensem=[]
    spread=[]
    for i in xrange(N_iter+2):
        print 'This is {0} iteration'.format(i+1)
       
        y_pred=np.zeros((Nobs,Ne))
        var_pred=np.zeros(y_pred.shape)
        start=time.time()
        for i in xrange(Nobs):    
            model_list,scaler_for_test=train_multifidelity_GP(X_H,Y_H[:,i],X_L,Y_L[:,i])
            y_pred[i,:],var_pred[i,:]=predict_multifidelity_GP(model_list,Ne,m1.T,scaler_for_test)
        
        
        # # test the accuracy of surrogate
        # para_tmp=para_mean.reshape(-1,1)+np.dot(Fi,m1)
        # y_original=forwardmodel(para_tmp,time_for_ogs_train,Nod_num_high,filename_n,filename_result,root_directory,sub_dir_high)
        # cor_coef=np.corrcoef(y_pred.ravel(),y_original.ravel())
        # print 'surrogate acc',cor_coef
        # surrogate_acc.append(cor_coef[0,1])
        
        
        elapsed_time=time.time()-start
        time_ensem.append(elapsed_time)
        print 'elapsed_time:',elapsed_time

        m1_error=m1-np.tile(np.mean(m1,1).reshape(-1,1),(1,Ne))
        y_error=y_pred-np.tile(np.mean(y_pred,1).reshape(-1,1),(1,Ne))

        Cmy=np.dot(m1_error,y_error.T)/(Ne-1)
        Cyy=np.dot(y_error,y_error.T)/(Ne-1)
        sd_=sd*np.sqrt(N_iter)
        Cd=np.diag(sd_**2)
        kgain=np.dot(Cmy,np.linalg.inv(Cyy+Cd))
        m2=np.zeros(m1.shape)
        randn_num=np.random.randn(Nobs,Ne)
        for i in range(Ne):
            obse=yobs+sd_*np.random.randn(Nobs)
            m2[:,i]=m1[:,i]+np.dot(kgain,obse-y_pred[:,i])  
        
        m1=m2
       
         # add posterior points
        var_pred_mean=np.mean(var_pred,0)
        var_sort_index=list(np.argsort(var_pred_mean))
        var_sort_index.reverse() 
        idx_sel=var_sort_index[:new_points_num]
        para_add_high=m2[:,idx_sel[:new_points_num_high]]
        para_add_low=m2[:,idx_sel[new_points_num_high:]]
        
        X_H=np.hstack((X_H.T,para_add_high))
        X_H=X_H.T
        X_L=np.hstack((X_L.T,para_add_low))
        X_L=X_L.T
        
        # high-fidelity points
        para_add_high_for_ogs=para_mean.reshape(-1,1)+np.dot(Fi,para_add_high)
        ry_high=forwardmodel(para_add_high_for_ogs,time_for_ogs_train,Nod_num_high,filename_n,filename_result,root_directory,sub_dir_high)
        Y_H=np.hstack((Y_H.T,ry_high))
        Y_H=Y_H.T
        
        # low-fidelity points
        para_add_low_for_ogs=para_mean.reshape(-1,1)+np.dot(Fi,para_add_low)
        ry_low=forwardmodel(para_add_low_for_ogs,time_for_ogs_train,Nod_num_low,filename_n,filename_result,root_directory,sub_dir_low)
        Y_L=np.hstack((Y_L.T,ry_low))
        Y_L=Y_L.T
        
        
        dy=np.sqrt(np.mean((np.mean(y_pred,axis=1)-y_obs)**2))
        dy_ensem.append(dy)
        
     
        parY=para_mean.reshape(-1,1)+np.dot(Fi,m2)
        spread_=np.sqrt(np.mean(np.var(parY,1)))
        print 'spread:',spread_
        spread.append(spread_)
    
    parY=para_mean.reshape(-1,1)+np.dot(Fi,m2)  
    updated_para_ave=np.mean(parY,1)

    return updated_para_ave,dy

    
    # np.savetxt('dy_ensem_ES_MF_GP.txt',dy_ensem)      
    # elapsed_time=time.time()-start
    # print 'elapsed_time:',elapsed_time 
       
    # plt.figure(1)
    # plt.plot(dy_ensem)
    # plt.title('data mismatch')  
    # plt.savefig('ES_MDA_MF_GP_data_mismatch.jpeg')      

    # # 
    # parY=para_mean.reshape(-1,1)+np.dot(Fi,m2)  #得到（Nod_num_high,N_H）的parY
    # updated_para_ave=np.mean(parY,1)
    # np.savetxt('parY_ES_MF_GP.txt',parY)
    # np.savetxt('updated_para_ES_MF_GP.txt',updated_para_ave)

    # y_pred_ensem=forwardmodel(parY,time_for_ogs_train,Nod_num_high,filename_n,filename_result,root_directory,sub_dir_high)
        
    # i_test=1
    # # write_Ki(Nod_num,updated_para_ave[:num_of_ki],i_test,filename_Ki,root_directory)
    # write_n(Nod_num_high,updated_para_ave,i_test,filename_n,root_directory,sub_dir_high)
    # runexe(i_test,root_directory,sub_dir_high)
    # y_hat=read_values(time_for_ogs_train,i_test,filename_result,root_directory,sub_dir_high)
    # # yobs_test=np.loadtxt('true_obs_for_test.txt')
    # # yobs_all=np.array(list(y_obs)+list(yobs_test))
    # # error=np.sqrt(np.mean((y_hat[-yobs_test.shape[0]:]-yobs_test)**2))
    # error=np.sqrt(np.mean((y_hat-y_obs)**2))
    # print 'rmse_test:',error
    # print 'updated_para_ave:',updated_para_ave
    # # 
    # plt.figure(2)
    # for i in xrange(Ne):
        # if i==Ne-1:
            # plt.plot(time_for_ogs_train,y_pred_ensem[:,i],'g',alpha=0.2,label='predicted_sample')
        # if np.any(y_pred_ensem[:,i]>10):
            # continue
        # plt.plot(time_for_ogs_train,y_pred_ensem[:,i],'g',alpha=0.2)
    # plt.plot(time_for_ogs_train,y_hat,'r',label='predicted_mean')
    # plt.scatter(time_for_ogs_train,y_obs,c='b',label='true_obs')

    # plt.legend(loc='best')
    # plt.savefig('ES_MDA_MF_GP_post_samples.jpeg') 










