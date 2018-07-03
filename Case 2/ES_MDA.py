# -*- coding: utf-8 -*-

def ES_MDA(Nod_num,sub_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    import os,sys,time
    import multiprocessing
    from utils.runexe import runexe
    from utils.write_para import write_Ki,write_n
    from utils.forwardmodel import forwardmodel
    from utils.read_values import read_values


    # parameter setting
    n1_mean=0.4
    n2_mean=0.43
    n3_mean=0.52
    n4_mean=0.6
    n5_mean=0.65
    n_std=0.03
    num_of_layer=5
    Ne=50
    Nod_num=Nod_num
    filename_Ki='DrawDown_Ki.direct'
    filename_n='DrawDown_n.direct'
    filename_result='DrawDown_time_POINT1.tec'
    time_for_ogs_train=np.loadtxt('true_obs_time.txt')
    root_directory=os.getcwd()
    # sub_dir='high_fidelity'
    sub_dir=sub_dir


    para_mean=np.array([n1_mean,n2_mean,n3_mean,n4_mean,n5_mean])
    input_dim=para_mean.shape[0]
    Fi=np.diag([n_std]*input_dim)


    # observations
    y_obs=np.loadtxt('true_obs.txt')
    Nobs=y_obs.shape[0]
    sd=0.05*y_obs
    yobs=y_obs+sd*np.random.randn(Nobs)



    mpr=np.random.randn(input_dim,Ne)
    m1=mpr
    N_iter=15
    dy_ensem=[]
    spread=[]
    start=time.time()
    for i in xrange(N_iter+2):
        print 'This is {0} iteration'.format(i+1)
        
        para_for_ogs=para_mean.reshape(-1,1)+np.dot(Fi,m1)   
        y_pred=forwardmodel(para_for_ogs,time_for_ogs_train,Nod_num,filename_n,filename_result,root_directory,sub_dir)
        

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
       

        dy=np.sqrt(np.mean((np.mean(y_pred,axis=1)-y_obs)**2))
        dy_ensem.append(dy)
        

        parY=para_mean.reshape(-1,1)+np.dot(Fi,m2)
        spread_=np.sqrt(np.mean(np.var(parY,1)))
        print 'spread:',spread_
        spread.append(spread_)

        
        
    parY=para_mean.reshape(-1,1)+np.dot(Fi,m2) 
    updated_para_ave=np.mean(parY,1)    
        
    return updated_para_ave,dy
        

    # np.savetxt('dy_ensem_ES.txt',dy_ensem)
        
    # elapsed_time=time.time()-start
    # print 'elapsed_time:',elapsed_time      
        
    # plt.figure(1)
    # plt.plot(dy_ensem)
    # plt.title('data mismatch')  
    # plt.savefig('ES_MDA_low_data_mismatch.jpeg')    

    # #
    # parY=para_mean.reshape(-1,1)+np.dot(Fi,m2)  #得到（Nod_num_high,N_H）的parY
    # updated_para_ave=np.mean(parY,1)
    # np.savetxt('parY_ES.txt',parY)
    # np.savetxt('updated_para_ES.txt',updated_para_ave)

    # y_pred_ensem=forwardmodel(parY,time_for_ogs_train,Nod_num,filename_n,filename_result,root_directory,sub_dir)

        
    # i_test=1
    # #write_Ki(Nod_num,updated_para_ave[:num_of_ki],i_test,filename_Ki,root_directory)
    # write_n(Nod_num,updated_para_ave,i_test,filename_n,root_directory,sub_dir)
    # runexe(i_test,root_directory,sub_dir)
    # y_hat=read_values(time_for_ogs_train,i_test,filename_result,root_directory,sub_dir)
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
    # plt.savefig('ES_MDA_low_post_samples.jpeg')   

    # plt.figure(3)
    # plt.plot(spread)
    # plt.ylabel('Spread')
    # plt.savefig('ES_MDA_low_para_spread.jpeg')   
    # plt.show()









