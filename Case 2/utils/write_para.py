# -*- coding: utf-8 -*-

import numpy as np
import os


# 传入5个ki组成的array
def write_Ki(Nod_num,para,i,filename_Ki,root_directory,sub_dir):
    para=np.squeeze(para)
    if para[0]<0:  #判断para是否是log之后的值，如果是则为负，则用exp换回去
        para=np.exp(para)
    value_list={}   #产生要添加的ki序列
    gas_file='DrawDown_{0}'.format(i)
#    gas_file='DrawDown_00'
    args_ki=os.path.join(root_directory,sub_dir,gas_file,filename_Ki)
#    for i in xrange(Nod_num):
#        value_list[i]=para[0]
    num_of_layer=para.shape[0]
    layer_points=Nod_num/num_of_layer
    for i in xrange(num_of_layer):
        for k in xrange(i*layer_points,(i+1)*layer_points):
            value_list[k]=para[i]   
            
    value_list_to_str=[]    #将序列转换成可以写入文件的字符串格式
    for i,j in value_list.iteritems():
        var=' '.join([str(i),str(j),'\n'])  #使用join进行字符串的拼接效率更高
        value_list_to_str.append(var)
    with open(args_ki,'w') as f:
        for line in value_list_to_str:
            f.write(line)
            pass
        f.write('#STOP')
    
def write_n(Nod_num,para,i,filename_n,root_directory,sub_dir):
    para=np.squeeze(para)
    value_list={}   #产生要添加的ki序列
    gas_file='DrawDown_{0}'.format(i)
#    gas_file='DrawDown_00'
    args_ki=os.path.join(root_directory,sub_dir,gas_file,filename_n)
#    for i in xrange(Nod_num):
#        value_list[i]=para[0]
    num_of_layer=para.shape[0]
    layer_points=Nod_num/num_of_layer
    for i in xrange(num_of_layer):
        for k in xrange(i*layer_points,(i+1)*layer_points):
            value_list[k]=para[i]     
    
    if len(value_list) != Nod_num:  #Nod_num不一定是层数的整数倍，这里最后将个数补齐为Nod_num个
        for k in xrange(len(value_list),Nod_num):
            value_list[k]=para[-1]

    
    value_list_to_str=[]    #将序列转换成可以写入文件的字符串格式
    for i,j in value_list.iteritems():
        var=' '.join([str(i),str(j),'\n'])  #使用join进行字符串的拼接效率更高
        value_list_to_str.append(var)
    with open(args_ki,'w') as f:
        for line in value_list_to_str:
            f.write(line)
            pass
        f.write('#STOP')
        
        
if __name__=='__main__':
    k1_mean=1.0e-11
    k2_mean=1.0e-11
    k3_mean=1.0e-11
    k4_mean=1.0e-11
    k5_mean=1.0e-11
    n1_mean=0.49863212
    n2_mean=0.4610569
    n3_mean=0.46095229
    n4_mean=0.72707135
    n5_mean=0.81496852
    k_std=0.6
    n_std=0.03
    num_of_layer=5
    Ne=100
    Nod_num=402
    filename_Ki='DrawDown_Ki.direct'
    filename_n='DrawDown_n.direct'
    root_directory=os.getcwd()
    root_directory=os.path.dirname(root_directory)
    para_mean=np.array([np.log(k1_mean),np.log(k2_mean),np.log(k3_mean),np.log(k4_mean),np.log(k5_mean),n1_mean,n2_mean,n3_mean,n4_mean,n5_mean])
    i=1
    sub_dir='high_fidelity'
#    write_Ki(Nod_num,para_mean[:5],i,filename_Ki,root_directory,sub_dir)
    write_n(Nod_num,para_mean[5:],1,filename_n,root_directory,sub_dir)
    
    
    

    
    