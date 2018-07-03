# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from ES_MDA import ES_MDA
from ES_MDA_MF_GP import ES_MDA_MF_GP


Npar=5
N=10
ES_MDA_high_para_ave=np.zeros((Npar,N))
ES_MDA_high_dy=np.zeros(N)

ES_MDA_low_para_ave=np.zeros((Npar,N))
ES_MDA_low_dy=np.zeros(N)

MFES_MDA_para_ave=np.zeros((Npar,N))
MFES_MDA_dy=np.zeros(N)

Nod_num_high=402
Nod_num_low=102
sub_dir_high='high_fidelity'
sub_dir_low='low_fidelity'

# 高保真度ES-MDA
for i in range(N):
    print 'this is {0} ES_MDA_high'.format(i)
    ES_MDA_high_para_ave[:,i],ES_MDA_high_dy[i]=ES_MDA(Nod_num_high,sub_dir_high)


# 低保真度ES-MDA
for i in range(N):
    print 'this is {0} ES_MDA_low'.format(i)
    ES_MDA_low_para_ave[:,i],ES_MDA_low_dy[i]=ES_MDA(Nod_num_low,sub_dir_low)
    

# 多保真度ES-MDA
for i in range(N):
    print 'this is {0} MF_ES_MDA'.format(i)
    MFES_MDA_para_ave[:,i],MFES_MDA_dy[i]=ES_MDA_MF_GP()


