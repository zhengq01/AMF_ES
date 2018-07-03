# -*- coding: utf-8 -*-
import os
import subprocess

def runexe(i,cur_directory,sub_dir):
    path_1=cur_directory
    path_2=r'DrawDown_{0}/ogs'.format(i)
    path_3=r'DrawDown_{0}/DrawDown'.format(i)
    args_exe=os.path.join(path_1,sub_dir,path_2)
    para=os.path.join(path_1,sub_dir,path_3)
    pp=subprocess.Popen((args_exe,para),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    pp.communicate()
    pp.wait()


    
if __name__=='__main__':
    # import multiprocessing
    # pool=multiprocessing.Pool(20)
    # Ne=100
    cur_directory=os.getcwd()
    root_directory=os.path.dirname(cur_directory)
    # for i in xrange(Ne):
        # pool.apply_async(runexe,(i,root_directory))
    # pool.close()
    # pool.join()
    # print 'over'
    runexe(3,root_directory)
    