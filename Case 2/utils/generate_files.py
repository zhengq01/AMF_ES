# -*- coding: utf-8 -*-

def remove_generate_files(N,sub_dir):
    import os
    import shutil
    current_directory=os.getcwd()
    root_directory=os.path.dirname(current_directory)
    for i in xrange(N):
        dir_name='DrawDown_{0}'.format(i)
        file_args=os.path.join(root_directory,sub_dir,dir_name)
        old=os.path.join(root_directory,'DrawDown_00')
        if os.path.exists(file_args):
            shutil.rmtree(file_args)
            shutil.copytree(old,file_args)
        else:
            shutil.copytree(old,file_args)

if __name__=='__main__':
    sub_dir='DrawDown_files'
    remove_generate_files(101)
    
    
    
