

import ns_sampling_modules as sm
import ns_plot_modules as pm
import os
import subprocess
import sys
from ns_password import PASSWORD
ML_DEVELOPABILITY = '/Users/bryce.johnson/Desktop/ML/Developability'
OUT_SCRIPT= 'joh14192@login.msi.umn.edu:/panfs/roc/groups/7/mart5523/joh14192/ns_nested_sampling_CPU.pbs.o'
JOH_DEVELOPABILITY = 'joh14192@login.msi.umn.edu:/home/mart5523/joh14192/Developability'
password='sshpass -p '+PASSWORD

def pull_data(nb_steps,nb_loops,nb_sequences=1000):
    dir_name=sm.make_directory(Nb_loops=nb_loops,Nb_steps=nb_steps,nb_sequences=nb_sequences)
    print(password + ' scp -r ' + JOH_DEVELOPABILITY + '/sampling_data/' +
    dir_name + ' ' +
    ML_DEVELOPABILITY + '/sampling_data')
    os_out=os.system(password + ' scp -r ' + JOH_DEVELOPABILITY + '/sampling_data/' +
                     dir_name +' '+
                     ML_DEVELOPABILITY)
    if os_out is 0:
        print('successful transmission of directory: ' + dir_name +' to directory ' +
              ML_DEVELOPABILITY + '/sampling_data')
    else:
        raise SystemError('unsuccessful transmission of directory: ' + dir_name +' to directory' +
                          ML_DEVELOPABILITY)

def push_scripts(scripts=None):
    if scripts is None:
        scripts=['ns_nested_sampling.py','ns_sampling_modules.py','ns_plot_modules.py','ns_nested_sampling_CPU.pbs']
    for i in scripts:
        os_out=os.system(password + ' scp ' + ML_DEVELOPABILITY+'/' + i +
                         ' ' + JOH_DEVELOPABILITY)
        if os_out is 0:
            print('sucessfully transferred file: ' + i + ' to ' + JOH_DEVELOPABILITY)
        else:
            raise SystemError('error transferring file : ' + i + ' to ' + JOH_DEVELOPABILITY)

def pull_zipped_data(nb_steps,nb_loops,nb_sequences=1000):
    dir_name=sm.make_directory(Nb_steps=nb_steps, Nb_loops=nb_loops,nb_sequences=nb_sequences)
    src=JOH_DEVELOPABILITY + '/sampling_data/' +dir_name+'/'+dir_name+'.zip'
    target=ML_DEVELOPABILITY+ '/sampling_data/'+dir_name
    os.system('mkdir '+target)
    cmd= password + ' scp -r '+src+' '+target

    # TODO: finish pulling zipped data
    # TODO: make a scp function that takes a src file and a target directory
    os_out=os.system(cmd)
    if os_out== 0:
        print('sucessfully transferred file: ' + src + ' to ' + target )
        os_out=unzip_data(dir_name=dir_name)
        if os_out==0:
            print('sucessfully unzipped data for '+dir_name)
def pull_output_script(job_nb):
    job_nb=str(job_nb)
    #print(password + ' scp ' + OUT_SCRIPT + job_nb + ' ' + ML_DEVELOPABILITY + '/sampling_data')
    os_out=os.system(password+' scp '+OUT_SCRIPT+job_nb+' '+ML_DEVELOPABILITY+ '/sampling_data')
    if os_out == 0:
        print('sucessfully transferred file: ' + OUT_SCRIPT +job_nb +' to ' + JOH_DEVELOPABILITY+'/sampling_data')
    else:
        raise SystemError('error transferring file : ' + OUT_SCRIPT+job_nb + ' to ' + JOH_DEVELOPABILITY+'/sampling_data')

def unzip_data(dir_name):
    fp='/sampling_data/' + dir_name
    cmd = 'unzip -o .'+ fp +'/'+dir_name+'.zip '
    os.system(cmd)

def zip_data(dir_name):
    lst=os.listdir(path='./sampling_data/'+dir_name)
    cmd='zip ./sampling_data/'+dir_name+'/'+dir_name+'.zip'
    for i in lst:
        cmd=cmd+' ./sampling_data/'+dir_name+'/'+i
    os.system(cmd)


#zip_data(dir_name=sm.make_directory(Nb_steps=5,Nb_loops=3))


unzip_data(dir_name=sm.make_directory(Nb_steps=4,Nb_loops=15000))



