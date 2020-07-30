import ns_sampling_modules as sm
import os
from ns_password import PASSWORD,MSI_DIRECTORY,LOCAL_DIRECTORY

OUT_SCRIPT= os.path.dirname(MSI_DIRECTORY)+'/ns_nested_sampling_CPU.pbs.o'

password='sshpass -p '+PASSWORD

def push_scripts(scripts=None):
    if scripts is None:
        scripts=['ns_nested_sampling.py','ns_sampling_modules.py','ns_plot_modules.py','ns_nested_sampling_CPU.pbs']
    for i in scripts:
        os_out=os.system(password + ' scp ' + LOCAL_DIRECTORY + '/' + i +
                         ' ' + MSI_DIRECTORY)
        if os_out is 0:
            print('sucessfully transferred file: ' + i + ' to ' + MSI_DIRECTORY)
        else:
            raise SystemError('error transferring file : ' + i + ' to ' + MSI_DIRECTORY)

def pull_zipped_data(nb_steps,nb_loops,nb_sequences=1000):
    dir_name=sm.make_directory(Nb_steps=nb_steps, Nb_loops=nb_loops,nb_sequences=nb_sequences)
    src= MSI_DIRECTORY + '/sampling_data/' + dir_name + '/' + dir_name + '.zip'
    target= LOCAL_DIRECTORY + '/sampling_data/' + dir_name
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
    os_out=os.system(password +' scp ' + OUT_SCRIPT + job_nb +' ' + LOCAL_DIRECTORY + '/sampling_data')
    if os_out == 0:
        print('sucessfully transferred file: ' + OUT_SCRIPT + job_nb +' to ' + LOCAL_DIRECTORY + '/sampling_data')
    else:
        raise SystemError('error transferring file : ' + OUT_SCRIPT + job_nb + ' to ' + LOCAL_DIRECTORY + '/sampling_data')

def unzip_data(dir_name):
    fp='./sampling_data/' + dir_name
    cmd = 'unzip -o '+ fp +'/'+dir_name+'.zip '
    os.system(cmd)

def zip_data(dir_name):
    lst=os.listdir(path='./sampling_data/'+dir_name)
    cmd='zip ./sampling_data/'+dir_name+'/'+dir_name+'.zip'
    for i in lst:
        cmd=cmd+' ./sampling_data/'+dir_name+'/'+i
    os.system(cmd)


#zip_data(dir_name=sm.make_directory(Nb_steps=5,Nb_loops=3))
# push_scripts(['main_DevRep_example.py'])


#push_scripts()