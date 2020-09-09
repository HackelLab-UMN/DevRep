import ns_sampling_modules as sm
import os
from ns_password import PASSWORD,MSI_DIRECTORY,LOCAL_DIRECTORY
import ns_data_modules as dm
from input_deck import inputs
password='sshpass -p '+PASSWORD

def push_scripts(scripts=None):
    if scripts is None:
        scripts=['ns_nested_sampling.py','ns_sampling_modules.py','ns_plot_modules.py','ns_nested_sampling_CPU.pbs','ns_main_sampling.py',
                 'submodels_module.py']
    for i in scripts:
        os_out=os.system(password + ' scp ' + LOCAL_DIRECTORY + '/' + i +
                         ' ' + MSI_DIRECTORY)
        if os_out is 0:
            print('sucessfully transferred file: ' + i + ' to ' + MSI_DIRECTORY)
        else:
            raise SystemError('error transferring file : ' + i + ' to ' + MSI_DIRECTORY)

def pull_zipped_data(c):
    dir_name=dm.make_directory(c)
    src= MSI_DIRECTORY + '/sampling_data/' + dir_name + '/' + dir_name + '.zip'
    target= LOCAL_DIRECTORY + '/sampling_data/' + dir_name
    os.system('mkdir '+target)
    cmd= password + ' scp '+src+' '+target

    # TODO: make a scp function that takes a src file and a target directory
    os_out=os.system(cmd)
    if os_out== 0:
        print('sucessfully transferred file: ' + dir_name+'.zip' + ' to ' + target )
        os_out=unzip_data(dir_name=dir_name)

def pull_output_script(job_nb):
    job_nb=str(job_nb)
    #print(password + ' scp ' + OUT_SCRIPT + job_nb + ' ' + ML_DEVELOPABILITY + '/sampling_data')
    os_out=os.system(password +' scp ' + MSI_DIRECTORY +'/ns_nested_sampling_CPU.pbs.o'+ job_nb +' ' + LOCAL_DIRECTORY + '/sampling_data')
    if os_out == 0:
        print('sucessfully transferred file: ' +  MSI_DIRECTORY +'/ns_nested_sampling_CPU.pbs.o'+ job_nb +' to ' + LOCAL_DIRECTORY + '/sampling_data')
    else:
        raise SystemError('error transferring file : ' +  MSI_DIRECTORY +'/ns_nested_sampling_CPU.pbs.o'+ job_nb + job_nb + ' to ' + LOCAL_DIRECTORY + '/sampling_data')

def unzip_data(dir_name):
    fp='./sampling_data/' + dir_name
    cmd = 'unzip -o '+ fp +'/'+dir_name+'.zip '
    os_out=os.system(cmd)
    if os_out == 0:
        print('sucessfully unzipped data for ' + dir_name)



#push_scripts(['ns_main_sampling.py']
# from ns_latest_runs import C
#
#
# for c in C:
#     pull_zipped_data(c=c)

#TODO: make some file that says what files need to be pulled from msi, but that would be advanced.
#zip_data(dir_name=sm.make_directory(Nb_steps=5,Nb_loops=3))
# push_scripts(['main_DevRep_example.py'])

push_scripts(['ns_main_sampling.py'])
# note you should only push
#push_scripts(['input_deck.py'])
#pull_output_script(job_nb=22177757)
#pull_zipped_data(nb_steps=5,nb_loops=10000,nb_mutations=6,mutation_type='static')
#push_scripts(['submodels_module.py','ns_main_sampling.py'])

#pull_output_script(job_nb=21795057