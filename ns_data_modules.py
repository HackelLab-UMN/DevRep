
import os
import  pandas as pd
import  numpy as np
import ns_sampling_modules as sm
from input_deck import names


def make_file_name(c,file_description,fileformat='pkl'):
    dir_name=make_directory(c=c)
    return './sampling_data/' + dir_name+'/'+ file_description +'.'+fileformat

def zip_data(c):
    dir_name=make_directory(c=c)
    lst=os.listdir(path='./sampling_data/'+dir_name)
    cmd='zip ./sampling_data/'+dir_name+'/'+dir_name+'.zip'
    for i in lst:
        cmd=cmd+' ./sampling_data/'+dir_name+'/'+i
    os.system(cmd)

def make_directory(c):
    return 'Nb_sequences_%i_Nbsteps_%i_Nb_loops_%i_%s_%i'%(c.Nb_sequences,c.nb_steps,c.nb_loops,c.mutation_type,c.nb_mutations)

def make_sequence_filename(loop_nb):
    return 'sequences_loop_%i'%loop_nb

def take_snapshot(self,c,loop_nb,times):
    # make the dataframe
    #todo: init a class that allows for different saving of data depending on run.
    n=names()
    self.original_seq.to_pickle(path=make_file_name(c=c, file_description=make_sequence_filename(loop_nb=loop_nb+1)))
    times.to_pickle(path=make_file_name(c,file_description=n.times_fn))
    to_pickle(c=c,data=self.nb_mutations,data_name=n.nb_mutation_fn)
    to_pickle(c=c, data=self.min_yield,data_name=n.min_yield_fn)
   # self.rng_times.to_pickle(path=make_file_name(c=c,file_description=n.random_fn))
    to_pickle(c=c,data=self.percent_pos,data_name=n.pp_fn)


def to_pickle(c,data,data_name):
    df = pd.DataFrame()
    df[data_name] = data
    df.to_pickle(path=make_file_name(c=c, file_description=data_name))


def save_run_stats(c,loops_2_show):
    # save initial run stats
    stats=pd.DataFrame()
    stats.loc[0,'nb_loops']=c.nb_loops
    stats.loc[0,'nb_steps']=c.nb_steps
    stats.loc[0,'start mutation number']=c.nb_mutations
    stats.loc[0,'mutation_type']=c.mutation_type
    stats.loc[0,'Nb_sequences']=c.Nb_sequences
    stats['Loops to show']=sm.convert2pandas(np.array([loops_2_show]))
    stats.to_pickle(path=make_file_name(c=c,file_description='run_stats'))

def read_pickle(c,file_description):
    return pd.read_pickle(make_file_name(c=c, file_description=file_description))

