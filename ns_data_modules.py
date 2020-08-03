
import os
import  pandas as pd
import  numpy as np
import ns_sampling_modules as sm

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

def take_snapshot(self,c,loop_nb):
    # make the dataframe
    self.original_seq.to_pickle(
        path=make_file_name(c=c, file_description=make_sequence_filename(loop_nb=loop_nb+1)))
    self.times.to_pickle(
        path=make_file_name(c,file_description='times')
    )

    mutation_nb_df=pd.DataFrame()
    mutation_nb_df['nb mutations']=self.nb_mutations
    mutation_nb_df.to_pickle(path=make_file_name(c=c,file_description='nb_mutations'))


    min_yield_df=pd.DataFrame()
    min_yield_df['min_yield']=self.min_yield
    min_yield_df.to_pickle(path=make_file_name(c=c,file_description='min_yield'))

    pp = []
    for i in np.arange(loop_nb):
        pp.append(sum(self.percent_pos[i]) / len(self.percent_pos[i]))

    pp_df=pd.DataFrame()
    pp_df['percent pos average']=pp
    pp_df.to_pickle(path=make_file_name(c=c,file_description='percent_pos_average'))


def save_run_stats(c,steps_2_show,loops_2_show):
    # save initial run stats
    stats=pd.DataFrame()
    stats.loc[0,'Nb_loops']=c.nb_loops
    stats.loc[0,'Nb_Steps']=c.nb_steps
    stats.loc[0,'start mutation number']=c.nb_mutations
    stats.loc[0,'mutation_type']=c.mutation_type
    stats.loc[0,'Nb_sequences']=c.Nb_sequences
    stats['Steps to show']=sm.convert2pandas(np.array([steps_2_show]))
    stats['Loops to show']=sm.convert2pandas(np.array([loops_2_show]))
    stats.to_pickle(path=make_file_name(c=c,file_description='run_stats'))

