import matplotlib as mpl

mpl.use('Agg')
import pandas as pd
import os
import ns_sampling_modules as sm
import ns_plot_modules as pm

from input_deck import inputs

def violin_loop_plots(c,strings_2_show=5):
    stats = pd.read_pickle(path=os.path.join(sm.make_directory(Nb_steps=c.nb_steps,
                                                               Nb_loops=c.nb_loops,
                                                               nb_sequences=c.Nb_sequences,
                                                               mutation_type=c.mutation_type,
                                                               nb_mutations=c.nb_mutations
                                                               ), # end direcotry
                                             'run_stats.pkl'))

    loops_2_show = sm.convert2numpy(df=stats, field='Loops to show')[0] + 1

    #TODO: solve the string to show problem , once you solve it then you can call this
    # script on the pbs script
    #l2s=loops_2_show.shape[0]
    #if l2s>strings_2_show:
    pm.violin_saved_dataset(nb_steps=c.nb_steps,nb_loops=c.nb_loops,loops_2_show=loops_2_show)


#loops_2_show=np.hstack((loops_2_show[0:2].copy(),loops_2_show[3:8:2].copy()))

# TODO: make sure your just passing the parameter c everywhere which is the input class
#   also make sure the inputs will be compatible for a smart sample,,,
#   so for repeated calls to nested_sample, you can still mess with the input deck.
c = inputs()
# violin_loop_plots(c=c)
dir_name=sm.make_directory(Nb_steps=c.nb_steps,Nb_loops=c.nb_loops,nb_sequences=c.Nb_sequences,
                  mutation_type=c.mutation_type,nb_mutations=c.nb_mutations)


# also include that on their too.... like I should just have one function call and then it
# gets my data not this craziness.
# im passing around things like an idiot is is so confusing...
src='./sampling_data/'+dir_name+'/nb_mutations.pkl'
df=pd.read_pickle(path=src)
