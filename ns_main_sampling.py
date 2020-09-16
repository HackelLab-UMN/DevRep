import ns_nested_sampling_ray as ns
import numpy as np
from contextlib import contextmanager
import os
import sys
from input_deck import inputs
import multiprocessing
import ns_data_modules as dm
# the input deck : specify these input parameters
import ns_show_results as sr

def driver(c,suppress_output=False):
    check_inputs(c=c)
    trial1 = ns.nested_sampling(Nb_sequences=c.Nb_sequences)
    step = c.nb_loops // c.nb_snapshots
    loops_2_show=np.arange(0,c.nb_loops+step,step)
    loops_2_show[-1]=c.nb_loops-1
    if suppress_output is True:
        with dm.suppress_stdout():
            times=trial1.nested_sample(c=c,loops_2_show=loops_2_show)
    else:
        times =  trial1.nested_sample(c=c,loops_2_show=loops_2_show)
        return times
    print('with %i cpus'%multiprocessing.cpu_count())
    print(times)


def check_inputs(c):
    if c.nb_loops<0 or c.nb_steps<0:
        raise AttributeError('Number of loops or number of steps is less than zero')
    if c.nb_snapshots > c.nb_loops:
        raise AttributeError('Number of snapshots is greater than Number of loops')
    if c.mutation_type != 'static' and c.mutation_type!='dynamic':
        raise AttributeError('invalid mutations type: %s'%c.mutation_type)
    if c.nb_mutations<0 or c.nb_mutations>16:
        raise AttributeError('Invalid # of mutations: %i'%c.nb_mutations)
    if c.Nb_sequences <0:
        raise AttributeError('Invalid # of sequences: %i'%c.Nb_sequences)

# c=inputs(nb_loops=100000,
#          nb_steps=5,
#          mutation_type='dynamic',
#          nb_mutations=10,
#          nb_snapshots=25,
#          Nb_sequences=10000,
#          yield2optimize='Developability',
#          nb_cores=8)


c=inputs(nb_loops=100000,
         nb_steps=5,
         mutation_type='dynamic',
         nb_mutations=10,
         nb_snapshots=25,
         Nb_sequences=10000,
         yield2optimize='IQ_Average_bc',
         nb_cores=8)

#todo:
# find a way to get the number of cores on a numa node
# stop output from cores so their is no core files in your directories.

if sys.platform=='darwin':
    suppress_output = False
else:
    suppress_output=True

driver(c=c,suppress_output=False)
# sr.main(C=[c])




