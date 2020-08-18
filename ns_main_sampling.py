import ns_nested_sampling as ns
import numpy as np
from contextlib import contextmanager
import os
import sys
from input_deck import inputs
# the input deck : specify these input parameters
import ns_show_results as sr

def driver(c,suppress_output=False):
    check_inputs(c=c)
    trial1 = ns.ns_random_sample(Nb_sequences=c.Nb_sequences)
    step = c.nb_loops // c.nb_snapshots
    loops_2_show=np.arange(0,c.nb_loops+step,step)
    loops_2_show[-1]=loops_2_show[-1]-1
    if suppress_output is True:
        with suppress_stdout():
            # TODO :  is to remove this and just have the class input, it would be so much cleaner
            #   that way
            trial1.nested_sample(c=c,loops_2_show=loops_2_show)
    else:
        times =  trial1.nested_sample(c=c,loops_2_show=loops_2_show)
        print(times)
        return times
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def check_inputs(c):
    if c.nb_loops<0 or c.nb_steps<0:
        raise AttributeError('Number of loops or number of steps is less than zero')
    if c.nb_snapshots > c.nb_loops:
        raise AttributeError('Number of snapshots is greater than Number of loops')
    if c.mutation_type != 'static' and c.mutation_type!='dynamic':
        raise AttributeError('invalid mutations type: %s'%c.mutation_type)
    if c.nb_mutations<0:
        raise AttributeError('Invalid # of mutations: %i'%c.nb_mutations)
    if c.Nb_sequences <0:
        raise AttributeError('Invalid # of sequences: %i'%c.Nb_sequences)


c=inputs(nb_loops=100000,
         nb_steps=5,
         mutation_type='static',
         nb_mutations=1,
         nb_snapshots=2,
         Nb_sequences=1000)
driver(c=c)
sr.main(C=[c])




