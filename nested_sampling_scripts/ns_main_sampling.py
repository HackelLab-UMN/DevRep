from nested_sampling_scripts import ns_nested_sampling as ns
import numpy as np
from contextlib import contextmanager
import os
import sys

# the input deck : specify these input parameters


'''THIS IS THE INPUT DECK :
    Number of loops --> number of monte carlo walks to do 
    Number of steps --> number of steps to take in a random walk, a single step can have multiple mutations
    Number of snapshots --> how often to save the data. 
    Mutation type -->as of right now two options , 'static' or 'dynamic'. 
    Number of mutations --> if static the a constant for number of mutations,
     if dynamic then the number of mutations to start with. 
'''


Number_loops=4
Number_steps=2
Number_snapshots=4



def driver(N_loops,N_steps,nb_snapshots=10,Nb_sequences=1000,suppress_output=False):
    trial1 = ns.ns_random_sample(Nb_sequences=Nb_sequences)
    step = N_loops // nb_snapshots
    loops_2_show=np.arange(0,N_loops+step,step)
    loops_2_show[-1]=loops_2_show[-1]-1
    if suppress_output is True:
        with suppress_stdout():
            trial1.nested_sample(N_loops=N_loops,N_steps=N_steps,loops_2_show=loops_2_show)
    else:
        times = trial1.nested_sample(N_loops=N_loops, N_steps=N_steps,loops_2_show=loops_2_show)
        print(times)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout



driver(N_loops=Number_loops, N_steps=Number_steps, nb_snapshots=Number_snapshots)



