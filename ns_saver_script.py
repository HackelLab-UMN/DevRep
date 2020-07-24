import time
import submodels_module as mb
import load_format_data
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import ns_sampling_modules as sm
import ns_plot_modules as pm
# compiled optimizer
import matplotlib as mpl
from collections import namedtuple
mpl.use('Agg')
from contextlib import contextmanager


df=pd.read_pickle(pm.make_file_name(dir_name=
                                    sm.make_directory(Nb_steps=2,Nb_loops=4),
                                    file_description='min_yield'))




#pm.plot_hist(dir_name='Nb_sequences_1000_Nbsteps_10_Nb_loops_1000',j=1000,i=10,seq=df)
