
import pandas as pd
import ns_sampling_modules as sm
import numpy as np
df=pd.read_pickle('./sampling_data/Nb_sequences_1000_Nbsteps_5_Nb_loops_100000_dynamic_10/sequences_loop_60001.pkl')


y=sm.convert2numpy(df,'Developability')

idx=np.argmax(y)


print(df.loc[idx,'Ordinal'])

