


import os
import submodels_module as mb

''' Run this script when first cloning DevRep to unzip all datasets '''


os.system('cd DevRep')

#specify all zip files to unzip for init here...
zip_files=['./datasets/seq_files.zip','./datasets/predicted/predicted_seq_files.zip','./model_stats/model_stat_files.zip','./trials/trials_files.zip']

for file in zip_files:
    cmd= 'unzip -o '+file+' -d '+os.path.dirname(file)
    os_out=os.system(cmd)
    if os_out is not 0:
        raise SystemError


# now much save the learned_embedding_*.pkl files for the best model.
# todo:  this is incorrect ... neeed to run main_DevRep_example.py