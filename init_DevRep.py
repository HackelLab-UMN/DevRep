


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

# set pandas df of sequences to be predicted, must contain a "Ordinal" column of paratope
# the file should be saved under /datasets/
df=['seq_to_assay_train_1,8,10'] #this is just an example

#import sequence_to_assay model (red box)
#currently use a embedding_fnn_linear model to predict assays 1,8,10.
#will probabaly change when I find the most accurate model
s2a_params=[[1,8,10],'emb_cnn',1]
s2a=mb.seq_to_assay_model(*s2a_params)

#now save the sequence embeddings, file is under /datasets/predicted/learned_embedding_[model properties], col='learned_embedding'
#saves 3 different embeddings from 3 different models
s2a.save_sequence_embeddings(df)
