import time
start_time=time.time()

import submodels_module as mb
import load_format_data
import pandas as pd 
import numpy as np

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
# s2a.save_sequence_embeddings(df)


#import the embedding_to_yield model (green blox)
#average prediction over the 3 different models, sum yield of both cell types
#currently using a ridge model, but will probably modify
e2y_params=['svm',1]

predicted_yield_per_model=[]
embeddings_per_model=[]
for i in range(1):
	#load model
	e2y=mb.sequence_embeding_to_yield_model(s2a_params+[i],*e2y_params)

	#save predictions from learned embeddings in s2a model
	input_df_description='learned_embedding_'+df[0]+'_'+s2a.model_name+'_'+str(i)

	# saved under input_df_description+embedding model properties, col='IQ_Average_bc','SH_Average_bc'
	e2y.save_predictions(input_df_description)

	#load predictions and add the two cell types yield together
	output_df_description='predicted/'+input_df_description+'_'+e2y.model_name+'_'+str(0)
	predicted_df=load_format_data.load_df(output_df_description)
	predicted_iq_yield=predicted_df['IQ_Average_bc'].to_numpy()
	predicted_sh_yield=predicted_df['SH_Average_bc'].to_numpy()
	predicted_added_yield=np.sum([predicted_iq_yield,predicted_sh_yield],axis=0) 
	predicted_yield_per_model.append(predicted_added_yield)
	embeddings_per_model.append(predicted_df['learned_embedding'])

#average over trials
predicted_yield_avg=np.average(predicted_yield_per_model,axis=0)

#load original df, save final df with a Developability column (which we want to maximize)
df_original=load_format_data.load_df(df[0])
df_original['Developability']=predicted_yield_avg.tolist()
embed_list=[]
for i in range(len(df_original)):
	embed_list.append(embeddings_per_model[0].iloc[i][0].tolist())
df_original['learned_embedding']=embed_list
df_original['IQ_Average_bc']=predicted_iq_yield.tolist()
df_original['SH_Average_bc']=predicted_sh_yield.tolist()

df_original.to_pickle('./datasets/'+df[0]+'_with_predictions.pkl')


print("--- %s seconds ---" % (time.time() - start_time))