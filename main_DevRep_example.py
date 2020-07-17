####
# Example script for predicting the recombinant yield of Gp2 paratope mutants in two strains of E.coli.
# Utilizing HT assays for training devrep, and using the devrep embedding to predict yield
# assays for training devrep are currently limited to 1,8,10 as those were the most predictive in the first paper
# Below is an example of training and testing and predicting the best performing architecture of DevRep
###


import submodels_module as modelbank

#define model parameters
#assays are numbered in order as found in SI table #
#model architectures for predicting yield are: ['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
#    										    'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']

devrep_mdl_param={'assays':[1,8,10], 'model_architecture':'emb_cnn', 'sample_fraction':1}
#initialize model based upon model parameters
mdl=modelbank.seq_to_assay_model(**devrep_mdl_param)

#cross-validate model
mdl.cross_validate_model()

#test the model on the limited test set
mdl.test_model()

#return the results from cv and testing
print(mdl.model_stats)

#plot the predicted results
#figure is saved in ./figures/
mdl.plot()

#save the learned embeddings
#saved as pickled files in ./datasets/predicted/learned_embeddings_*
#saves 10 different embeddings from the 10 created models with the same hyperparameters
mdl.save_sequence_embeddings()



#load devrep embedding to yield model
#each embedding from the different models of the same architecture trains the top model independently 
#model architectures for predicting yield are 'ridge','forest','svm','fnn'
top_mdl_param={'model_architecture':'svm', 'sample_fraction':1}
devrep_param_list=[devrep_mdl_param['assays'],devrep_mdl_param['model_architecture'],devrep_mdl_param['sample_fraction']]
for i in range(10):
	#initialize embedding to yield model
	emb_mdl=modelbank.sequence_embeding_to_yield_model(devrep_param_list+[i],**top_mdl_param)

	#cross validate
	emb_mdl.cross_validate_model()

	#limit test set to compare to paper one results 
	emb_mdl.limit_test_set([1,8,10])

	#test model
	emb_mdl.test_model()

	#return the results from cv and testing
	print(emb_mdl.model_stats)

	if i==0:
		#plot the predicted results for a single embedding 
		#figure is saved in ./figures/
		emb_mdl.plot()

		# save predictions of the same set predicted in the first paper
		df_str='learned_embedding_seq_to_assay_train_1,8,10_seq_assay'+','.join([str(x) for x in devrep_mdl_param['assays']])+'_'+devrep_mdl_param['model_architecture']+'_'+str(devrep_mdl_param['sample_fraction'])+'_'+str(i)

		#final save file for example is ./datasets/predicted/learned_embedding_seq_to_assay_train_1,8,10_seq_assay1,8,10_emb_cnn_1_0_embedding_seq_assay1,8,10_emb_cnn_1_0_yield_svm_1_0.pkl
		#result is a pickeled pandas DataFrame with yields predicted under IQ_Average_bc and SH_Average_bc for the two strains
		emb_mdl.save_predictions(input_df_description=df_str)

