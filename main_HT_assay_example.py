####
# Example script for predicting the recombinant yield of Gp2 paratope mutants in two strains of E.coli.
# Utilizing HT assays and/or paratope sequence as inputs.
# There are 10 assays from which all combinations were tried (10C1+10C2+..10C10=1024) in 4 different model architectures
# Each assay combination + architecture was cross-validated to train hyper parameters, then tested on a left-out test set
# Below shows the example of training the models shown in Figure 3d of Golinski et. al 2020.
###


import submodels_module as modelbank


#define model parameters
#assays are numbered in order as found in SI table #
#model architectures for predicting yield are: 'ridge','forest','svm','fnn'
assay_mdl_param={'assays':[1,8,10], 'model_architecture':'forest', 'sample_fraction':1}
#initialize model based upon model parameters
mdl=modelbank.assay_to_yield_model(**assay_mdl_param)

### 
# Other model options
# 
# one-hot sequence to yield model
# seq_to_yield_param={'model_architecture':'forest', 'sample_fraction':1}
# mdl=modelbank.seq_to_yield_model(**seq_to_yield_param)

# assays and sequence model
# uses same params as assay model
# mdl=modelbank.seqandassay_to_yield_model(**assay_mdl_param)

# strain only control model
# strain_only_param={'model_architecture':'ridge', 'sample_fraction':1}}


#cross-validate model
mdl.cross_validate_model()

#the test set of sequences was comprised of mutants only observed in a subset of the 10 assays
#this requires a modification to subsample the set for those sequences seen the assays the model uses
mdl.limit_test_set([1,8,10])

#test the model on the limited test set
mdl.test_model()

#return the results from cv and testing
print(mdl.model_stats)

#plot the predicted results
#figure is saved in ./figures/
mdl.plot()

#save the predicted yields for sequnces just seen in the HT assays
if not hasattr(mdl,'assay_str'):
	#set the dataset to use for control and sequence only models to compare to assay model
	mdl.assay_str=','.join([str(x) for x in [1,8,10]])
#current example is saved as: ./datasets/predicted/seq_to_assay_train_1,8,10_assays1,8,10_yield_forest_1_0.pkl
#result is a pickeled pandas DataFrame with yields predicted under IQ_Average_bc and SH_Average_bc for the two strains
mdl.save_predictions()
