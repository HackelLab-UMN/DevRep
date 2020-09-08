import sys
import submodels_module as modelbank
from itertools import combinations
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats as ttest
import load_format_data
import numpy as np


def main():

    toggle_no=int(sys.argv[1])
    ## If this function is run, then the program must be run through the terminal, with an integer input. The integer input must be within
    ## the following range [0,4]. This integer input is stored in the variable toggle_no. 
    # c_models=['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
    #     'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
    c_models=['ridge','fnn','emb_fnn_flat','small_emb_rnn_linear','emb_cnn']
    ## A string list, c_models is created which has different types of regression models in it. 
    for ss in [0.01,0.1,.5]:
        ## An interable object ss is created to iterate through the list containing different sample sizes. 
        c=modelbank.seq_to_assay_model([1,8,10],c_models[toggle_no],ss)
        ## For each sample size, a seq_to_assay_model object defined in the submodels_module.py program is created. It is instantiated with a
        ## integer list listing the assays to be used to build the model, the 'toggle_no' index of the c_models to determine the regression model used
        ## and the ss iterbale to determine different sample fractions. 
        c.cross_validate_model()
        c.test_model()
        ## Then the cross_validate_model() function of the parent model class is run
        ## This determines the hyperparameters for the regression model.  Finally the
        ## hyperparameters are used along with the training dataset to train the regression model in the test_model() function.
        c.save_predictions()
        ## Then the save_predictions() function defined in the x_to_assay_model parent class is used to save the assay score predictions
        ## of the test dataset to use with the assay_to_yield_model predictions. 
        if 'emb' in c_models[toggle_no]:
            ## If the regression model used is of an embedded format then the save_sequence_embeddings() function defined in the 
            ## x_to_assay_model parent class is run to save the sequence embeddings of the model built above. 
            c.save_sequence_embeddings()





if __name__ == '__main__':
    main()
## Given that the function defined above is named main, the function is run in the following if statement. 

# c_models=['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
#         'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
# c_models.reverse()
# c_names=['Linear Model','One-Hot','Flatten AA Prop','Max AA Prop','Linear Top, Max AA Prop','Recurrent','Small Recurrent','Small Recurrent + Atn','Linear Top, Small Recurrent',
# 		'Convolutional','Small Convolutional','Small Convolutional + Atn','Linear Top, Small Convolutional']
# c_names.reverse()
# c_mdl_test_loss,c_mdl_test_std=[],[]
# for arch in c_models:
# 	c_prop=[[1,8,10],arch,1]
# 	mdl=modelbank.seq_to_assay_model(*c_prop)
# 	c_mdl_test_loss.append(mdl.model_stats['test_avg_loss'])
# 	c_mdl_test_std.append(mdl.model_stats['test_std_loss'])



# control_model=modelbank.control_to_assay_model([1,8,10],'ridge',1)
# control_loss=control_model.model_stats['test_avg_loss']
# exploded_df,_,_=load_format_data.explode_assays([1,8,10],control_model.testing_df)
# exp_var=np.average(np.square(np.array(exploded_df['y_std'])))


# fig,ax=plt.subplots(1,1,figsize=[5,5],dpi=300)
# x=[-1,len(c_models)-2]

# ax.axvline(control_loss,x[0],x[1],color='red',linestyle='--',label='Assay Type Control')
# ax.axvline(exp_var,x[0],x[1],color='purple',linestyle='--',label='Experimental Variance')

# oh_test_loss=c_mdl_test_loss[-1]
# oh_test_std=c_mdl_test_std[-1]
# ax.axvline(oh_test_loss,x[0],x[1],color='green',linestyle='--',label='One-Hot Linear')
# oh_plus=[oh_test_loss+oh_test_std]*2
# oh_min=[oh_test_loss-oh_test_std]*2
# ax.fill_betweenx(x,oh_plus,oh_min,alpha=0.2,color='green')

# oh_test_loss=c_mdl_test_loss[-2]
# oh_test_std=c_mdl_test_std[-2]
# ax.axvline(oh_test_loss,x[0],x[1],color='orange',linestyle='--',label='One-Hot FNN')
# oh_plus=[oh_test_loss+oh_test_std]*2
# oh_min=[oh_test_loss-oh_test_std]*2
# ax.fill_betweenx(x,oh_plus,oh_min,alpha=0.2,color='orange')

# c_models=c_models[:-2]
# c_mdl_test_loss=c_mdl_test_loss[:-2]
# c_mdl_test_std=c_mdl_test_std[:-2]
# c_names=c_names[:-2]
# ax.barh(range(len(c_models)),c_mdl_test_loss,yerr=c_mdl_test_std,height=0.8,color='black')
# ax.set_yticks(range(len(c_models)))
# ax.set_yticklabels(c_names)
# ax.legend(fontsize=6)
# ax.tick_params(axis='both', which='major', labelsize=6)
# ax.set_xlabel('Test Loss',fontsize=6)
# # ax.set_xlim([0.35,0.75])
# ax.set_ylim(x)
# ax.set_title('Assay Score Predictions',fontsize=6)
# fig.tight_layout()
# fig.savefig('./seq_to_assay_arch.png')
# plt.close()