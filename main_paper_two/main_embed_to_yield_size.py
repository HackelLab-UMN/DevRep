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
    ## This program is run on the terminal command prompt with an integer input in the range [0,1003]. This number is stored
    ## in the toggle_no variable.
    
    if toggle_no>=1000:
        toggle_no=toggle_no-1000
        gpu=True
        ## If the toggle_no is greater than or equal to 1000, then the toggle_no is reduced by 1000 and a boolean variable
        ## GPU is created and set to True
    else:
        gpu=False
        ## If the number is less than 1000 then the GPU boolean is set to Flase
    ## Then the code enters another if-else loop 
    if toggle_no>=200:
        toggle_no=toggle_no-200
        toggle_three=2
        ## If the toggle_no is greater than or equal 200, then the toggle_no is reduced by 200 and a new integer variable toggle_three
        ## is created and set to 2
    elif toggle_no>=100:
        toggle_no=toggle_no-100
        toggle_three=1
        ## If however the toggle_no is in the range [100,200), then the toggle_no is reduced by 100 and a new integer variable toggle_three
        ## is created and set to 1
    else:
        toggle_three=0
        ## If however the toggle is in the range [0,100), then the integer variable toglle_three is created and set to 0
    ## Then the code enters another if-else loop 
    if toggle_no>=8:
        toggle_no=toggle_no-8
        toggle_one=2
        ## If the new toggle_no is greater than or equal to 8, then the toggle_no is reduced by 8, and a new integer variable
        ## toggle_one is created and set to 2
    elif toggle_no>=4:
        toggle_no=toggle_no-4
        toggle_one=1
        ## If the new toggle_no is in the range [4,8), then the toggle_no is reduced by 4, and a new integer variable
        ## toggle_one is created and set to 1
    else:
        toggle_one=0
        ## However, if the toggle_no is less than 4, then the toggle_one variable is set to 0

    toggle_two =(toggle_no%4)

    # c_models=['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
    #     'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
    
    c_models=['emb_fnn_flat','small_emb_rnn_linear','emb_cnn']
    ## A string list c_models is created which has different types of embedded neural networks model regression stored in it
    c_ss_list=[0.01,0.1,.5,1]
    ## A float list c_ss_list is created to have different sample fraction or sample sizes in it. 
    c_prop=[[1,8,10],c_models[toggle_one],c_ss_list[toggle_two]]
    ## Finally the 'toggle_one' index position of the c_models list along with the 'toggle_two' index position of the c_ss_list
    ## along with a list of assays to be used are stored in the c_prop list
    
    # c=modelbank.seq_to_assay_model(c_prop)
    # c.save_sequence_embeddings()

    d_ss_list=[.3,.5,1]


    for i in range(10):
        ## the iterable object i is in [0,10)
        if gpu:
            ## If the gpu boolean created is true then, the following if-segment is executed
            ## Initally a sequence_embedding_to_yield_model is created and it is instantiated with the c_prop
            ## list with the iterable i also added to it, along with a 'fnn' string to indicate a feedforward neural
            ## network regression along with the 'toggle_three' index of the d_ss_list to indicate the sample fraction
            d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'fnn',d_ss_list[toggle_three])
            d.cross_validate_model()
            d.limit_test_set([1,8,10])
            d.test_model()
            ## Once the object is created the cross_validate_model() function of the model parent class is run which
            ## determined the hyperparameters for this model, then the limit_test_set() function from the x_to_yield_model
            ## parent class is run which modifies the class testing dataframe to reflect data only from the 1,8,10 assays
            ## Finally the test_model() function from the model parent class is run to train the model using the hyperparameters
            ## and the training class dataframe
            
        else:
            ## If the gpu boolean created is false then, the following else-segment is executed
            ## Initally a sequence_embedding_to_yield_model is created and it is instantiated with the c_prop
            ## list with the iterable i also added to it, along with a 'ridge' string to indicate a ridge regression
            ## along with the 'toggle_three' index of the d_ss_list to indicate the sample fraction.
            ## Once the object is created the cross_validate_model() function of the model parent class is run which
            ## determined the hyperparameters for this model, then the limit_test_set() function from the x_to_yield_model
            ## parent class is run which modifies the class testing dataframe to reflect data only from the 1,8,10 assays
            ## Finally the test_model() function from the model parent class is run to train the model using the hyperparameters
            ## and the training class dataframe. Then the same model is going to be trained with a randomforest regression andf
            ## and a epsilon-suppourt vector regression, for the same sample size and assay training and c_prop list
            d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'ridge',d_ss_list[toggle_three])
            d.cross_validate_model()
            d.limit_test_set([1,8,10])
            d.test_model()

            d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'forest',d_ss_list[toggle_three])
            d.cross_validate_model()
            d.limit_test_set([1,8,10])
            d.test_model()

            d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'svm',d_ss_list[toggle_three])
            d.cross_validate_model()
            d.limit_test_set([1,8,10])
            d.test_model()


if __name__ == '__main__':
    main()

# c_models=['emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
#         'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
# c_models.reverse()
# c_names=['Flatten AA Prop','Max AA Prop','Linear Top, Max AA Prop','Recurrent','Small Recurrent','Small Recurrent + Atn','Linear Top, Small Recurrent',
#       'Convolutional','Small Convolutional','Small Convolutional + Atn','Linear Top, Small Convolutional']
# c_names.reverse()
# a_models=['ridge','svm','forest']
# c_mdl_test_loss,c_mdl_test_std=[],[]
# for arch in c_models:
#   c_prop=[[1,8,10],arch,1]
#   min_cv_loss,min_test_loss=np.inf,np.inf
#   for top_arch in a_models:
#       cur_cv_loss,cur_test_loss=[],[]
#       for i in range(10):
#           mdl=modelbank.sequence_embeding_to_yield_model(c_prop+[i],top_arch,1)
#           cur_cv_loss.append(mdl.model_stats['cv_avg_loss'])
#           cur_test_loss.append(mdl.model_stats['test_avg_loss'])
#       if np.mean(cur_cv_loss)<min_cv_loss:
#           min_cv_loss=np.mean(cur_cv_loss)
#           min_test_loss=np.mean(cur_test_loss)
#           min_test_std=np.std(cur_test_loss)
#   c_mdl_test_loss.append(min_test_loss)
#   c_mdl_test_std.append(min_test_std)

# oh_test_loss=[]
# oh_model=modelbank.seq_to_yield_model('forest',1)
# oh_test_loss.append(oh_model.model_stats['test_avg_loss'])
# for i in range(9):
#   oh_model.change_sample_seed(i)
#   oh_test_loss.append(oh_model.model_stats['test_avg_loss'])
# oh_test_std=np.std(oh_test_loss)
# oh_test_loss=np.mean(oh_test_loss)

# assay_test_loss=[]
# assay_model=modelbank.assay_to_yield_model([1,8,10],'forest',1)
# assay_test_loss.append(assay_model.model_stats['test_avg_loss'])
# for i in range(9):
#   assay_model.change_sample_seed(i)
#   assay_test_loss.append(assay_model.model_stats['test_avg_loss'])
# assay_test_std=np.std(assay_test_loss)
# assay_test_loss=np.mean(assay_test_loss)

# control_model=modelbank.control_to_yield_model('ridge',1)
# control_loss=control_model.model_stats['test_avg_loss']
# control_model.limit_test_set([1,8,10])
# exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
# exp_var=np.average(np.square(np.array(exploded_df['y_std'])))


# fig,ax=plt.subplots(1,1,figsize=[5,5],dpi=300)
# x=[-1,len(c_models)]

# ax.axvline(control_loss,x[0],x[1],color='red',linestyle='--',label='Cell Type Control')

# ax.axvline(assay_test_loss,x[0],x[1],color='blue',linestyle='--',label='Assay Model')
# assay_plus=[assay_test_loss+assay_test_std]*2
# assay_min=[assay_test_loss-assay_test_std]*2
# ax.fill_betweenx(x,assay_plus,assay_min,alpha=0.2,color='blue')

# ax.axvline(oh_test_loss,x[0],x[1],color='green',linestyle='--',label='One-Hot Sequence')
# oh_plus=[oh_test_loss+oh_test_std]*2
# oh_min=[oh_test_loss-oh_test_std]*2
# ax.fill_betweenx(x,oh_plus,oh_min,alpha=0.2,color='green')

# ax.axvline(exp_var,x[0],x[1],color='purple',linestyle='--',label='Experimental Variance')


# ax.barh(range(len(c_models)),c_mdl_test_loss,yerr=c_mdl_test_std,height=0.8,color='black')
# ax.set_yticks(range(len(c_models)))
# ax.set_yticklabels(c_names)
# ax.legend(fontsize=6)
# ax.tick_params(axis='both', which='major', labelsize=6)
# ax.set_xlabel('$Test^2$ Loss',fontsize=6)
# ax.set_xlim([0.35,0.75])
# ax.set_ylim(x)
# ax.set_title('Yield Predictions',fontsize=6)
# fig.tight_layout()
# fig.savefig('./embed_to_yield_strategies.png')
# plt.close()