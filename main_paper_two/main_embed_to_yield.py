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
    ## If this function is used, then the program must be run on the terminal with an integer input, this input is stored in toggle_no.
    if toggle_no>100:
    	toggle_no=toggle_no % 13
    	gpu=True
        ## If the toggle_no is greater than 100, then a boolean is created, GPU, and it is set to True. Then the toggle_no is set to
        ## the modulus of the toggle_no and 13.
    else:
    	gpu=False
        toggle_no = toggle_no % 13
        ## If the number is less than 100, then the boolean GPU is set to false. Then the toggle_no is set to the modulus of the toggle_no
        ## and 13. 

    c_models=['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
        'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
    ## A string list, c_models, is created with different strings in it corresponding to the regression models defined in model_architecture.py program
    c_prop=[[1,8,10],c_models[toggle_no],1]
    ## c_prop has an integer list in it for the assays to be used to build the model, the toggle_no index of c_models to show the regression model
    ## that we are to use and a float to indicate sample fraction.
    # c=modelbank.seq_to_assay_model(c_prop)
    # c.save_sequence_embeddings()

    for i in range(10):
        ## An iterbale object i is created to run through the following int range [0,9]
    	if gpu:
            ## If the gpu boolean created is true,then a sequence_embedding_to_yield_model object defined in the submodels_module.py object is created 
            ## with a list containing the elements of c_prop and the iterable object, the model regression , feedforward neural network and a sample 
            ## fraction of 1. Then the cross_validate_model() function defined in the model parent class is run which determines the hyperparameters
            ## for this model. Following this the limit-test_set() function defined in the x_to_yield_model parent class is run which modifies the 
            ## testing data, testing_df, to reflect data from the 1,8 and 10 assays. Finally the test_model() function from the model parent class is 
            ## run to use the hyperparameters and the training data to train the model.
		    d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'fnn',1)
		    d.cross_validate_model()
		    d.limit_test_set([1,8,10])
		    d.test_model()
    	else:
            ## If the gpu boolean created is true,then a sequence_embedding_to_yield_model object defined in the submodels_module.py object is created 
            ## with a list containing the elements of c_prop and the iterable object, the model regression, ridge regression, and a sample 
            ## fraction of 1. Then the cross_validate_model() function defined in the model parent class is run which determines the hyperparameters
            ## for this model. Following this the limit-test_set() function defined in the x_to_yield_model parent class is run which modifies the 
            ## testing data, testing_df, to reflect data from the 1,8 and 10 assays. Finally the test_model() function from the model parent class is 
            ## run to use the hyperparameters and the training data to train the model.
		    d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'ridge',1)
		    d.cross_validate_model()
		    d.limit_test_set([1,8,10])
		    d.test_model()

		    d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'forest',1)
		    d.cross_validate_model()
		    d.limit_test_set([1,8,10])
		    d.test_model()

		    d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'svm',1)
		    d.cross_validate_model()
		    d.limit_test_set([1,8,10])
		    d.test_model()
            ## Then the same sequence_embedding_to_yield objectis built with a randomforrest regression and a epsilon-suppourt vector regression and 
            ## its hyperparameters are determined, its testing dataset is modified to relefct the 1,8 and 10 data and finally the model is trained
            ## using the hyperparameters and the training dataset. 
# if __name__ == '__main__':
#     main()

c_models=['emb_fnn_flat','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
        'emb_cnn','small_emb_cnn','small_emb_atn_cnn']
c_models.reverse()
## c_models is a string list with each string corresponding to a regression model defined in the model_architecture.py program. The c_models
## only conatins various neural network regressions in it. The order of the c_models is then reversed.
c_names=['Flatten AA Prop','Small Recurrent','Small Recurrent + Atn','Linear Top, Small Recurrent',
        'Convolutional','Small Convolutional','Small Convolutional + Atn']
c_names.reverse()
## c_names is a string list where each string is the names of the bars that is to be constructed. The inital order of the c_names list is reversed. 
a_models=['ridge','svm','forest']
## a_models is a list containing three different regression models: ridge, randomforrest and epsilon-suppourt vector regression
c_mdl_test_loss,c_mdl_test_std=[],[]
## Two empty lists c_mdl_test_loss and c_mdl_test_std are created to track the regression loss and the standard deviation of the loss
## for different models.
for arch in c_models:
    ## An iterable object arch is created to run through each element in the c_models list. 
	c_prop=[[1,8,10],arch,1]
    ## A integer list, along with the arch iterable and a sample fraction of 1, is stored in the c_prop list to be used when building models.
	min_cv_loss,min_test_loss=np.inf,np.inf
    ## Two objects min_cv_loss and min_test_loss are set to infinity to represent a worst-case model that has an infinite loss. 
	for top_arch in a_models:
        ## Another iterable object top_arch is created to run through the elements in the a_models. 
		cur_cv_loss,cur_test_loss=[],[]
        ## Two other empty lists are created to track the regression loss for the test and cross_validated model as the top_arch iterbale changes.
        ## The cross validated and test losses are stored in cur_cv_loss and cur_test_loss respectively. 
		for i in range(10):
            ## Iterable i is created to run through the integer range [0,9].
			mdl=modelbank.sequence_embeding_to_yield_model(c_prop+[i],top_arch,1)
            ## A sequence_embedding_to_yield_model object defined in submodels_module is created and it is instantiated with the c_prop list
            ## with the i appended to the list, the top_arch axting as the regression model and 1 for the sample fraction. This is stored in the 
            ## variable mdl.
			cur_cv_loss.append(mdl.model_stats['cv_avg_loss'])
			cur_test_loss.append(mdl.model_stats['test_avg_loss'])
            ## Then  the average regression loss for the cross validated model and the test model is accessed from the cv_avg_loss and test_avg_loss
            ## columns in the model_stats dataframe of the mdl object and appended to the cur_cv_loss and cur_test_loss lists respectively. 
		if np.mean(cur_cv_loss)<min_cv_loss:
			min_cv_loss=np.mean(cur_cv_loss)
			min_test_loss=np.mean(cur_test_loss)
			min_test_std=np.std(cur_test_loss)
            ## If the average of the cur_cv_loss list is less than infinity, then the average of the cur_cv_loss and cur_test_loss is calculated and stored in
            ## the min_cv_loss and min_test_loss variables respectively, then the standard deviation for the cur_test_loss list are calculated and stored in
            ## the min_test_std variable. 
	c_mdl_test_loss.append(min_test_loss)
	c_mdl_test_std.append(min_test_std)
    ## The above calculoated min_test_loss and min_test_Std values are then added to the c_mdl_test_loss and c_mdl_test_std lists respectively

oh_test_loss=[]
oh_model=modelbank.seq_to_yield_model('forest',1)
oh_test_loss.append(oh_model.model_stats['test_avg_loss'])
## Then a seq_to_yield_model object defined in submodels_module.py is created and it is instantiated with a model architecture of randomforest regression
## and a sample fraction of 1, then the average test regression loss, for this model is accessed from the model_stats dataframe and stored in a new list
## oh_test_loss. 
for i in range(9):
    ## An iterbale i is created to run through the integer range [0,9]
	oh_model.change_sample_seed(i)
    ## The change_sample_seed() function defined in the x_to_yield_model object parent class, is used to change the sample_seed class
    ## variable of the oh_model defined above to i. 
	oh_test_loss.append(oh_model.model_stats['test_avg_loss'])
    ## Then the average test loss of the updated oh_model is accessed from the model_stats dataframe and appended to the oh_test_loss created above.
oh_test_std=np.std(oh_test_loss)
oh_test_loss=np.mean(oh_test_loss)
## Then the average and standard deviation of the oh_test_loss list and stored in the variables oh_test_loss and oh_test_std respectively. 
assay_test_loss=[]
assay_model=modelbank.assay_to_yield_model([1,8,10],'forest',1)
assay_test_loss.append(assay_model.model_stats['test_avg_loss'])
## an assay_to_yield_model object defined in the submodels_module.py is created with an integer list to show the assay to be used to build the
## model, the model is going to use a random forest regression model and it has a sample fraction of 1. The object is stored under the name
## assay_model. Then the average test loss of the assay_model present in the class dataframe model_stats is accessed and stored in a new list 
## called assay_test_loss. 
for i in range(9):
	assay_model.change_sample_seed(i)
	assay_test_loss.append(assay_model.model_stats['test_avg_loss'])
    ## Similar to what happend with the oh_model, the assay_model 's sample_seed value is changed and the new average test loss is stored 
    ## in the assay_test_loss list cretaed. 
assay_test_std=np.std(assay_test_loss)
assay_test_loss=np.mean(assay_test_loss)
## Then the average and standard deviation of the assay_test_loss list is calculated and set to assay_test_loss and assay_test_std respectively.

control_model=modelbank.control_to_yield_model('ridge',1)
control_loss=control_model.model_stats['test_avg_loss']
control_model.limit_test_set([1,8,10])
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
## A new control_to_yield_model() object os created from the submodel_module.py program, the object is instantiated with a
## ridge model regression and a sample fraction of 1. The average test loss of this model is then accessed and stored in control_loss variable.
## Then using the limit_test_set() function of the x_to_yield_model parent class the testing_df class dataframe is changed to reflect the 1,8,10 assay scores. 
## Then the explode_yield() function of the load_format_data.py program is run using the testing_df of the mdl object. The output is stored in exploded_df. 
## The y_std column in exploded_df is accessed and the mean squarred is calculated and stored in exp_var.

fig,ax=plt.subplots(1,1,figsize=[2.5,2.5],dpi=300)
## A Figure and an Axis object is created, the graph created has only one subplot in it and the figure size is a 2.5 by 2.5
x=[-1,len(c_models)]
## A x list is created to be [-1,7]
ax.axvline(control_loss,x[0],x[1],color='red',linestyle='--',label='Cell Type Control')
## A vertical line is first created at the control_loss position (in the x-axis) and it is red in color and labelled 'Cell Type Control'
## The line ranges from [-1,7]=[ymin,ymax]
ax.axvline(assay_test_loss,x[0],x[1],color='blue',linestyle='--',label='Assay Model')
## Similarly another vertical line is constructed at the assay_test_loss position (in the x-axis) and it is blue in color and labelled
## 'Assay Model', this line also stretches along the same length as the previous line
assay_plus=[assay_test_loss+assay_test_std]*2
assay_min=[assay_test_loss-assay_test_std]*2
## Given the standard deviation, the upper and lower limit of the assay_model's regression loss is calculated and stored in a list twice
## under the names assay_plus and assay_min.
ax.fill_betweenx(x,assay_plus,assay_min,alpha=0.2,color='blue')
## Two horizontal parrallel lines are constructed across the [-1,7] x-axis rnge, the higher line represents the higher limit of the regression loss
## and vice-versa. The area between the lines are shaded blue.
ax.axvline(oh_test_loss,x[0],x[1],color='green',linestyle='--',label='One-Hot Sequence')
## Another vertical line is constructed at the oh_test_loss position (in the x-axis), and it is green in color and labelled 'One_Hot Sequence'
## this line also stretches along the same length as the previous vertical line constructed. 
oh_plus=[oh_test_loss+oh_test_std]*2
oh_min=[oh_test_loss-oh_test_std]*2
ax.fill_betweenx(x,oh_plus,oh_min,alpha=0.2,color='green')
## Similar to the high and low limits constructed for the assa_model's regression loss, the same is done for the oh_model and it is also 
## plotted as two horizontal parrallel lines which is shaded in green. 
ax.axvline(exp_var,x[0],x[1],color='purple',linestyle='--',label='Experimental Variance')
## Another vertical line is constructed at the exp_var position on the x-axis and it is purple in color and labelled 'Experimental Variance'
ax.barh(range(len(c_models)),c_mdl_test_loss,yerr=c_mdl_test_std,height=0.8,color='black')
## Then a horizontal bar graph is constructed, the bars are position at [0,1,2,3,4,5,6] and the heights are the given by the respective indices
## in the c_mdl_test_loss list and the bar heigh error bars are given by the respective indices in c_mdl_test_std list. The bars are constructed black in color
ax.set_yticks(range(len(c_models)))
ax.set_yticklabels(c_names)
# ax.legend(fontsize=6,framealpha=1)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('$Test^2$ Loss',fontsize=6)
ax.set_xlim([0.35,0.75])
ax.set_ylim(x)
ax.set_title('Yield Predictions',fontsize=6)
## First the ticks along the y-axis are marked at the [0,1,2,3,4,5,6] positions and they are labelled with the strings in c_names
## Then the fontsize of the ticks on both the axes are changed. Following this the x-axis is labelled '$Test^2$ Loss' and the x-axis and
## y-axis range is set to [035,0.75] and [-1,7] respectively. The figure is then given the title: 'Yield Predictions'.
fig.tight_layout()
fig.savefig('./embed_to_yield_strategies.png')
plt.close()
## Finally the plot is squeezed to fit within the figure size and it is saved as a png file under the name 'embed_to_yield_strategies'.
## Finally the pyplot module is closed 



