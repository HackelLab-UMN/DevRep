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
    ## This function when run should be run on the terminal and it requires an integer input in the range [0,12].this input is stored in toggle_no
    c_models=['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
        'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
    ## c_models is a string list, where each element corresponds to a regression model. 
    c=modelbank.seq_to_assay_model([1,8,10],c_models[toggle_no],1)
    ## A seq_to_assay_model object defined in submodels_module is created and instantiated with an integer list showing the assays to be used
    ## when building the model, the 'toogl_no' index of the c_model to show the kind of regression model used an a float,1, to show the sample
    ## fraction. 
    c.cross_validate_model()
    c.test_model()
    ## The cross_validate_model() and test_model() functions defined in the model parent class are run. This determines the best hyperparameters
    ## for this particular model, then the model is trained for the given hyperparameters and training dataset. 
    c.save_predictions()
    c.save_sequence_embeddings()
    ## Following this the save_prediction() and save_sequence_embeddings() functions defined in the x_to_assay_model parent class is run.
    ## The assay score predictions andthe sequence embedding of the models are saved 

# if __name__ == '__main__':
#     main()

c_models=['ridge','fnn','emb_fnn_flat','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
        'emb_cnn','small_emb_cnn','small_emb_atn_cnn']
c_models.reverse()
## c_models is a string list with each string corresponding to a regression model defined in the model_architecture.py program.
## The order of the c_models is then reversed.
c_names=['Linear Model','One-Hot','Flatten AA Prop','Small Recurrent','Small Recurrent + Atn','Linear Top, Small Recurrent',
        'Convolutional','Small Convolutional','Small Convolutional + Atn']
c_names.reverse()
## c_names is a string list where each string is the names of the bars that is to be constructed. The inital order of the c_names list is reversed.
c_mdl_test_loss,c_mdl_test_std=[],[]
## Two empty lists c_mdl_test_loss and c_mdl_test_std are created to track the regression loss and the standard deviation of the loss
## for different models.

for arch in c_models:
    ## An iterbale arch is created to work through each element in the c_models list.
	c_prop=[[1,8,10],arch,1]
    ## An integer list with different assays to be used to build a model, the iterable arch and a sample fraction of 1 are stored
    ## in a list c_prop
	mdl=modelbank.seq_to_assay_model(*c_prop)
    ## An object mdl, whoch is of type seq_to_assay_model is created. This object is defined in the submodel_module and it is instantiated with
    ## the elements of the c_prop list. 
	c_mdl_test_loss.append(mdl.model_stats['test_avg_loss'])
	c_mdl_test_std.append(mdl.model_stats['test_std_loss'])
    ## The average and standard deviation of the test regression loss, saved in the test_avg_loss and test_std_loss columns in the mdl class dataframe
    ## model_stats is accessed and added to the c_mdl_test_loss and c_mdl_test_std lists respectively. 

control_model=modelbank.control_to_assay_model([1,8,10],'ridge',1)
control_loss=control_model.model_stats['test_avg_loss']
exploded_df,_,_=load_format_data.explode_assays([1,8,10],control_model.testing_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
## A new control_to_yield_model() object os created from the submodel_module.py program, the object is instantiated with a
## ridge model regression and a sample fraction of 1. The average test loss of this model is then accessed and stored in control_loss variable.
## Then the explode_yield() function of the load_format_data.py program is run using the testing_df of the mdl object. The output is stored in exploded_df. 
## The y_std column in exploded_df is accessed and the mean squarred is calculated and stored in exp_var.

fig,ax=plt.subplots(1,1,figsize=[2.5,2.5],dpi=300)
## A Figure and an Axis object is created, the graph created has only one subplot and is of size 2.5by2.56
x=[-1,len(c_models)-2]
## An x list is created with the following elements [-1,7]
ax.axvline(control_loss,x[0],x[1],color='red',linestyle='--',label='Assay Type Control')
ax.axvline(exp_var,x[0],x[1],color='purple',linestyle='--',label='Experimental Variance')
## Two vertical lines are created on the plot, the first one is red in color and labelled 'Assay Type Control' and it is at x = control_loss.
## the second line is at x = exp_var and it is purple in color and is labelled 'Experimental Varience'. Both the lines range from [ymin,ymax]=[-1,7]
oh_test_loss=c_mdl_test_loss[-1]
oh_test_std=c_mdl_test_std[-1]
## The last elements of the c_mdl_test_loss and c_mdl_test_std are stored in the oh_test_loss and oh_test_std variables respectively.
ax.axvline(oh_test_loss,x[0],x[1],color='green',linestyle='--',label='One-Hot Linear')
## Another vertical line is constructed at x = oh_test_loss, this line is green in color and labelled 'One-Hot Linear'
oh_plus=[oh_test_loss+oh_test_std]*2
oh_min=[oh_test_loss-oh_test_std]*2
## Given that oh_test_loss and oh_test_std is know the high and low limit of the regression loss is calculated and stored in a list twice
## labelled oh_plus and oh_min respectively.
ax.fill_betweenx(x,oh_plus,oh_min,alpha=0.2,color='green')
## Two horizontal parrallel lines are constructed across the [-1,7] x-axis range, the higher line represents the higher limit of the regression loss
## and vice-versa for the oh_loss. The area between the lines are shaded green.
oh_test_loss=c_mdl_test_loss[-2]
oh_test_std=c_mdl_test_std[-2]
## The second last elements of the c_mdl_test_loss and c_mdl_test_std are stored in the oh_test_loss and oh_test_std variables respectively.
ax.axvline(oh_test_loss,x[0],x[1],color='orange',linestyle='--',label='One-Hot FNN')
## Another vertical line is constructed at x = oh_test_loss, this line is green in color and labelled 'One-Hot FNN'
oh_plus=[oh_test_loss+oh_test_std]*2
oh_min=[oh_test_loss-oh_test_std]*2
ax.fill_betweenx(x,oh_plus,oh_min,alpha=0.2,color='orange')
## Similar to the previous, two horizontal parrallel lines are constructed to represent the higher and lower limit of the regression loss and it 
## is shaded in orange. 
c_models=c_models[:-2]
c_mdl_test_loss=c_mdl_test_loss[:-2]
c_mdl_test_std=c_mdl_test_std[:-2] 
c_names=c_names[:-2]
## The last two objects from the c_models, c_mdl_test_loss, c_mdl_test_std lists and c_names are removed.
ax.barh(range(len(c_models)),c_mdl_test_loss,yerr=c_mdl_test_std,height=0.8,color='black')
## A horizontal bar graph is created with the bar positions at [0,1,2,3,4,5,6], and the respective bar heights are the indices correspodning in the 
## c_mdl_test_loss list and the bar height error bars are the respective indices in the c_mdl_test_std list. The bars are shaded in black.
ax.set_yticks(range(len(c_models)))
ax.set_yticklabels(c_names)
## The y-axis ticks are shown at [0,1,2,3,4,5,6] and labelled their corresponding index in the c_names list. 
# ax.legend(fontsize=6,framealpha=1)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('Test Loss',fontsize=6)
## Following this the tick size is changend in both the x and y axis and the x-axis is labelled 'Test Loss'
# ax.set_xlim([0.35,0.75])
ax.set_ylim(x)
ax.set_title('Assay Score Predictions',fontsize=6)
## The y-axis range is set to range from [-1,7], then the graph title is set to be 'Assay Score Predictions'
fig.tight_layout()
fig.savefig('./seq_to_assay_arch.png')
plt.close()
## Finally the graph is squeezed to fit into the figure size. Then the graph is saved as a png file under the name 'seq_to_assay_arch'
## The plyplot module is then closed.



