import sys
import submodels_module as modelbank
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats as ttest
import load_format_data

def main():
    '''
    compare test performances when reducing training sample size. This version is for first paper, predicting yield from assays and one-hot encoded sequence. 
    '''
    ## This program should be run on the terminal with an integer input. The integer input should be within the following range: [0,11]
    ## This number is then saved in integer a. 
    a=int(sys.argv[1])
    if a<4:
        b=0
        ## If a is less than 4, then the integer b is set 0.
    elif a<8:
        a=a-4
        b=1
        ## If a is in the range [4,7], then integer b is set to 1 and a is reduced by 4. 
    elif a<12:
        a=a-8
        b=2
        ## If a is in the range [8,11], then integer b is set to 2 and a is reduced by 8. 
    else:
        print('incorrect toggle number')
        ## If the integer input is outside the valid range then an error message is printed. 

    arch_list=['ridge','svm','forest','fnn']
    stringency_list= ['high','medium','low']
    ## Two string lists are created one contains the different types of regression models,arch_list, the other one
    ## contains various levels, stringency_list. 
    mdl=modelbank.twogate_assay_to_yield_model([1,8,10],stringency_list[b],arch_list[a],1)
    ## Then a twogate_assay_to_yield_model object defined in submodels_module is created and it is instantiated with
    ## an integer list to show the assays used to construct the model, the 'b' index of the stringency_list, the 'a' index
    ## of the arch_list for the regression model and thn float 1 for sample fraction. 
    mdl.cross_validate_model()
    mdl.limit_test_set([1,8,10])
    mdl.test_model()
    ## Initally the cross_validate_model() function of the parent model class is run
    ## This determines the hyperparameters for the regression model. Then the limit_test_set()
    ## function of the parent class x_to_yield_model is run to modify the testing_df class dataframe
    ## to reflect only the 1,8,10 assays scores. Finally the hyperparameters are used along with the 
    ## training dataset to train the regression model.
    mdl=modelbank.seqandtwogateassay_to_yield_model([1,8,10],stringency_list[b],arch_list[a],1)
    ## Then another submodel_module object called seqandtwogatesassay_to_yield_model is created and it is instantiated with an 
    ## integer list to show the assays used to construct the model, the b index of the stringency_list and the 'a' index of the
    ## arch_list to show regression model  and a float,1, to show the sample fraction. 
    mdl.cross_validate_model()
    mdl.limit_test_set([1,8,10])
    mdl.test_model()
    ## Then like above the cross_validate_model(), limit_test_set() and test_model() function are run. 

# if __name__ == '__main__':
#     main()

arch_list=['ridge','svm','forest','fnn']
stringency_list= ['high','medium','low','4-gate']
## Two string lists are created one contains the different types of regression models,arch_list, the other one
## contains various levels, stringency_list. 
loss_per_mdl,std_per_mdl=[],[]
for i in range (2):
    loss_per_str,std_per_str=[],[]
    for stringency in stringency_list:
        cv_loss,test_loss,test_std=np.inf,np.inf,0
        for arch in arch_list:
            if i==0:
                if stringency=='4-gate':
                    mdl=modelbank.assay_to_yield_model([1,8,10],arch,1)
                else:
                    mdl=modelbank.twogate_assay_to_yield_model([1,8,10],stringency,arch,1)
            else:
                if stringency=='4-gate':
                    mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch,1)
                else:
                    mdl=modelbank.seqandtwogateassay_to_yield_model([1,8,10],stringency,arch,1)
            if mdl.model_stats['cv_avg_loss'] < cv_loss:
                cv_loss=mdl.model_stats['cv_avg_loss']
                test_loss=mdl.model_stats['test_avg_loss']
                test_std=mdl.model_stats['test_std_loss']
        loss_per_str.append(test_loss)
        std_per_str.append(test_std)
    loss_per_mdl.append(loss_per_str)
    std_per_mdl.append(std_per_str)

seq_model=modelbank.seq_to_yield_model('forest',1)
seq_loss=seq_model.model_stats['test_avg_loss']
seq_std=seq_model.model_stats['test_std_loss']
x=[-0.3,2.3]
seq_plus=[seq_loss+seq_std]*2
seq_min=[seq_loss-seq_std]*2

control_model=modelbank.control_to_yield_model('ridge',1)
control_loss=control_model.model_stats['test_avg_loss']
control_model.limit_test_set([1,8,10])
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=300)

xloc=[1.2,0.6,0,2]
ax.axhline(seq_loss,-0.5,4.5,color='green',linestyle='--',label='Sequence Model')
ax.axhline(control_loss,-0.5,2.5,color='red',linestyle='--',label='Control')
ax.axhline(exp_var,-0.5,2.5,color='purple',linestyle='--',label='Experimental')
ax.bar(np.subtract(xloc,0.075),loss_per_mdl[0],yerr=std_per_mdl[0],label='Assay',width=0.15,color='blue')
ax.bar(np.add(xloc,0.075),loss_per_mdl[1],yerr=std_per_mdl[1],label='Seq and Assay',width=0.15,color='orange')
ax.fill_between(x,seq_plus,seq_min,alpha=0.2,color='green')
ax.set_xticks(xloc)
ticklabels=['High','Medium','Low','4-Gate']
ax.set_xlabel('Sort Stringency',fontsize=6)
ax.set_xticklabels(ticklabels)

# ax.legend(fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_ylabel('$Test^2$ Loss',fontsize=6)
ax.set_ylim([0.35,0.8])
ax.set_xlim([-0.3,2.3])

fig.tight_layout()
fig.savefig('./two_gate.png')
plt.close()

