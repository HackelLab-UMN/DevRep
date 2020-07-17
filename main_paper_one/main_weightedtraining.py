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
    compare test performances when weighting the training dataset by the average log2 number of observations
    '''


    a=int(sys.argv[1])
    if a<4:
        b=0
    elif a<8:
        a=a-4
        b=1
    else:
        print('incorrect toggle number')



    arch_list=['ridge','svm','forest','fnn']

    if b==0:
        mdl=modelbank.seqandweightedassay_to_yield_model([1,8,10],arch_list[a],1)
    elif b==1:
        mdl=modelbank.weighted_assay_to_yield_model([1,8,10],arch_list[a],1)
    mdl.cross_validate_model()
    mdl.limit_test_set([1,8,10])
    mdl.test_model()



# if __name__ == '__main__':
#     main()

loss_per_model,std_per_model=[],[]
arch_list=['ridge','svm','forest','fnn']

for i in range (4):
    cv_loss,test_loss,test_std=np.inf,np.inf,0
    for arch in arch_list:
        if i==0:
            mdl=modelbank.assay_to_yield_model([1,8,10],arch,1)
        elif i==1:
            mdl=modelbank.weighted_assay_to_yield_model([1,8,10],arch,1)
        elif i==2:
            mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch,1)
        else:
            mdl=modelbank.seqandweightedassay_to_yield_model([1,8,10],arch,1)
        if mdl.model_stats['cv_avg_loss'] < cv_loss:
            cv_loss=mdl.model_stats['cv_avg_loss']
            test_loss=mdl.model_stats['test_avg_loss']
            test_std=mdl.model_stats['test_std_loss']
    loss_per_model.append(test_loss)
    std_per_model.append(test_std)

seq_model=modelbank.seq_to_yield_model('forest',1)
seq_loss=seq_model.model_stats['test_avg_loss']
seq_std=seq_model.model_stats['test_std_loss']
x=[-0.3,0.8]
seq_plus=[seq_loss+seq_std]*2
seq_min=[seq_loss-seq_std]*2

control_model=modelbank.control_to_yield_model('ridge',1)
control_loss=control_model.model_stats['test_avg_loss']
control_model.limit_test_set([1,8,10])
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=300)

xloc=[0,0.5]
ax.axhline(seq_loss,-0.5,4.5,color='green',linestyle='--',label='Sequence Model')
ax.axhline(control_loss,-0.5,2.5,color='red',linestyle='--',label='Control')
ax.axhline(exp_var,-0.5,2.5,color='purple',linestyle='--',label='Experimental')
ax.bar(np.subtract(xloc[0],0.1),loss_per_model[0],yerr=std_per_model[0],label='Non-Weighted',width=0.2,color='blue')
ax.bar(np.add(xloc[0],0.1),loss_per_model[1],yerr=std_per_model[1],label='Weighted',width=0.2,color='blue',alpha=0.3)
ax.bar(np.subtract(xloc[1],0.1),loss_per_model[2],yerr=std_per_model[2],width=0.2,color='orange')
ax.bar(np.add(xloc[1],0.1),loss_per_model[3],yerr=std_per_model[3],width=0.2,color='orange',alpha=0.3)
ax.fill_between(x,seq_plus,seq_min,alpha=0.2,color='green')
ax.set_xticks([xloc[0]-0.1,xloc[0]+0.1,xloc[1]-0.1,xloc[1]+0.1])
ticklabels=['None','$Log_2$','None','$Log_2$']
ax.set_xticklabels(ticklabels)

# ax.legend(fontsize=6)
ax.set_xlabel('Training Sample Weighting',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_ylabel('$Test^2$ Loss',fontsize=6)
ax.set_ylim([0.35,0.8])
ax.set_xlim([-0.3,0.8])

fig.tight_layout()
fig.savefig('./Weighting_by_obs.png')
plt.close()