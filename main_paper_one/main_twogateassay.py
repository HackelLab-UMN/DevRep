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


    a=int(sys.argv[1])
    if a<4:
        b=0
    elif a<8:
        a=a-4
        b=1
    elif a<12:
        a=a-8
        b=2
    else:
        print('incorrect toggle number')



    arch_list=['ridge','svm','forest','fnn']
    stringency_list= ['high','medium','low']

    mdl=modelbank.twogate_assay_to_yield_model([1,8,10],stringency_list[b],arch_list[a],1)
    mdl.cross_validate_model()
    mdl.limit_test_set([1,8,10])
    mdl.test_model()

    mdl=modelbank.seqandtwogateassay_to_yield_model([1,8,10],stringency_list[b],arch_list[a],1)
    mdl.cross_validate_model()
    mdl.limit_test_set([1,8,10])
    mdl.test_model()





# if __name__ == '__main__':
#     main()

arch_list=['ridge','svm','forest','fnn']
stringency_list= ['high','medium','low','4-gate']

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

