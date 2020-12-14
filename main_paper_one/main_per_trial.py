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
    trains a model to predict yield only using one trial instead of average of 3
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
    elif a<16:
        a=a-12
        b=3

    else:
        print('incorrect toggle number')



    arch_list=['ridge','svm','forest','fnn']
    if b<2:

        for trial in range(1,4):
            if b==0:
                mdl=modelbank.seqandstassay_to_yield_model([1,8,10],trial,arch_list[a],1)
            elif b==1:
                mdl=modelbank.stassay_to_yield_model([1,8,10],trial,arch_list[a],1)

            mdl.cross_validate_model()
            mdl.limit_test_set([1,8,10])
            mdl.test_model()

    else:
        trials_list=[[1,2],[1,3],[2,3]]
        for trials in trials_list:
            if b==2:
                mdl=modelbank.seqandttassay_to_yield_model([1,8,10],trials,arch_list[a],1)
            elif b==3:
                mdl=modelbank.ttassay_to_yield_model([1,8,10],trials,arch_list[a],1)

            mdl.cross_validate_model()
            mdl.limit_test_set([1,8,10])
            mdl.test_model()




# if __name__ == '__main__':
#     main()

trials_list=[[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]]
arch_list=['ridge','svm','forest','fnn']

loss_per_model,std_per_model=[],[]
for i in range(2):
    loss_per_trial,std_per_trial=[],[]
    for trials in trials_list:
        low_cv,low_mse,low_std=np.inf,np.inf,0
        for arch in arch_list:
            if i==0:
                if len(trials)==1:
                    mdl=modelbank.seqandstassay_to_yield_model([1,8,10],trials[0],arch,1)
                elif len(trials)==2:
                    mdl=modelbank.seqandttassay_to_yield_model([1,8,10],trials,arch,1)
                else:
                    mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch,1)
            else:
                if len(trials)==1:
                    mdl=modelbank.stassay_to_yield_model([1,8,10],trials[0],arch,1)
                elif len(trials)==2:
                    mdl=modelbank.ttassay_to_yield_model([1,8,10],trials,arch,1) 
                else:
                    mdl=modelbank.assay_to_yield_model([1,8,10],arch,1)

            if mdl.model_stats['cv_avg_loss']<low_cv:
                low_cv=mdl.model_stats['cv_avg_loss']
                low_mse=mdl.model_stats['test_avg_loss']
                low_std=mdl.model_stats['test_std_loss']
        loss_per_trial.append(low_mse)
        std_per_trial.append(low_std)
    loss_per_model.append(loss_per_trial)
    std_per_model.append(std_per_trial)

seq_model=modelbank.seq_to_yield_model('forest',1)
# seq_model.limit_test_set([1,8,10])
# seq_model.test_model()
seq_loss=seq_model.model_stats['test_avg_loss']
seq_std=seq_model.model_stats['test_std_loss']

x=[-0.5,6.5]
seq_plus=[seq_loss+seq_std]*2
seq_min=[seq_loss-seq_std]*2

control_model=modelbank.control_to_yield_model('ridge',1)
control_loss=control_model.model_stats['test_avg_loss']
control_model.limit_test_set([1,8,10])
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))


fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=300)

# # xloc=[0,0.5,1,2,2.5,3,4]
# # ax.axhline(seq_loss,-0.5,4.5,color='green',linestyle='--',label='Sequence')
# # ax.bar(np.subtract(xloc,0.075),loss_per_model[1],yerr=std_per_model[1],label='Assays',width=0.15)
# # ax.bar(np.add(xloc,0.075),loss_per_model[0],yerr=std_per_model[0],label='Sequence and Assays',width=0.15)
# # ax.fill_between(x,seq_plus,seq_min,alpha=0.2,color='green')
# # ax.set_xticks(xloc)
# # ticklabels=[str(x) for x in trials_list]
# # ax.set_xticklabels(ticklabels)

# # ax.legend(fontsize=6)
# # ax.tick_params(axis='both', which='major', labelsize=6)
# # ax.set_ylabel('$Test^2$ Loss',fontsize=6)
# # ax.set_xlabel('Trials',fontsize=6)
# # ax.set_ylim([0.55,0.7])
# # ax.set_xlim([-0.5,4.5])

# # fig.tight_layout()
# # fig.savefig('./changing_trials.png')
# # plt.close()

# # print('seq and assays')
# # for i in range(len(loss_per_model[0])-1):
# #     print(ttest(loss_per_model[0][i],std_per_model[0][i],10,loss_per_model[0][-1],std_per_model[0][-1],10))

# # print('assays')
# # for i in range(len(loss_per_model[1])-1):
# #     print(ttest(loss_per_model[1][i],std_per_model[1][i],10,loss_per_model[1][-1],std_per_model[1][-1],10))


ax.axhline(seq_loss,-0.5,2.5,color='green',linestyle='--',label='Sequence')
ax.fill_between(x,seq_plus,seq_min,alpha=0.2,color='green')

ax.axhline(control_loss,-0.5,2.5,color='red',linestyle='--',label='Control')
ax.axhline(exp_var,-0.5,2.5,color='purple',linestyle='--',label='Experimental')

ax.errorbar([-0.05,0,0.05],loss_per_model[1][0:3],yerr=std_per_model[1][0:3],marker='o',color='blue',ls='none',alpha=0.2)
single_trial_average=np.average(loss_per_model[1][0:3])
ax.plot([-0.25,0.25],[single_trial_average]*2,linestyle='-',color='blue')

ax.errorbar([0.95,1,1.05],loss_per_model[1][3:6],yerr=std_per_model[1][3:6],marker='o',color='blue',ls='none',alpha=0.2)
two_trial_average=np.average(loss_per_model[1][3:6])
ax.plot([0.75,1.25],[two_trial_average]*2,linestyle='-',color='blue')

ax.errorbar(2,loss_per_model[1][6],yerr=std_per_model[1][6],marker='o',color='blue',ls='none',alpha=0.2)
ax.plot([1.75,2.25],[loss_per_model[1][6]]*2,linestyle='-',color='blue',label='Assay')

ax.errorbar([-0.05,0,0.05],loss_per_model[0][0:3],yerr=std_per_model[0][0:3],marker='o',color='orange',ls='none',alpha=0.2)
single_trial_average=np.average(loss_per_model[0][0:3])
ax.plot([-0.25,0.25],[single_trial_average]*2,linestyle='-',color='orange')

ax.errorbar([0.95,1,1.05],loss_per_model[0][3:6],yerr=std_per_model[0][3:6],marker='o',color='orange',ls='none',alpha=0.2)
two_trial_average=np.average(loss_per_model[0][3:6])
ax.plot([0.75,1.25],[two_trial_average]*2,linestyle='-',color='orange')

ax.errorbar(2,loss_per_model[0][6],yerr=std_per_model[0][6],marker='o',color='orange',ls='none',alpha=0.2)
ax.plot([1.75,2.25],[loss_per_model[0][6]]*2,linestyle='-',color='orange',label='Seq and Assay')



# ax.legend(fontsize=6)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['1','2','3'])
ax.set_xlim([-0.5,2.5])
ax.set_ylim([0.35,0.8])
ax.set_xlabel('Number of Trials',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_ylabel('$Test^2$ Loss',fontsize=6)
fig.tight_layout()
fig.savefig('./changing_trials.png')
plt.close()