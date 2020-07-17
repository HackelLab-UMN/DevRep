import sys
import submodels_module as modelbank
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
    elif a==12:
        b=3
        a=a-12
    else:
        print('incorrect toggle number')



    arch_list=['ridge','svm','forest','fnn']

    # size_list=[0.055,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    size_list=[0.7,0.8,0.9,1]

    for size in size_list:
        if b==0:
            mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch_list[a],size)
        elif b==1: #1,5,9,12
            mdl=modelbank.assay_to_yield_model([1,8,10],arch_list[a],size)
        elif b==2: 
            mdl=modelbank.seq_to_yield_model(arch_list[a],size)
        elif b==3:
            mdl=modelbank.control_to_yield_model(arch_list[a],size)

        for seed in range(9): #no seed is seed=42
            mdl.change_sample_seed(seed)
            mdl.cross_validate_model()
            mdl.limit_test_set([1,8,10])
            mdl.test_model()



# if __name__ == '__main__':
#     main()
arch_list=['ridge','svm','forest']
best_arch_list=[]
size_list=[0.055,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
loss_per_model,std_per_model=[],[]
cv_loss_per_model,cv_std_per_model=[],[]
for b in range(4):
    loss_per_size,std_per_size=[],[]
    cv_loss_per_size,cv_std_per_size=[],[]
    for size in size_list:
        min_cv,min_cv_std,min_test,min_std=np.inf,0,np.inf,0

        for arch in arch_list:
            if b==0:
                mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch,size)
            elif b==1: #1,5,9,12
                mdl=modelbank.assay_to_yield_model([1,8,10],arch,size)
            elif b==2: 
                mdl=modelbank.seq_to_yield_model(arch,size)
            elif b==3:
                mdl=modelbank.control_to_yield_model('ridge',size)

            cur_cv_loss=[]
            cur_test_loss=[]
            cur_cv_loss.append(mdl.model_stats['cv_avg_loss'])
            cur_test_loss.append(mdl.model_stats['test_avg_loss'])
            for seed in range(9):
                mdl.change_sample_seed(seed)
                cur_cv_loss.append(mdl.model_stats['cv_avg_loss'])
                cur_test_loss.append(mdl.model_stats['test_avg_loss'])
            if np.average(cur_cv_loss)<min_cv:
                min_cv=np.average(cur_cv_loss)
                min_cv_std=np.std(cur_cv_loss)
                if cur_test_loss[0]==np.inf:
        
                    print(mdl.model_name)
                    print(cur_test_loss)
                min_test=np.average(cur_test_loss)
                min_std=np.std(cur_test_loss)
                best_arch=arch

        best_arch_list.append(best_arch)
        loss_per_size.append(min_test)
        std_per_size.append(min_std)
        cv_loss_per_size.append(min_cv)
        cv_std_per_size.append(min_cv_std)

    loss_per_model.append(loss_per_size)
    std_per_model.append(std_per_size)
    cv_loss_per_model.append(cv_loss_per_size)
    cv_std_per_model.append(cv_std_per_size)

size_list=np.multiply(size_list,len(mdl.training_df))

control_model=modelbank.control_to_yield_model('ridge',1)
control_model.limit_test_set([1,8,10])
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

exploded_df,_,_=load_format_data.explode_yield(control_model.training_df)
cv_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

fig,ax=plt.subplots(1,2,figsize=[4,2],dpi=300,sharey=True)

ax[0].errorbar(np.add(size_list,0),cv_loss_per_model[3],yerr=cv_std_per_model[3],label='Strain Only',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='red')
ax[0].errorbar(np.add(size_list,0),cv_loss_per_model[2],yerr=cv_std_per_model[2],label='OH Sequence',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='blue')
ax[0].errorbar(np.add(size_list,0),cv_loss_per_model[1],yerr=cv_std_per_model[1],label=r"$P_{PK37},G_{SH},\beta_{SH}$",marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='black')
ax[0].errorbar(np.add(size_list,0),cv_loss_per_model[0],yerr=cv_std_per_model[0],label=r"$Seq.&\ P_{PK37},G_{SH},\beta_{SH}$",marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='orange')
ax[0].axhline(cv_exp_var,0,198,color='purple',linestyle='--',label='Experimental Variance')
ax[0].legend(fontsize=6,framealpha=1)
ax[0].tick_params(axis='both', which='major', labelsize=6)
ax[0].set_ylabel('CV Loss',fontsize=6)
ax[0].set_xlabel('Number of Training Sequences',fontsize=6)
ax[0].set_ylim([0.3,1])
# ax[0].axis('scaled')

ax[1].errorbar(np.add(size_list,0),loss_per_model[3],yerr=std_per_model[3],label='Strain Only',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='red')
ax[1].errorbar(np.add(size_list,0),loss_per_model[2],yerr=std_per_model[2],label='OH Sequence',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='blue')
ax[1].errorbar(np.add(size_list,0),loss_per_model[1],yerr=std_per_model[1],label='Assays',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='black')
ax[1].errorbar(np.add(size_list,0),loss_per_model[0],yerr=std_per_model[0],label='Sequence and Assays',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='orange')
ax[1].axhline(exp_var,0,198,color='purple',linestyle='--',label='Experimental Variance')
ax[1].tick_params(axis='both', which='major', labelsize=6)
ax[1].set_ylabel('$Test^2$ Loss',fontsize=6)
ax[1].set_xlabel('Number of Training Sequences',fontsize=6)
ax[1].set_ylim([0.3,1])
# ax[1].axis('scaled')

fig.tight_layout()
fig.savefig('./changing_sample_size.png')
plt.close()

