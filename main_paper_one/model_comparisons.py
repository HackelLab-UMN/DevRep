import submodels_module as modelbank
import numpy as np
from itertools import combinations
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import load_format_data 
import pandas as pd



a=[1,2,3,4,5,6,7,8,9,10]
combin_list=[]
for i in range(1,11):
    combin_list_temp=combinations(a,i)
    for j in combin_list_temp:
        combin_list.append(j)
# combin_list=combin_list[0:10]

b_models=['ridge','forest','svm','fnn']
b_models=[b_models[0]]

mdl_combin,loss_list=[],[]
for combin in combin_list:
    model_list=[]
    # print(combin)
    min_model,min_loss=[],np.inf
    for arch in b_models:
        mdl=modelbank.assay_to_yield_model(combin,arch,1)
        # _=mdl.get_best_trial() #check to make sure fully trained, every model run at least 24 hours on MSI, slowest (>=5/50 iterations)
        if mdl.model_stats['cv_avg_loss'] < min_loss:
            min_model=mdl.model_name
            min_loss=mdl.model_stats['cv_avg_loss']
        del(mdl)

    mdl_combin.append(min_model)
    loss_list.append(min_loss)

control_model=modelbank.control_to_yield_model('ridge',1)
control_model_loss=control_model.model_stats['cv_avg_loss']
exploded_df,_,_=load_format_data.explode_yield(control_model.training_df)
cv_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

seq_model=modelbank.seq_to_yield_model('forest',1)
seq_model_cv_loss=seq_model.model_stats['cv_avg_loss']

fig,ax = plt.subplots(1,1,figsize=[1.25,1.5],dpi=300)
ax.hist(loss_list,bins=20,color='black',orientation='horizontal')
ax.set_ylabel('CV Loss (MSE)',fontsize=6)
ax.set_xlabel('# of Assay Combin.',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.axhline(y=control_model_loss,label='Strain Only Control',color='red')
ax.axhline(y=seq_model_cv_loss,label='OH Sequence Model',color='blue')
# ax.legend(fontsize=6,loc='center',framealpha=1)
plt.tight_layout()
fig.savefig('./combination_cv_loss.png')

a=np.array([loss_list,mdl_combin])
b=pd.DataFrame(a.transpose())
b.to_pickle('./combo_cv_losses_linear.pkl')
c=b.sort_values(by=[0])
c[0:7]
