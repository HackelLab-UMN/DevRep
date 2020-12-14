import submodels_module as modelbank
import numpy as np
from itertools import combinations
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import load_format_data

#test the linear versus nonlinear test performance for different assay combinations 

def get_loss_list(model_list):
        model_loss_list,model_loss_std_list,model_name_list=[],[],[]
        for model in model_list:
            # model.get_best_trial()
            model_name_list.append(model.model_name)
            model_loss_list.append(model.model_stats['cv_avg_loss'])
            model_loss_std_list.append(model.model_stats['cv_std_loss'])
        results=pd.DataFrame([model_name_list,model_loss_list,model_loss_std_list])
        results=results.transpose()
        results.columns=['Model Name','CV Loss','CV Std']
        return results


lin_models=['ridge']
nonlin_models=['svm','forest','fnn']

combin_list=[[1,8,10],[1,8],[1,10],[8,10],[1],[8],[10]]
nonlin_stats,lin_stats=[],[]

#add the seq and 1,8,10 model
model_list=[]
for arch in nonlin_models:
    model_list.append(modelbank.seqandassay_to_yield_model([1,8,10],arch,1))
results=get_loss_list(model_list)
best_idx=results['CV Loss'].astype('float').idxmin()
best_nonlinmodel=model_list[best_idx]
best_nonlinmodel.limit_test_set([1,8,10])
# best_nonlinmodel.test_model()
nonlin_stats.append([best_nonlinmodel.model_stats['test_avg_loss'],best_nonlinmodel.model_stats['test_std_loss']])

linmodel=modelbank.seqandassay_to_yield_model([1,8,10],'ridge',1)
linmodel.limit_test_set([1,8,10])
# linmodel.test_model()
lin_stats.append([linmodel.model_stats['test_avg_loss'],linmodel.model_stats['test_std_loss']])

for combin in combin_list:
    model_list=[]
    for arch in nonlin_models:
        model_list.append(modelbank.assay_to_yield_model(combin,arch,1))
    results=get_loss_list(model_list)
    best_idx=results['CV Loss'].astype('float').idxmin()
    best_nonlinmodel=model_list[best_idx]
    best_nonlinmodel.limit_test_set([1,8,10])
    # best_nonlinmodel.test_model()
    nonlin_stats.append([best_nonlinmodel.model_stats['test_avg_loss'],best_nonlinmodel.model_stats['test_std_loss']])

    linmodel=modelbank.assay_to_yield_model(combin,'ridge',1)
    linmodel.limit_test_set([1,8,10])
    # linmodel.test_model()
    lin_stats.append([linmodel.model_stats['test_avg_loss'],linmodel.model_stats['test_std_loss']])

control_model=modelbank.control_to_yield_model('ridge',1)
control_model.limit_test_set([1,8,10])
# control_model.test_model()
control_model_stats=[control_model.model_stats['test_avg_loss'],control_model.model_stats['test_std_loss']]

exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

lin_stats=np.array(lin_stats)
nonlin_stats=np.array(nonlin_stats)

fig,ax=plt.subplots(1,1,figsize=[1.8,1.5],dpi=300)

y_pos = np.arange(len(lin_stats))

ax.barh(y_pos-0.2,lin_stats[:,0],xerr=lin_stats[:,1],label='Linear',height=0.4,color='#FC766AFF')
ax.barh(y_pos+0.2,nonlin_stats[:,0],xerr=nonlin_stats[:,1],label='Nonlinear',height=0.4,color="#5B84B1FF")
ax.tick_params(axis='both', which='major', labelsize=6)
ax.tick_params(axis='y', which='major', length=0)
ax.set_xlabel('Test Loss',fontsize=6)
ax.invert_yaxis()
ax.legend(fontsize=6,framealpha=1)
ax.set_xlim([0.5,1])
# ax[0].axis('scaled')

fig.tight_layout()
fig.savefig('./figure5b.png')
plt.close()