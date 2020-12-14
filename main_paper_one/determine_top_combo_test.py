import submodels_module as modelbank
import numpy as np
from itertools import combinations
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import load_format_data

#Determine the most generalizable model from the top CV models

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


b_models=['ridge','forest','svm','fnn']

combin_list=[[1,5,8,10],[1,8,10],[1,5,8,9,10],[1,4,5,8,10],[1,5,7,8,10],[1,5,7,8,9,10],[1,7,8,10]]
best_models, cv_stats=[],[]
for combin in combin_list:
    model_list=[]
    for arch in b_models:
        model_list.append(modelbank.assay_to_yield_model(combin,arch,1))
    results=get_loss_list(model_list)
    best_idx=results['CV Loss'].astype('float').idxmin()
    best_models.append(model_list[best_idx])
    cv_stats.append(results.iloc[best_idx])

test1_stats=[]
control_model=modelbank.control_to_yield_model('ridge',1)
control_model.limit_test_set([1,4,5,7,8,9,10])
# control_model.test_model()
test1_stats.append([control_model.model_stats['test_avg_loss'],control_model.model_stats['test_std_loss']])

exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
test1_stats.append([exp_var,0])

for model in best_models:
	model.limit_test_set([1,4,5,7,8,9,10])
	model.test_model()
	test1_stats.append([model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']])



test1_stats=np.array(test1_stats)


fig,ax=plt.subplots(1,1,figsize=[1,1.7],dpi=300)

y_pos = np.arange(len(test1_stats))
colors=['gray','gray']+['black']*(len(test1_stats)-2)
bar_list=ax.barh(y_pos,test1_stats[:,0],xerr=test1_stats[:,1],color=colors)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.tick_params(axis='y', which='major', length=0)
ax.set_xlabel('Test Loss',fontsize=6)
ax.invert_yaxis()
ax.set_xlim([0.35,0.62])
# ax[0].axis('scaled')

fig.tight_layout()
fig.savefig('./figure3c.png')
plt.close()