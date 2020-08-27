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

    if toggle_no>100:
    	toggle_no=toggle_no-100
    	gpu=True
    else:
    	gpu=False

    c_models=['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
        'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
    c_prop=[[1,8,10],c_models[toggle_no],1]
    # c=modelbank.seq_to_assay_model(c_prop)
    # c.save_sequence_embeddings()

    for i in range(10):
    	if gpu:
		    d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'fnn',1)
		    d.cross_validate_model()
		    d.limit_test_set([1,8,10])
		    d.test_model()
    	else:
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


# if __name__ == '__main__':
#     main()

c_models=['emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
        'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
c_models=['emb_fnn_flat','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
        'emb_cnn','small_emb_cnn','small_emb_atn_cnn']
c_models.reverse()
c_names=['Flatten AA Prop','Max AA Prop','Linear Top, Max AA Prop','Recurrent','Small Recurrent','Small Recurrent + Atn','Linear Top, Small Recurrent',
		'Convolutional','Small Convolutional','Small Convolutional + Atn','Linear Top, Small Convolutional']
c_names=['Flatten AA Prop','Small Recurrent','Small Recurrent + Atn','Linear Top, Small Recurrent',
        'Convolutional','Small Convolutional','Small Convolutional + Atn']
c_names.reverse()
a_models=['ridge','svm','forest']
c_mdl_test_loss,c_mdl_test_std=[],[]
for arch in c_models:
	c_prop=[[1,8,10],arch,1]
	min_cv_loss,min_test_loss=np.inf,np.inf
	for top_arch in a_models:
		cur_cv_loss,cur_test_loss=[],[]
		for i in range(10):
			mdl=modelbank.sequence_embeding_to_yield_model(c_prop+[i],top_arch,1)
			cur_cv_loss.append(mdl.model_stats['cv_avg_loss'])
			cur_test_loss.append(mdl.model_stats['test_avg_loss'])
		if np.mean(cur_cv_loss)<min_cv_loss:
			min_cv_loss=np.mean(cur_cv_loss)
			min_test_loss=np.mean(cur_test_loss)
			min_test_std=np.std(cur_test_loss)
	c_mdl_test_loss.append(min_test_loss)
	c_mdl_test_std.append(min_test_std)

oh_test_loss=[]
oh_model=modelbank.seq_to_yield_model('forest',1)
oh_test_loss.append(oh_model.model_stats['test_avg_loss'])
for i in range(9):
	oh_model.change_sample_seed(i)
	oh_test_loss.append(oh_model.model_stats['test_avg_loss'])
oh_test_std=np.std(oh_test_loss)
oh_test_loss=np.mean(oh_test_loss)

assay_test_loss=[]
assay_model=modelbank.assay_to_yield_model([1,8,10],'forest',1)
assay_test_loss.append(assay_model.model_stats['test_avg_loss'])
for i in range(9):
	assay_model.change_sample_seed(i)
	assay_test_loss.append(assay_model.model_stats['test_avg_loss'])
assay_test_std=np.std(assay_test_loss)
assay_test_loss=np.mean(assay_test_loss)

control_model=modelbank.control_to_yield_model('ridge',1)
control_loss=control_model.model_stats['test_avg_loss']
control_model.limit_test_set([1,8,10])
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))


fig,ax=plt.subplots(1,1,figsize=[2.5,2.5],dpi=300)
x=[-1,len(c_models)]

ax.axvline(control_loss,x[0],x[1],color='red',linestyle='--',label='Cell Type Control')

ax.axvline(assay_test_loss,x[0],x[1],color='blue',linestyle='--',label='Assay Model')
assay_plus=[assay_test_loss+assay_test_std]*2
assay_min=[assay_test_loss-assay_test_std]*2
ax.fill_betweenx(x,assay_plus,assay_min,alpha=0.2,color='blue')

ax.axvline(oh_test_loss,x[0],x[1],color='green',linestyle='--',label='One-Hot Sequence')
oh_plus=[oh_test_loss+oh_test_std]*2
oh_min=[oh_test_loss-oh_test_std]*2
ax.fill_betweenx(x,oh_plus,oh_min,alpha=0.2,color='green')

ax.axvline(exp_var,x[0],x[1],color='purple',linestyle='--',label='Experimental Variance')


ax.barh(range(len(c_models)),c_mdl_test_loss,yerr=c_mdl_test_std,height=0.8,color='black')
ax.set_yticks(range(len(c_models)))
ax.set_yticklabels(c_names)
# ax.legend(fontsize=6,framealpha=1)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('$Test^2$ Loss',fontsize=6)
ax.set_xlim([0.35,0.75])
ax.set_ylim(x)
ax.set_title('Yield Predictions',fontsize=6)
fig.tight_layout()
fig.savefig('./embed_to_yield_strategies.png')
plt.close()