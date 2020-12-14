import pandas as pd 
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt




df=pd.read_pickle('seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')
num_cystines=[]
for index,seq in df.iterrows():
    num_cystines_per_seq=0
    for aa in seq['Paratope']:
        if aa=='C':
            num_cystines_per_seq=num_cystines_per_seq+1
    num_cystines.append(num_cystines_per_seq)

unique, counts = np.unique(num_cystines, return_counts=True)

predicted=dict(zip(unique,counts))


df=pd.read_pickle('seq_and_assay_best_sequences.pkl')

num_cystines=[]
for index,seq in df.iterrows():
    num_cystines_per_seq=0
    for aa in seq['Paratope']:
        if aa=='C':
            num_cystines_per_seq=num_cystines_per_seq+1
    num_cystines.append(num_cystines_per_seq)

unique, counts = np.unique(num_cystines, return_counts=True)

best=dict(zip(unique,counts))

conditional_prob=[]
error_bars=[]
for i in range(8):
    conditional_prob.append((best[i]/predicted[i]))
    error_bars.append(1/predicted[i])
fig,ax = plt.subplots(1,1,figsize=[1.25,1.25],dpi=300)

rects=ax.bar(x=list(range(8)),height=conditional_prob,yerr=error_bars,color='black',error_kw=dict(lw=1, capsize=1, capthick=1))
ax.set_xticks(list(range(8)))
ax.set_xlabel('Cys in Sequence',fontsize=6)
ax.set_ylabel('p(Dev+ | # of Cys)',fontsize=6)
# ax.set_ylim([0,1])
ax.tick_params(labelsize=6)

plt.tight_layout()
fig.savefig('./cystine_prob.png')

# fig,ax=plt.subplots(1,1,figsize=[1.25,1.25],dpi=300)
# fig,ax=plt.subplots(1,1,figsize=[5.25,5.25],dpi=300)

# lbs=[str(x) for x in range(8)]
# ax.bar(list(range(8)),percent_in_best,color='black')
# ax.set_xticks(list(range(8)))
# ax.set_ylabel('% Dev+',fontsize=6)
# ax.set_xlabel('Cys in Sequence',fontsize=6)
# ax.tick_params(labelsize=6)
# plt.tight_layout()
# fig.savefig('./cystine_pie.png')

