import multiprocessing
from Bio import SeqIO
import numpy as np
from functools import partial
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from scipy.stats import mannwhitneyu as mw




sorts=["$P_{PK37}$","$P_{Urea}$","$P_{Guan}$","$P_{PK55}$","$P_{TL55}$","$P_{TL75}$","$G_{I^q}$","$G_{SH}$",r"$\beta_{I^q}$",r"$\beta_{SH}$",'$Y_{I^q}$','$Y_{SH}$']


best_data=pd.read_pickle('./seq_and_assay_best_sequences.pkl')
original_data=pd.read_pickle('seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl') #all predicted

#matplotlib plot
# fig,ax=plt.subplots(1,1,figsize=[1.5,2.5],dpi=300)
# data=[]
# for i in [1,8,10]:
#     data.append(best_data['Sort'+str(i)+'_mean_score'].tolist())
# violin_parts=ax.violinplot(data,positions=[0,1,2],showmedians=True,showextrema=False,points=100,widths=.9)
# for pc in violin_parts['bodies']:
#     pc.set_color('k')
# violin_parts['cmedians'].set_color('r')
# ax.set_xticks([0,1,2])
# ax.set_xticklabels([sorts[0],sorts[7],sorts[9]],rotation=90)
# ax.set_ylim([0,1])
# ax.set_ylabel('Assay Score',fontsize=6)
# ax.tick_params(axis='both', which='major', labelsize=6)
# fig.tight_layout()
# fig.savefig('./best_score.png')
# plt.close

#seaborn plot
fig,ax=plt.subplots(1,1,figsize=[1.9,1.4],dpi=300)
best_sort_list,original_sort_list=[],[]
for i in [1,8,10]:
    best_sort_cols=best_data[['Sort'+str(i)+'_mean_score']]
    best_sort_cols['Assay Score']=best_sort_cols['Sort'+str(i)+'_mean_score']
    best_sort_cols['Assay Name']=sorts[i-1]
    best_sort_cols=best_sort_cols.drop('Sort'+str(i)+'_mean_score',axis='columns')
    best_sort_list.append(best_sort_cols)

    original_sort_cols=original_data[['Sort'+str(i)+'_mean_score']]
    original_sort_cols['Assay Score']=original_sort_cols['Sort'+str(i)+'_mean_score']
    original_sort_cols['Assay Name']=sorts[i-1]
    original_sort_cols=original_sort_cols.drop('Sort'+str(i)+'_mean_score',axis='columns')
    original_sort_list.append(original_sort_cols)
best_sort=pd.concat(best_sort_list)
best_sort['Sequence Population']=['$Dev^+$']*len(best_sort)

original_sort=pd.concat(original_sort_list)
original_sort['Sequence Population']=['All Predicted']*len(original_sort)

comb_sort=pd.concat([best_sort,original_sort])

#test if increasing or decreasing scores
b=best_sort[best_sort['Assay Name']==sorts[9]]['Assay Score']
o=original_sort[original_sort['Assay Name']==sorts[9]]['Assay Score']
print(b)
print(mw(b,o,alternative='greater'))
print(mw(b,o,alternative='less'))


a_palette = ['#1f77b4', '#FF0000'] #match colors 
sns.set_palette(a_palette)
ax=sns.violinplot(data=comb_sort,x='Assay Name',y='Assay Score',hue='Sequence Population',hue_order=['All Predicted','$Dev^+$'],cut=0,linewidth=.75,split=True,gridsize=20,scale='width',inner="quart",saturation=1)
ax.set_ylim([0,1])
for l in ax.lines:
    l.set_linewidth(0.75)
    l.set_color('black')
    l.set_alpha(.75)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.xaxis.label.set_size(6)
ax.yaxis.label.set_size(6)
ax.xaxis.label.set_visible(False)
ax.set_xlabel('blank',fontsize=6)
ax.legend(fontsize=6,loc='best')
# ax.legend().set_visible(False)
fig.tight_layout()
fig.savefig('./sort_score_comparisons.png')
plt.close



#plot interactions between assays
# combolist=[[1,8],[1,10],[8,10]]
# for i,combo in enumerate(combolist):
#     fig,ax=plt.subplots(1,1,figsize=[1.5,1.5])
#     x_data=best_data['Sort'+str(combo[0])+'_mean_score'].tolist()
#     y_data=best_data['Sort'+str(combo[1])+'_mean_score'].tolist()
#     # s=sns.jointplot(x=x_data,y=y_data,kind='kde',height=2,space=0,ratio=4,xlim=[0,1],ylim=[0,1])
#     s=sns.kdeplot(x_data,y_data,shade=True,shade_lowest=False,ax=ax)
#     # s.ax_joint.set_xticklabels(s.ax_joint.get_xmajorticklabels(),fontsize=6)
#     # s.ax_joint.set_yticklabels(s.ax_joint.get_ymajorticklabels(),fontsize=6)
#     # ax.set_xticks([0,0.5,1])
#     # ax.set_yticks([0,0.5,1])
#     if i==0:
#         ax.set_xlim([0.8,1])
#         ax.set_ylim([0,0.5])
#     elif i==1:
#         ax.set_xlim([0.8,1])
#         ax.set_ylim([0.25,0.75])
#     elif i==2:
#         ax.set_xlim([0,0.5])
#         ax.set_ylim([0.25,0.75])
#     ax.tick_params(axis='both', which='major', labelsize=6)
#     ax.set_xlabel(sorts[combo[0]-1],fontsize=6)
#     ax.set_ylabel(sorts[combo[1]-1],fontsize=6)
#     # s.ax_joint.set_xlabel(sorts[combo[0]-1],fontsize=6)
#     # s.ax_joint.set_ylabel(sorts[combo[1]-1],fontsize=6)
#     plt.tight_layout()
#     plt.savefig('./best_score_combo'+str(combo[0])+'_'+str(combo[1])+'.png',dpi=300)
#     plt.close()


