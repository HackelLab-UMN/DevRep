import numpy as np
from itertools import combinations
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pickle import load,dump
from scipy.stats import spearmanr as spr
from scipy.stats import mannwhitneyu as mw




a=[1,2,3,4,5,6,7,8,9,10]
combin_list=[]
for i in range(1,11):
    combin_list_temp=combinations(a,i)
    for j in combin_list_temp:
        combin_list.append(j)


mi_table=load(open('mi_table.pkl','rb'))

rho_table=load(open('rho_table.pkl','rb'))

df = pd.read_pickle('./combo_cv_losses.pkl')

df_linear = pd.read_pickle('./combo_cv_losses_linear.pkl')

cfs_mi,cfs_rho=[],[]

color_list=[]

for combin in combin_list:
	if 10 in combin:
		color_list.append('red')
	else:
		color_list.append('black')
	a2y_mi,a2y_rho=0,0
	a2a_mi,a2a_rho=0,0
	for i, assay in enumerate(combin):
		a2y_mi = a2y_mi + mi_table[assay-1][10] + mi_table[assay-1][11]
		a2y_rho = a2y_rho + np.abs(rho_table[assay-1][10]) + np.abs(rho_table[assay-1][11])
		for j in range(i+1,len(combin)):
			a2a_mi = a2a_mi + mi_table[assay-1][combin[j]-1]
			a2a_rho = a2a_rho + np.abs(rho_table[assay-1][combin[j]-1])
	k = len(combin)
	cfs_mi.append(a2y_mi/np.sqrt(k+2*a2a_mi))
	cfs_rho.append(a2y_rho/np.sqrt(k+2*a2a_rho))

df['cfs_mi']=cfs_mi
df['cfs_rho']=cfs_rho
df['cv_loss_linear']=df_linear[0].values.astype('float')
df['cv_loss_best']=df[0].values.astype('float')
df['plot_color']=color_list

contains_bsh=df[df['plot_color']=='red']['cfs_mi'].values
doesnt=df[df['plot_color']=='black']['cfs_mi'].values

print(mw(contains_bsh,doesnt,alternative='greater'))
print(mw(contains_bsh,doesnt,alternative='less'))

contains_bsh=df[df['plot_color']=='red']['cv_loss_best'].values
doesnt=df[df['plot_color']=='black']['cv_loss_best'].values

print(mw(contains_bsh,doesnt,alternative='greater'))
print(mw(contains_bsh,doesnt,alternative='less'))



fig,ax = plt.subplots(2,2,figsize=[3,3],dpi=300,sharey=True,sharex='col')

for col in ['black','red']:
	df_cur=df[df['plot_color']==col]


	ax[0,0].scatter(df_cur['cfs_rho'],df_cur['cv_loss_best'],color=col,s=1,alpha=0.1)
	ax[0,0].set_ylabel('Best CV Loss (MSE)' ,fontsize=6)
	# ax[0,0].set_xlabel('CFS ('+ r'$\rho$' +')',fontsize=6)
	# ax[0,0].legend(('black','red'),fontsize=6)

	if col == 'red':
		label='Contains ' + r"$\beta_{SH}$"
	else:
		label= None
	ax[0,1].scatter(df_cur['cfs_mi'],df_cur['cv_loss_best'],color=col,s=1,alpha=0.1,label=label)
	# ax[0,1].set_xlabel('CFS (MI)', fontsize=6)
	# ax[0,1].set_ylabel('Best CV Loss (MSE)' ,fontsize=6)

	ax[1,0].scatter(df_cur['cfs_rho'],df_cur['cv_loss_linear'],color=col,s=1,alpha=0.1)
	ax[1,0].set_ylabel('Linear CV Loss (MSE)' ,fontsize=6)
	ax[1,0].set_xlabel('CFS ('+ r'$\rho$' +')',fontsize=6)

	ax[1,1].scatter(df_cur['cfs_mi'],df_cur['cv_loss_linear'],color=col,s=1,alpha=0.1)
	ax[1,1].set_xlabel('CFS (MI)', fontsize=6)
	# ax[1,1].set_ylabel('Linear CV Loss (MSE)' ,fontsize=6)

# print(df.iloc[89])
top_model_name="$P_{PK37}$, $G_{SH}$, " + r"$\beta_{SH}$"
ax[0,0].scatter(df.iloc[89]['cfs_rho'],df.iloc[89]['cv_loss_best'],c='red',s=8,alpha=1,marker='x',edgecolor ='black',lw = 1)
ax[0,1].scatter(df.iloc[89]['cfs_mi'],df.iloc[89]['cv_loss_best'],c='red',s=8,alpha=1,marker='x',edgecolor ='black',lw = 1,label=top_model_name)
ax[1,0].scatter(df.iloc[89]['cfs_rho'],df.iloc[89]['cv_loss_linear'],c='red',s=8,alpha=1,marker='x',edgecolor ='black',lw = 1)
ax[1,1].scatter(df.iloc[89]['cfs_mi'],df.iloc[89]['cv_loss_linear'],c='red',s=8,alpha=1,marker='x',edgecolor ='black',lw = 1)

leg= ax[0,1].legend(fontsize=6,edgecolor='gray',borderpad=0.1)
for lh in leg.legendHandles: 
    lh.set_alpha(.8)

ax=np.ravel(ax)
for i in range(len(ax)):
	ax[i].tick_params(axis='both', which='both', labelsize=6)

plt.tight_layout()
fig.savefig('./CFS_plot.png')

# print(spr(df['cfs_rho'],df['cv_loss_best']))
# print(spr(df['cfs_mi'],df['cv_loss_best']))
# print(spr(df['cfs_rho'],df['cv_loss_linear']))
# print(spr(df['cfs_mi'],df['cv_loss_linear']))


fig,ax = plt.subplots(1,1,figsize=[1.9,1.5],dpi=300,sharey=True,sharex='col')
for col in ['black','red']:
	df_cur=df[df['plot_color']==col]
	if col == 'red':
		label='Contains ' + r"$\beta_{SH}$"
	else:
		label= None
	ax.scatter(df_cur['cfs_mi'],df_cur['cv_loss_best'],color=col,s=1,alpha=0.1,label=label)

ax.scatter(df.iloc[89]['cfs_mi'],df.iloc[89]['cv_loss_best'],c='red',s=8,alpha=1,marker='x',edgecolor ='black',lw = 1,label=top_model_name)
leg= ax.legend(fontsize=6)
for lh in leg.legendHandles: 
    lh.set_alpha(.8)
ax.tick_params(axis='both', which='both', labelsize=6)
ax.set_ylabel('CV Loss (MSE)' ,fontsize=6)
ax.set_xlabel('CFS (MI)', fontsize=6)
ax.set_xlim([0,0.75])
ax.set_ylim([0.48,.8])
plt.tight_layout()
fig.savefig('./CFS_plot_fig5c.png')