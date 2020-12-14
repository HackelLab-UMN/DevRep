import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from joblib import dump, load


sets=['purple','green','orange','red']


fig, ax=plt.subplots(1,3,figsize=[4.3,1.3],dpi=300)

for i in sets:
	data=pd.read_csv('./'+i+'_'+'yeast.txt',sep='\t',header=None)
	x=data[0].to_numpy()
	x[x<=0]=1
	x=np.log10(x)
	y=data[1].to_numpy()
	y[y<=0]=1
	y=np.log10(y)
	# ax[0].scatter(data[0].tolist(),data[1].tolist(),color=i,alpha=0.1,s=1)
	sns.kdeplot(x,y,color=i,shade=True,ax=ax[0],shade_lowest=False)
ax[0].set_xlabel('HA Signal',fontsize=6)
ax[0].set_ylabel('c-Myc Signal',fontsize=6)
ax[0].set_xticks([])
ax[0].set_ylim([1,4.6])
ax[0].set_yticks([])

x_total=[]
cutoff=[]
for i in sets:
	data=pd.read_csv('./'+i+'_'+'gfp.txt',sep='\t',header=None)
	x=data[0].to_numpy()
	for y in x:
		x_total.append(y)
	cutoff.append(min(x))
adj=min(x_total)+0.1
x_total=x_total-adj
x_total=np.log10(x_total)

N,bins,patches=ax[1].hist(x_total,bins=50)
for i in range(len(patches)):
	if bins[i]>np.log10(cutoff[0]-adj):
		patches[i].set_facecolor('purple')
		patches[i].set_alpha(0.7)
	elif bins[i]>np.log10(cutoff[1]-adj):
		patches[i].set_facecolor('green')
		patches[i].set_alpha(0.7)
	elif bins[i]>np.log10(cutoff[2]-adj):
		patches[i].set_facecolor('orange')
		patches[i].set_alpha(0.7)
	else:
		patches[i].set_facecolor('red')
		patches[i].set_alpha(0.7)
ax[1].set_xlabel('GFP Signal',fontsize=6)
ax[1].set_ylabel('% of Cells',fontsize=6)
ax[1].set_xlim([1,5])
ax[1].set_xticks([])
ax[1].set_yticks([])


data=pd.read_pickle('../merge_data/name_scores.pkl')

freq_list=[]
for i in [37,38,39,40]:
	freq_list.append('RPI1-'+str(i)+'_freq')
data=data[data['Sort10_1_count']>100]


scale1=load('../make_datasets/Sort10_quantileTransformer.joblib')
scale2=load('../make_datasets/Sort10_minmaxscaler.joblib')

score=data['Sort10_1_score'].to_numpy()
score=scale1.transform(score.reshape(-1,1))
score=scale2.transform(score.reshape(-1,1))

for i in range(250):
	x=np.array([0,1,2,3],dtype=np.float)
	x = x[:,np.newaxis]
	y_o=data.iloc[i][freq_list[0]]
	y=np.array(data.iloc[i][freq_list].to_numpy()-y_o,dtype=np.float)
	print(x)
	print(y)
	# b, m = polyfit(x, y, 1)
	m, _, _, _ = np.linalg.lstsq(x, y)
	if score[i]>0.7:
		color='purple'
	elif score[i]>0.4:
		color='green'
	elif score[i]>0.25:
		color='orange'
	else:
		color='red'
	ax[2].plot(x,m*x,color=color,alpha=0.1,linewidth=1)



ax[2].set_xlabel('[Ampicillin]',fontsize=6)
ax[2].set_ylabel('% of Cells',fontsize=6)
ax[2].set_ylim([-0.00005,0.0001])
ax[2].set_yticks([])
ax[2].set_xticks([0,1,2,3])
ax[2].tick_params(axis='x',direction='in')
ax[2].set_xticklabels([])









fig.tight_layout()
fig.savefig('Fig1B.png')
plt.close()