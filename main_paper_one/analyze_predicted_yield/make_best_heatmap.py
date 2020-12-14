import pandas as pd 
import numpy as np
import math
from sklearn import preprocessing
from functools import partial
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors
from matplotlib.colors import LogNorm





def main():

    # pool=['seq_and_assay','seq','assay']
    # pool=pool[2]

    pool='cless_seq_and_assay'
    # pool='cplus_seq_and_assay'
    otu_table=pd.read_pickle('./'+pool+'_best_sequences.pkl')

    x_a=otu_table.loc[:,'One_Hot'].values.tolist()
    for i in range(len(x_a)):
        x_a[i]=x_a[i].tolist()
    x_a=np.array(x_a)
    
    total=np.sum(x_a,axis=0)
    frequency_new=np.log2(total/len(x_a))
    # frequency_new=total/len(x_a)
    min_new_freq=np.nanmin(np.ma.masked_invalid(frequency_new))
    frequency_new[frequency_new == -np.inf] = min_new_freq-1

    # otu_table=pd.read_pickle('./seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')
    otu_table=pd.read_pickle('otherc_less_seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')
    # otu_table=pd.read_pickle('otherc_plus_seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')

    x_a=otu_table.loc[:,'One_Hot'].values.tolist()
    for i in range(len(x_a)):
        x_a[i]=x_a[i].tolist()
    x_a=np.array(x_a)
    
    total=np.sum(x_a,axis=0)
    frequency_old=np.log2(total/len(x_a))
    # frequency_old=total/len(x_a)

    frequency=np.subtract(frequency_new,frequency_old)
    # frequency=frequency_old
    vmax=np.nanmax(np.ma.masked_invalid(frequency))
    vmin=min(frequency)
    print(vmax,vmin)

    frequency = np.ma.masked_where(frequency == -np.inf, frequency)

    frequency=frequency.reshape(16,21)
    frequency=pd.DataFrame(frequency)
    frequency.columns=list("ACDEFGHIKLMNPQRSTVWXY")
    # frequency['Stop']=frequency["Z"]
    frequency['Gap']=frequency['X']
    frequency=frequency[['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Gap']]
    frequency.index=['7','8','9','9b','9c','10','11','12','34','35','36','36b','36c','37','38','39']
    frequency['Aromatic (F,W,Y)']=frequency[['F','W','Y']].mean(axis=1)
    # frequency['Small (A,G,S)']=frequency[['A','G','C','S']].mean(axis=1)
    frequency['Small (A,C,G,S)']=frequency[['A','G','S']].mean(axis=1)
    frequency['Non-Polar Aliphatic (A,G,I,L,M,P,V)']=frequency[['P','M','I','L','V','A','G']].mean(axis=1)
    # frequency['Polar Uncharged (C,N,Q,S,T)']=frequency[['C','S','Q','S','T']].mean(axis=1)
    frequency['Polar Uncharged (N,Q,S,T)']=frequency[['S','Q','S','T']].mean(axis=1)
    frequency['Negative Charged (D,E)']=frequency[['D','E']].mean(axis=1)
    frequency['Positive Charged (H,K,R)']=frequency[['H','K','R']].mean(axis=1)
    frequency['Hydrophobic (A,F,G,I,L,M,P,V,W,Y)']=frequency[['A','F','G','I','L','M','P','V','W','Y']].mean(axis=1)
    # frequency['Hydrophilic (C,D,E,H,K,N,Q,R,S,T)']=frequency[['C','D','E','H','K','N','Q','R','S','T']].mean(axis=1)
    frequency['Hydrophilic (D,E,H,K,N,Q,R,S,T)']=frequency[['D','E','H','K','N','Q','R','S','T']].mean(axis=1)
    frequency=frequency.transpose()
    frequency['Loop 1 (8-11)']=frequency[['8','9','9b','9c','10','11']].mean(axis=1)
    frequency['Loop 2 (34-39)']=frequency[['34','35','36','36b','36c','37','38','39']].mean(axis=1)
    frequency['Loop 1 & Loop 2']=frequency[['8','9','9b','9c','10','11','34','35','36','36b','36c','37','38','39']].mean(axis=1)

    frequency=frequency[['7','8','9','9b','9c','10','11','12','34','35','36','36b','36c','37','38','39','Loop 1 (8-11)','Loop 2 (34-39)','Loop 1 & Loop 2']]
    frequency.to_csv('./'+pool+'_best_sequences_heatmap.csv')

    fig,ax = plt.subplots(1,1,figsize=[3.25,2.75],dpi=300)
    cmap=mpl.cm.bwr
    cmap.set_bad('gray')
    cmap.set_under('black')
    cmap.set_over('maroon')

    heat_map=sns.heatmap(frequency.transpose(),square=True, vmin=-1.1, vmax=1.1,center=0,cmap=cmap,cbar_kws={"shrink": 0.3,"extend":'both',"ticks":[-1,-0.5,0,0.5,1]})
    heat_map.figure.axes[-1].set_ylabel('$Log_2$ Enrichment',size=6)
    heat_map.figure.axes[-1].tick_params(labelsize=6)
    ax.set_yticks([x+0.5 for x in list(range(19))])
    ax.set_yticklabels(['7','8','9','9b','9c','10','11','12','34','35','36','36b','36c','37','38','39','Loop 1','Loop 2','Loop 1&2'])
    ax.set_ylim([19.5,-0.5])

    ax.set_xticks([x+0.5 for x in list(range(29))])
    # pooled_AA=['Aromatic (F,W,Y)','Small (A,C,G,S)','Non-Polar Aliphatic (A,G,I,L,M,P,V)','Polar Uncharged (C,N,Q,S,T)','Negative Charged (D,E)','Positive Charged (H,K,R)',
    #             'Hydrophobic (A,F,G,I,L,M,P,V,W,Y)','Hydrophilic (C,D,E,H,K,N,Q,R,S,T)']
    pooled_AA=['Aromatic','Small*','Non-Polar','Polar*','Negative','Positive',
                'Hydrophobic','Hydrophilic*']
    # ax.set_xticklabels(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','$-$']+pooled_AA)
    ax.set_xticklabels(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','$-$'],rotation=0)
    ax.set_xlim([-0.5,29.5])
    ax.tick_params(labelsize=6)
    # ax.set_xlabel('Amino Acid',fontsize=6)
    # ax.set_ylabel('Position',fontsize=6)
    plt.tight_layout(pad=0.2)
    fig.savefig('./'+pool+'_best_sequences_heatmap.png')





if __name__ == '__main__':
    main()