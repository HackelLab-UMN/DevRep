import pandas as pd 
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns





def main():

    frequency=pd.read_csv('../plot_preditions/seq_and_assay_best_sequences_heatmap.csv',header=0,index_col=0) #change to name of csv file with AA frequencies you want to plot
    fig,ax = plt.subplots(1,1,figsize=[1.75,1.75],dpi=300)

    frequency=frequency.iloc[0:20,0:16]


    plot_frequency=False #change to False if you want to plot log2 enrichments instead
    if plot_frequency:
        cmap=mpl.cm.Reds
        cmap.set_bad('gray')
        heat_map=sns.heatmap(frequency,square=True, vmin=0, vmax=1 ,cmap=cmap,cbar_kws={"shrink": 0.2,"extend":'min',"ticks":[0,0.5,1]})
        heat_map.figure.axes[-1].set_ylabel('Frequency',size=6)
    else:
        cmap=mpl.cm.bwr
        cmap.set_bad('gray')
        cmap.set_under('black')
        heat_map=sns.heatmap(frequency.transpose(),square=True, vmin=-1.1, vmax=1.1,center=0,cmap=cmap,cbar_kws={"shrink": 0.6,"extend":'min',"ticks":[]})
        heat_map.figure.axes[-1].set_ylabel('$Log_2$ Enrichment',size=6)


    heat_map.figure.axes[-1].tick_params(labelsize=6)
    ax.set_yticks([x+0.5 for x in list(range(16))])
    ax.set_yticklabels(['7','8','9','9b','9c','10','11','12','34','35','36','36b','36c','37','38','39'])
    ax.set_ylim([16.5,-0.5])
    ax.set_yticks([])
    ax.set_ylabel('Position',fontsize=6)

    ax.set_xticks([x+0.5 for x in list(range(21))])
    ax.set_xticklabels(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Gap'])
    ax.set_xlim([-0.5,20.5])
    ax.set_xticks([])
    ax.set_xlabel('Amino Acid',fontsize=6)
    ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig('./figure1c_heatmap.png')





if __name__ == '__main__':
    main()