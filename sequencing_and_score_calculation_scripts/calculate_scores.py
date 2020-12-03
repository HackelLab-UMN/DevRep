import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.linear_model import LinearRegression


def calc_slope(row):
    if row[0]>0:
        x=np.array([0,1,2,3])
        model=LinearRegression(fit_intercept=False).fit(x.reshape(-1,1),row-row[0])
        return model.coef_[0]
    else:
        return np.nan

def main():
    num_cores = multiprocessing.cpu_count()
    pool=multiprocessing.Pool(processes=12) 

    otu_table=pd.read_csv('./match_otutab.txt',sep='\t',header=0)
    # otu_table=pd.read_csv('./match_otutab.txt',sep='\t',header=0,nrows=1000)
    gate_rank=pd.read_csv('./gate_rank.csv',header=None)
    cells_per_read=pd.read_csv('./cells_per_read.csv',header=None)

    for i in range(1,4): # for each trial
        sortnum=1
        for j in range(1,32,4): #for each sort
            for k in range(0,4): #for each gate
                otu_table['RPI'+str(i)+'-'+str(j+k)]=otu_table['RPI'+str(i)+'-'+str(j+k)]*cells_per_read.iloc[j+k-1,i-1] #convert reads to cells
            otu_table['Sort'+str(sortnum)+'_'+str(i)+'_count']=otu_table['RPI'+str(i)+'-'+str(j)]+otu_table['RPI'+str(i)+'-'+str(j+1)]+otu_table['RPI'+str(i)+'-'+str(j+2)]+otu_table['RPI'+str(i)+'-'+str(j+3)] #add cells per sort
            otu_table['Sort'+str(sortnum)+'_'+str(i)+'_count'].replace(float(0),np.nan,inplace=True)
            otu_table['Sort'+str(sortnum)+'_'+str(i)+'_score']=otu_table['RPI'+str(i)+'-'+str(j)]*gate_rank.iloc[j-1,i-1]+otu_table['RPI'+str(i)+'-'+str(j+1)]*gate_rank.iloc[j+1-1,i-1]+otu_table['RPI'+str(i)+'-'+str(j+2)]*gate_rank.iloc[j+2-1,i-1]+otu_table['RPI'+str(i)+'-'+str(j+3)]*gate_rank.iloc[j+3-1,i-1] #calculate score
            otu_table['Sort'+str(sortnum)+'_'+str(i)+'_score']=otu_table['Sort'+str(sortnum)+'_'+str(i)+'_score']/otu_table['Sort'+str(sortnum)+'_'+str(i)+'_count']
            sortnum=sortnum+1
        otu_table['Sort'+str(9)+'_'+str(i)+'_count']=otu_table['RPI'+str(i)+'-'+str(33)]
        otu_table['Sort'+str(9)+'_'+str(i)+'_count'].replace(float(0),np.nan,inplace=True) #no-amp control of IQ
        otu_table['Sort'+str(10)+'_'+str(i)+'_count']=otu_table['RPI'+str(i)+'-'+str(37)]
        otu_table['Sort'+str(10)+'_'+str(i)+'_count'].replace(float(0),np.nan,inplace=True) #no-amp control of SH
        for k in range(33,41): #for each gate
            otu_table['RPI'+str(i)+'-'+str(k)+'_freq']=otu_table['RPI'+str(i)+'-'+str(k)]/cells_per_read.iloc[k-1,i-1] #calc frac of reads
        otu_table['Sort'+str(9)+'_'+str(i)+'_score']=otu_table[['RPI'+str(i)+'-33_freq','RPI'+str(i)+'-34_freq','RPI'+str(i)+'-35_freq','RPI'+str(i)+'-36_freq']].apply(calc_slope,axis=1)
        otu_table['Sort'+str(10)+'_'+str(i)+'_score']=otu_table[['RPI'+str(i)+'-37_freq','RPI'+str(i)+'-38_freq','RPI'+str(i)+'-39_freq','RPI'+str(i)+'-40_freq']].apply(calc_slope,axis=1)


    for sortnum in range(1,11): #for each sort find mean counts, score, and stdev
        otu_table['Sort'+str(sortnum)+'_mean_count']=otu_table[['Sort'+str(sortnum)+'_1_count','Sort'+str(sortnum)+'_2_count','Sort'+str(sortnum)+'_3_count']].mean(axis=1,skipna=False)
        otu_table['Sort'+str(sortnum)+'_mean_score']=otu_table[['Sort'+str(sortnum)+'_1_score','Sort'+str(sortnum)+'_2_score','Sort'+str(sortnum)+'_3_score']].mean(axis=1,skipna=False)
        otu_table['Sort'+str(sortnum)+'_std_score']=otu_table[['Sort'+str(sortnum)+'_1_score','Sort'+str(sortnum)+'_2_score','Sort'+str(sortnum)+'_3_score']].std(axis=1,skipna=False)

        fig, ax1 = plt.subplots(1)
        data_to_plot=otu_table['Sort'+str(sortnum)+'_mean_score']
        data_to_plot=data_to_plot[~np.isnan(otu_table['Sort'+str(sortnum)+'_mean_count'])]
        ax1.hist(data_to_plot,100)
        ax1.set_title('Total # of Proteins='+str(len(data_to_plot)))
        ax1.set_xlabel('Mean Assay Score (n=3)')
        ax1.set_ylabel('# of Proteins')
        fig.tight_layout()
        fig.savefig('Sort'+str(sortnum)+'_mean_score.png')

        fig, ax1 = plt.subplots(1)
        y=otu_table['Sort'+str(sortnum)+'_std_score'][~np.isnan(otu_table['Sort'+str(sortnum)+'_mean_count'])]
        x=otu_table['Sort'+str(sortnum)+'_mean_count'][~np.isnan(otu_table['Sort'+str(sortnum)+'_mean_count'])]
        _,_,_,img=ax1.hist2d(np.log10(x), y, bins=(50, 50), cmap=plt.cm.jet,cmin=1)
        fig.colorbar(img)
        ax1.set_title('Total # of Proteins='+str(len(x)))
        ax1.set_xlabel('Log10(Mean # Observations)')
        ax1.set_ylabel('Standard Deviation')
        fig.tight_layout()
        fig.savefig('Sort'+str(sortnum)+'_std_count.png')

    otu_table.to_pickle('./match_scores.pkl')

if __name__ == '__main__':
    main()