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
from itertools import combinations



def get_combo_freq(sites,x_a,x_b):
    has_combo_a,has_combo_b=0,0
    for i in range(len(x_a)):
        if len(sites)==2:
            if (x_a[i,sites[0],1]==1) and (x_a[i,sites[1],1]==1):
                has_combo_a=has_combo_a+1
        elif len(sites)==3:
            if (x_a[i,sites[0],1]==1) and (x_a[i,sites[1],1]==1) and (x_a[i,sites[2],1]==1):
                has_combo_a=has_combo_a+1
        elif len(sites)==4:
            if (x_a[i,sites[0],1]==1) and (x_a[i,sites[1],1]==1) and (x_a[i,sites[2],1]==1) and (x_a[i,sites[3],1]==1):
                has_combo_a=has_combo_a+1

    for i in range(len(x_b)):
        if len(sites)==2:
            if (x_b[i,sites[0],1]==1) and (x_b[i,sites[1],1]==1):
                has_combo_b=has_combo_b+1
        elif len(sites)==3:
            if (x_b[i,sites[0],1]==1) and (x_b[i,sites[1],1]==1) and (x_b[i,sites[2],1]==1):
                has_combo_b=has_combo_b+1
        elif len(sites)==4:
            if (x_b[i,sites[0],1]==1) and (x_b[i,sites[1],1]==1) and (x_b[i,sites[2],1]==1) and (x_b[i,sites[3],1]==1):
                has_combo_b=has_combo_b+1


    pina=has_combo_a/len(x_a)
    pinb=has_combo_b/len(x_b)

    return pina, np.log2(pina)-np.log2(pinb)

def limit_by_nocys(df,num_cys_req):
    num_cystines=[]
    for index,seq in df.iterrows():
        num_cystines_per_seq=0
        for aa in seq['Paratope']:
            if aa=='C':
                num_cystines_per_seq=num_cystines_per_seq+1
        num_cystines.append(num_cystines_per_seq)

    df['Number of Cystines']=num_cystines

    return df[df['Number of Cystines']==num_cys_req]


#load data
no_cys=3
pool=['seq_and_assay','seq','assay']
pool=pool[0]

#get Dev+ combinations
otu_table=pd.read_pickle('./'+pool+'_best_sequences.pkl')
otu_table=limit_by_nocys(otu_table,no_cys)

x_a=otu_table.loc[:,'One_Hot'].values.tolist()
for i in range(len(x_a)):
    x_a[i]=x_a[i].tolist()
x_a=np.array(x_a)
total=np.sum(x_a,axis=0)
frequency=total/len(x_a)
frequency=frequency.reshape(16,21)
frequency=pd.DataFrame(frequency)
frequency.columns=list("ACDEFGHIKLMNPQRSTVWXY")
frequency['Gap']=frequency['X']
frequency=frequency[['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Gap']]
frequency.index=['7','8','9','9b','9c','10','11','12','34','35','36','36b','36c','37','38','39']


a=['7','8','9','9b','9c','10','11','12','34','35','36','36b','36c','37','38','39']
combin_list=combinations(a,no_cys)

x_a=x_a.reshape(len(x_a),16,21)


#get original data to determine enrichment of combinations
otu_table=pd.read_pickle('./seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')
otu_table=limit_by_nocys(otu_table,no_cys)

x_b=otu_table.loc[:,'One_Hot'].values.tolist()
for i in range(len(x_b)):
    x_b[i]=x_b[i].tolist()
x_b=np.array(x_b)
x_b=x_b.reshape(len(x_b),16,21)

#calculate the frequency and enrichment and MI of each cys combination in Dev+
combin_info=[]
for combin in combin_list:
    pa=frequency['C'][combin[0]]
    pb=frequency['C'][combin[1]]
    if no_cys==2:
        sites=[a.index(combin[0]),a.index(combin[1])]
        papbpcpd=pa*pb
    elif no_cys==3:
        pc=frequency['C'][combin[2]]
        sites=[a.index(combin[0]),a.index(combin[1]),a.index(combin[2])]
        papbpcpd=pa*pb*pc
    elif no_cys==4:
        pc=frequency['C'][combin[2]]
        p_d=frequency['C'][combin[3]]
        sites=[a.index(combin[0]),a.index(combin[1]),a.index(combin[2]),a.index(combin[3])]
        papbpcpd=pa*pb*pc*p_d

    pabcd,enrich=get_combo_freq(sites,x_a,x_b)
    mi=pabcd*np.log2(pabcd/papbpcpd)
    combin_info.append([combin,pabcd,mi,enrich,pabcd*enrich])

combin_info=pd.DataFrame(combin_info)
combin_info.columns=['Combination','Frequency','Mutual Information','Enrichment','Freqenrich']
high_freq_combin=combin_info.sort_values('Frequency',ascending=False)
high_mi_combin=combin_info.sort_values('Mutual Information',ascending=False)
high_enrich_combin=combin_info.sort_values('Enrichment',ascending=False)
high_enrich_combin=combin_info.sort_values('Freqenrich',ascending=False)






