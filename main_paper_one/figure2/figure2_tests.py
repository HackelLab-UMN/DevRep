import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import load
from scipy.stats import ttest_ind_from_stats as tt
from scipy.stats import mannwhitneyu as mw
from functools import partial

sorts=["$Protease_{PK37}$","$Protease_{Urea}$","$Protease_{Guan}$","$Protease_{PK55}$","$Protease_{TL55}$","$Protease_{TL75}$","$GFP_{I^q}$","$GFP_{SH}$",r'$\beta$'+"-$lactamase_{I^q}$",r'$\beta$'+"-$lactamase_{SH}$",'$Yield_{I^q}$','$Yield_{SH}$']


df_a=pd.read_pickle('../make_datasets/seq_to_assay_training_data.pkl')
df_b=pd.read_pickle('../make_datasets/assay_to_dot_training_data.pkl')
df_c=pd.read_pickle('../make_datasets/seq_to_dot_test_data.pkl')

library_df=pd.concat([df_a,df_b,df_c],sort=False)

stop_df=pd.read_pickle('../make_datasets/stop_data.pkl')



#caluclate mean variances
# lib_and_stop=pd.concat([library_df,stop_df],sort=False)
# lib_and_stop=lib_and_stop[~lib_and_stop['SH_Average'].isna()]
# lib_and_stop['SH_Average_std']=lib_and_stop[['SH_Trial1_Adj','SH_Trial2_Adj','SH_Trial3_Adj']].std(axis=1)
# stdev=lib_and_stop['SH_Average_std'].tolist()
# var=np.square(np.array(stdev))
# print(np.sqrt(np.mean(var)))





#compare cc+ / cc- 
ccp=library_df[library_df['CC']==True]
ccn=library_df[library_df['CC']==False]

# for i in range(1,11):
for i in [7]:
    ccp_i=ccp[~ccp['Sort'+str(i)+'_mean_score'].isna()]
    ccp_list=ccp_i['Sort'+str(i)+'_mean_score'].tolist()
    ccn_i=ccn[~ccn['Sort'+str(i)+'_mean_score'].isna()]
    ccn_list=ccn_i['Sort'+str(i)+'_mean_score'].tolist()

    t,p=mw(ccp_list,ccn_list,alternative='greater')
    print(t,p)
    print(len(ccn_list),np.median(ccn_list))
    print(len(ccp_list),np.median(ccp_list))

for i in ['IQ_Average','IQ_Average_bc','SH_Average','SH_Average_bc']:
    ccp_i=ccp[~ccp[i].isna()]
    ccp_list=ccp_i[i].tolist()
    print(len(ccp_list),np.median(ccp_list))

    ccn_i=ccn[~ccn[i].isna()]
    ccn_list=ccn_i[i].tolist()
    print(len(ccn_list),np.median(ccn_list))


    t,p=mw(ccp_list,ccn_list,alternative='two-sided')
    print(t,p)




#load gar data
gar_df=pd.read_pickle('./gar_scores.pkl')
for i in [9,10]:
    trans_1=load('../make_datasets/Sort'+str(i)+'_quantileTransformer.joblib')
    temp_a=trans_1.transform(gar_df['Sort'+str(i)+'_mean_score'].to_numpy().reshape(-1,1))[0][0]
    trans_2=load('../make_datasets/Sort'+str(i)+'_minmaxscaler.joblib')
    gar_df['Sort'+str(i)+'_mean_score']=trans_2.transform(np.array(temp_a).reshape(-1,1))[0][0]

    temp_b1=trans_1.transform(gar_df['Sort'+str(i)+'_1_score'].to_numpy().reshape(-1,1))[0][0]
    temp_b2=trans_1.transform(gar_df['Sort'+str(i)+'_2_score'].to_numpy().reshape(-1,1))[0][0]
    temp_b3=trans_1.transform(gar_df['Sort'+str(i)+'_3_score'].to_numpy().reshape(-1,1))[0][0]

    temp_b11=trans_2.transform(np.array(temp_b1).reshape(-1,1))[0][0]
    temp_b22=trans_2.transform(np.array(temp_b2).reshape(-1,1))[0][0]
    temp_b33=trans_2.transform(np.array(temp_b3).reshape(-1,1))[0][0]

    gar_df['Sort'+str(i)+'_std_score']=np.std([temp_b11,temp_b22,temp_b33])

gar_yield={'IQ_yield':7.575,'IQ_std':2.996,'SH_yield':16.82,'SH_std':13.26} #from IQ and SH dots.xlsx on msi round3/plateseq 
gar_transyield={'IQ_yield':0.74422,'IQ_std':0.4558,'SH_yield':1.194,'SH_std':0.611} #from IQ and SH dots.xlsx on msi round3/plateseq 


## test frac of stop < GaR
def t_test(sort,row):
    row_mean=row['Sort'+str(i)+'_mean_score']
    row_std=row['Sort'+str(i)+'_std_score']

    gar_mean=gar_df['Sort'+str(i)+'_mean_score']
    gar_std=gar_df['Sort'+str(i)+'_std_score']
    
    t,p=tt(row_mean,row_std,3,gar_mean,gar_std,3)

    #note tt is two-sided, but we want one-sided, so divide p/2
    p=p/2

    #alpha=0.05
    # print(p)
    # print(t)
    if p[0]<0.05 and t[0]<0: #change to > for b_lac
        return 1
    else:
        return 0

## for each assay
# stop_df=stop_df.iloc[0:100]
# for i in range(1,11):
#     stop_i=stop_df[~stop_df['Sort'+str(i)+'_mean_score'].isna()]
#     t_test_i=partial(t_test,i)
#     stop_i['less_than_gar']=stop_i.apply(t_test_i,axis=1)
#     print(100*sum(stop_i['less_than_gar'])/len(stop_i))
#     print(len(stop_i))


def yield_t_test(strain,transformed,row):
    if transformed:
        row_mean=row[strain+'_Average_bc']
        row_std=row[strain+'_Average_bc_std']

        gar_mean=gar_transyield[strain+'_yield']
        gar_std=gar_transyield[strain+'_std']

    else:
        row_mean=row[strain+'_Average']
        row_std=np.std(row[[strain+'_Trial1_Adj',strain+'_Trial2_Adj',strain+'_Trial3_Adj']])

        gar_mean=gar_yield[strain+'_yield']
        gar_std=gar_yield[strain+'_std']

    t,p=tt(row_mean,row_std,3,gar_mean,gar_std,3)
    p=p/2
    # print(p)
    # print(t)
    if p<0.05 and t<0:
        return 1
    else:
        return 0



# for i in ['IQ','SH']:
#     stop_i=stop_df[~stop_df[i+'_Average'].isna()]

#     yield_test_i=partial(yield_t_test,i,False)
#     stop_i['less_than_gar']=stop_i.apply(yield_test_i,axis=1)
#     print(100*sum(stop_i['less_than_gar'])/len(stop_i))
#     print(len(stop_i))

#     yield_test_i=partial(yield_t_test,i,True)
#     stop_i['less_than_gar']=stop_i.apply(yield_test_i,axis=1)
#     print(100*sum(stop_i['less_than_gar'])/len(stop_i))
#     print(len(stop_i))


