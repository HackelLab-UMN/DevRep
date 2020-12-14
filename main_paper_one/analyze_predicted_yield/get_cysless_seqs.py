import pandas as pd 

#divide up sequences based upon if they have additional cys outside 7,12

df=pd.read_pickle('seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')
df=df[df['CC']==True]

keep=[]
for index,seq in df.iterrows():
    num_cystines_per_seq=0
    otherc=False
    for j in [1,2,3,4,5,6,8,9,10,11,12,13,14,15]: #exclude spot 7(0)+12(7)
        if seq['Paratope'][j] == 'C':
            otherc=True

    keep.append(otherc)

df['otherc']=keep

df_c=df[df['otherc']==True]


df_c.to_pickle('otherc_plus_seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')
pool='cplus_seq_and_assay'
df_c=df_c[df_c['IQ_Average_bc']>0]
df_c=df_c[df_c['SH_Average_bc']>0.75]
df_c.to_pickle('./'+pool+'_best_sequences.pkl')

df_nc=df[df['otherc']==False]
df_nc.to_pickle('otherc_less_seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')
pool='cless_seq_and_assay'
df_nc=df_nc[df_nc['IQ_Average_bc']>0]
df_nc=df_nc[df_nc['SH_Average_bc']>0.75]
df_nc.to_pickle('./'+pool+'_best_sequences.pkl')

