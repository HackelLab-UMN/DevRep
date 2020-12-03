import multiprocessing
import numpy as np
from functools import partial
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn import preprocessing
from joblib import dump, load


def encode_paratope_OH(enc,paratope,axis):
    paratope=np.array(list(paratope))
    one_encode=enc.transform(paratope.reshape(-1,1))
    return one_encode.flatten()

def encode_paratope_ord(enc_ord,paratope,axis):
    paratope=np.array(list(paratope))
    ord_encode=enc_ord.transform(paratope.reshape(-1,1))
    return ord_encode.flatten()


def main():
    otu_table=pd.read_pickle('./name_scores_yield.pkl')

    otu_table_stop=otu_table[otu_table['Stop']==True] #isolate negative controls
    otu_table=otu_table[otu_table['Stop']==False] #remove negative controls
    otu_table.loc[:,'CC']=otu_table['CC'].replace(True,1)
    otu_table.loc[:,'CC']=otu_table['CC'].replace(False,0)

    # otu_table=otu_table.iloc[0:10000]

    #Create one-hot-encoded and ordinal encoded sequences
    AAlist=np.array(list("XACDEFGHIKLMNPQRSTVWY"))
    enc_OH=preprocessing.OneHotEncoder(sparse=False)
    enc_OH.fit(AAlist.reshape(-1,1))
    dump(enc_OH,'./one_hot_encoder.joblib')
    enc_ord=preprocessing.OrdinalEncoder()
    enc_ord.fit(AAlist.reshape(-1,1))
    dump(enc_ord,'./ordinal_encoder.joblib')


    encode_OH=partial(encode_paratope_OH,enc_OH)
    encode_ord=partial(encode_paratope_ord,enc_ord)

    otu_table.loc[:,'One_Hot']=otu_table['Paratope'].apply(encode_OH,axis=1)
    otu_table.loc[:,'Ordinal']=otu_table['Paratope'].apply(encode_ord,axis=1)

    #Combine IQ and SH yields
    yielddata=otu_table[otu_table[['IQ_Average','SH_Average']].notnull().any(axis=1)]
# 
    # #Isolate all10 and final test data
    check_list=[]
    for i in range(1,11):   #range(10):
        check_list.append('Sort'+str(i)+'_mean_score')
    testdata=yielddata[yielddata[check_list].isnull().any(axis=1)]  # find dot values with missing real assay scores
    all10=yielddata[yielddata[check_list].notnull().all(axis=1)]  # find dot values with missing real assay scores
    
    traindata=otu_table[~otu_table['DNA'].isin(yielddata['DNA'])] #remove test data + all10  from assay training data


    #box-cox transform of all10 yields
    iq_yields=all10['IQ_Average'].dropna().to_numpy()
    sh_yields=all10['SH_Average'].dropna().to_numpy()
    yields_comb=np.concatenate((iq_yields,sh_yields),axis=0)

    boxcox=load('./Yield_boxcox_fit.joblib')
    print(boxcox.lambdas_)
    del boxcox

    boxcox=preprocessing.PowerTransformer()
    boxcox.fit(yields_comb.reshape(-1,1))
    print('weraewflkawejflkwejfwljaefw')
    print(boxcox.lambdas_)
    # dump(boxcox,'./Yield_boxcox_fit.joblib')
    # boxcox=load('./Yield_boxcox_fit.joblib')

    for i in ["IQ","SH"]:
        for j in ["Average",'Trial1_Adj','Trial2_Adj','Trial3_Adj']:
            test_temp=testdata[i+'_'+j].dropna().to_numpy()
            test_nan_loc=testdata[i+'_'+j].notnull()
            testdata.loc[test_nan_loc,i+'_'+j+'_bc']=boxcox.transform(test_temp.reshape(-1,1))
            
            all10_temp=all10[i+'_'+j].dropna().to_numpy()
            all10_nan_loc=all10[i+'_'+j].notnull()
            all10.loc[all10_nan_loc,i+'_'+j+'_bc']=boxcox.transform(all10_temp.reshape(-1,1))
            
            stop_temp=otu_table_stop[i+'_'+j].dropna().to_numpy()
            stop_nan_loc=otu_table_stop[i+'_'+j].notnull()
            otu_table_stop.loc[stop_nan_loc,i+'_'+j+'_bc']=boxcox.transform(stop_temp.reshape(-1,1))

        j=["Average",'Trial1_Adj','Trial2_Adj','Trial3_Adj']
        testdata.loc[:,i+'_Average_bc_std']=testdata[[i+'_'+j[1]+'_bc',i+'_'+j[2]+'_bc',i+'_'+j[3]+'_bc']].std(axis=1,skipna=False)
        all10.loc[:,i+'_Average_bc_std']=all10[[i+'_'+j[1]+'_bc',i+'_'+j[2]+'_bc',i+'_'+j[3]+'_bc']].std(axis=1,skipna=False)
        otu_table_stop.loc[:,i+'_Average_bc_std']=otu_table_stop[[i+'_'+j[1]+'_bc',i+'_'+j[2]+'_bc',i+'_'+j[3]+'_bc']].std(axis=1,skipna=False)


    #scale/normalize b-lac trials based upon training distribution 
    for i in [9,10]:
        not_null=traindata[traindata['Sort'+str(i)+'_mean_count'].notnull()]
        temp=not_null['Sort'+str(i)+'_mean_score'].values
        scale1=preprocessing.QuantileTransformer(output_distribution='normal')
        temp1=scale1.fit_transform(temp.reshape(-1,1))
        dump(scale1,'./Sort'+str(i)+'_quantileTransformer.joblib')
        scale2=preprocessing.MinMaxScaler(feature_range=(0,1))
        scale2.fit(temp1.reshape(-1,1))
        dump(scale2,'./Sort'+str(i)+'_minmaxscaler.joblib')

        for j in ['1','2','3','mean']:
            train_temp=traindata['Sort'+str(i)+'_'+j+'_score'].values
            traindata.loc[:,'Sort'+str(i)+'_'+j+'_score']=scale2.transform(scale1.transform(train_temp.reshape(-1,1)).reshape(-1,1))
            all10_temp=all10['Sort'+str(i)+'_'+j+'_score'].values
            all10.loc[:,'Sort'+str(i)+'_'+j+'_score']=scale2.transform(scale1.transform(all10_temp.reshape(-1,1)).reshape(-1,1))
            test_temp=testdata['Sort'+str(i)+'_'+j+'_score'].values
            testdata.loc[:,'Sort'+str(i)+'_'+j+'_score']=scale2.transform(scale1.transform(test_temp.reshape(-1,1)).reshape(-1,1))
            stop_temp=otu_table_stop['Sort'+str(i)+'_'+j+'_score'].values
            otu_table_stop.loc[:,'Sort'+str(i)+'_'+j+'_score']=scale2.transform(scale1.transform(stop_temp.reshape(-1,1)).reshape(-1,1))

        traindata.loc[:,'Sort'+str(i)+'_std_score']=traindata[['Sort'+str(i)+'_1_score','Sort'+str(i)+'_2_score','Sort'+str(i)+'_3_score']].std(axis=1,skipna=False)
        all10.loc[:,'Sort'+str(i)+'_std_score']=all10[['Sort'+str(i)+'_1_score','Sort'+str(i)+'_2_score','Sort'+str(i)+'_3_score']].std(axis=1,skipna=False)
        otu_table_stop.loc[:,'Sort'+str(i)+'_std_score']=otu_table_stop[['Sort'+str(i)+'_1_score','Sort'+str(i)+'_2_score','Sort'+str(i)+'_3_score']].std(axis=1,skipna=False)





    testdata.to_csv('./testdata.csv')
    all10.to_csv('./traindata.csv')
    otu_table_stop.to_csv('./stopdata.csv')

    traindata.to_pickle('./seq_to_assay_training_data.pkl')
    all10.to_pickle('./assay_to_dot_training_data.pkl')
    testdata.to_pickle('./seq_to_dot_test_data.pkl')
    otu_table_stop.to_pickle('./stop_data.pkl')










if __name__ == '__main__':
    main()
        
