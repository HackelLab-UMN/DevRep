import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from joblib import dump, load


def load_datasets():
    library=pd.read_pickle('../plot_preditions/seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')

    transformer=load('../make_datasets/Yield_boxcox_fit.joblib')

    _in=library['IQ_Average_bc'].to_numpy()
    library['IQ_Average']=transformer.inverse_transform(_in.reshape(-1,1))

    _in=library['SH_Average_bc'].to_numpy()
    library['SH_Average']=transformer.inverse_transform(_in.reshape(-1,1))
    return library



def main():

    library=load_datasets()

    fig, ax=plt.subplots(1,1,figsize=[1.5,1.2],dpi=300)

    iq_data=library[library['IQ_Average'].notnull()]
    iq_data=iq_data.rename(columns={'IQ_Average': 'Yield (mg/L)'})
    iq_data['Cell Type']='Strain $I^q$\nN='+str(len(iq_data))

    sh_data=library[library['SH_Average'].notnull()]
    sh_data=sh_data.rename(columns={'SH_Average': 'Yield (mg/L)'})
    sh_data['Cell Type']='Strain $SH$\nN='+str(len(sh_data))

    data=pd.concat([iq_data,sh_data])
    violin_parts=sns.violinplot(data=data,x="Cell Type",y="Yield (mg/L)",ax=ax,color='black',inner=None,scale='width')
    ax.set_ylabel('Predicted Yield (mg/L)',fontsize=6)
    ax.set_ylim([0,30])
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel('',visible=False)
    fig.tight_layout()
    fig.savefig('Fig1c_predicted.png')
    plt.close()



if __name__ == '__main__':
    main()
        
