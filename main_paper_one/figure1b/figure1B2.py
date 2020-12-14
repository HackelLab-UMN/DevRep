import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

def load_datasets():
    library=pd.read_pickle('../make_datasets/seq_to_assay_training_data.pkl')
    # test=pd.read_pickle('./seq_to_dot_test_data.pkl')
    # all10=pd.read_pickle('./assay_to_dot_training_data.pkl')

    # library=pd.concat([test,all10])

    return library



def main():

    library=load_datasets()

    fig, ax=plt.subplots(1,1,figsize=[1.75,1.3],dpi=300)

    sort_one=library[library['Sort1_mean_score'].notnull()]
    sort_one=sort_one.rename(columns={'Sort1_mean_score': 'Assay Score'})
    sort_one['Sort']='$Protease$\nN='+str(len(sort_one))

    sort_eight=library[library['Sort8_mean_score'].notnull()]
    sort_eight=sort_eight.rename(columns={'Sort8_mean_score': 'Assay Score'})
    sort_eight['Sort']='$sGFP$\nN='+str(len(sort_eight))

    sort_ten=library[library['Sort10_mean_score'].notnull()]
    sort_ten=sort_ten.rename(columns={'Sort10_mean_score': 'Assay Score'})
    sort_ten['Sort']=r'$\beta$'+"-lactamase\nN="+str(len(sort_ten))

    data=pd.concat([sort_one,sort_eight,sort_ten])
    violin_parts=sns.violinplot(data=data,x="Sort",y="Assay Score",ax=ax,color='black',inner=None,scale='width')
    ax.set_ylabel('Assay Score',fontsize=6)
    ax.set_ylim([0,1])
    ax.set_yticks([0,1])
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.set_ylabel('',visible=False)
    ax.set_xlabel('',visible=False)
    ax.tick_params(axis='x', which='major', length=0)
    fig.tight_layout()
    fig.savefig('Fig1B2.png')
    plt.close()



if __name__ == '__main__':
    main()
        