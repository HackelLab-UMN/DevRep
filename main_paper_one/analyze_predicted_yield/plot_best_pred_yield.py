import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

def main():
    df=pd.read_pickle('./seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')

    #df=pd.read_pickle('./seq_to_assay_train_1,8,10_assays1,8,10_yield_forest_1_0.pkl') #plot from assay predictions
    # df=pd.read_pickle('./otherc_plus_seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl') #plot for seq w/ C not at 7,12
    # df=pd.read_pickle('./otherc_less_seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl') #plot for seq w/ only C at 7,12
    
    iq_pred=df['IQ_Average_bc'].to_list()
    sh_pred=df['SH_Average_bc'].to_list()
    fig,ax=plt.subplots(1,1)

    s=sns.jointplot(x=iq_pred,y=sh_pred,kind='kde',height=1.5,space=0,ratio=5,joint_kws={'shade_lowest':False},xlim=[-1.5,1],ylim=[-0.25,2]) #blue
    # s=sns.jointplot(x=iq_pred,y=sh_pred,kind='kde',height=1.5,space=0,ratio=5,joint_kws={'shade_lowest':False},xlim=[-1.5,1],ylim=[-0.25,2],color='red')
    
    #draw lines at cutoff 
    # s.ax_joint.axvline(x=0,color='red')
    # s.ax_joint.axhline(y=0.75,color='red')
    
    s.ax_joint.set_xticks([-1,0,1])
    s.ax_joint.set_yticks([0,1,2]) 
    s.ax_joint.set_xticklabels(['-1','0','1'],fontsize=6)
    s.ax_joint.set_yticklabels(['0','1','2'],fontsize=6)
    s.ax_joint.set_xlabel('Predicted $Yield_{I^q}$',fontsize=6)
    s.ax_joint.set_ylabel('Predicted $Yield_{SH}$',fontsize=6)
    plt.savefig('./assays1,8,10_yield_predictions.png',dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
        
