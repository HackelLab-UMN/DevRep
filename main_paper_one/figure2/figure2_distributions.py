import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import load

sorts=["$Protease_{PK37}$","$Protease_{Urea}$","$Protease_{Gdn}$","$Protease_{PK55}$","$Protease_{TL55}$","$Protease_{TL75}$","$GFP_{I^q}$","$GFP_{SH}$",r'$\beta$'+"-$lactamase_{I^q}$",r'$\beta$'+"-$lactamase_{SH}$",'$Yield_{I^q}$','$Yield_{SH}$']


df_a=pd.read_pickle('../make_datasets/seq_to_assay_training_data.pkl')
df_b=pd.read_pickle('../make_datasets/assay_to_dot_training_data.pkl')
df_c=pd.read_pickle('../make_datasets/seq_to_dot_test_data.pkl')

library_df=pd.concat([df_a,df_b,df_c],sort=False)

stop_df=pd.read_pickle('../make_datasets/stop_data.pkl')

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
for i in range(1,11):
    gar_df['Sort'+str(i)+'_se_score']=gar_df['Sort'+str(i)+'_std_score']/np.sqrt(3)

gar_yield={'IQ_yield':7.57,'IQ_se':1.06*np.sqrt(3),'SH_yield':16.8,'SH_se':4.69*np.sqrt(3)} #from IQ and SH dots.xlsx on msi round3/plateseq 

fig,ax=plt.subplots(2,6,figsize=[3.5,2],dpi=300,sharey='row')
for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
    if i<7:
        ax_cur=ax[0,i-1]
    else:
        ax_cur=ax[1,i-7]
    if i<11:
        df_col='Sort'+str(i)+'_mean_score'
    elif i==11:
        df_col='IQ_Average'
    else:
        df_col='SH_Average'
    ccp_temp=library_df[library_df['CC']==True][df_col].to_numpy()
    ccp_data=ccp_temp[~np.isnan(ccp_temp)]
    ccpdf=pd.DataFrame(ccp_data,columns=['Assay Score'])
    ccpdf['Population']='CC+'

    ccn_temp=library_df[library_df['CC']==False][df_col].to_numpy()
    ccn_data=ccn_temp[~np.isnan(ccn_temp)]
    ccndf=pd.DataFrame(ccn_data,columns=['Assay Score'])
    ccndf['Population']='CC$-$'

    # lib_temp=library_df['Sort'+str(i)+'_mean_score'].to_numpy()
    # lib_data=lib_temp[~np.isnan(lib_temp)]
    # ldf=pd.DataFrame(lib_data,columns=['Assay Score'])
    # ldf['Population']='Library'

    stop_temp=stop_df[df_col].to_numpy()
    stop_data=stop_temp[~np.isnan(stop_temp)]
    sdf=pd.DataFrame(stop_data,columns=['Assay Score'])
    sdf['Population']='Stop'


    if i<11:
        gar_score=gar_df[df_col].to_numpy()
        gar_se=gar_df['Sort'+str(i)+'_std_score'].to_numpy()
    elif i==11:
        gar_score=gar_yield['IQ_yield']
        gar_se=gar_yield['IQ_se']
    else:
        gar_score=gar_yield['SH_yield']
        gar_se=gar_yield['SH_se']

    data=pd.concat([ccndf,ccpdf,sdf])
    violin_parts=sns.violinplot(data=data,y="Population",x="Assay Score",ax=ax_cur,color='black',inner='quartile',scale='width',orient='h')
    for l in violin_parts.lines:
        l.set_color('white')
        l.set_linewidth(0.5)

    ax_cur.errorbar(x=gar_score,y=3,xerr=gar_se,marker='o',color='black',mfc='white',ms=2)
    ax_cur.set_yticklabels(['CC$-$','CC+','Stop','GaR'])
    # ax_cur.yaxis.set_tick_params(pad=16)
    ax_cur.set_yticks([0,1,2,3])
    ax_cur.tick_params(axis='y',length=0)
    # ax_cur.set_ylim([3.5,-0.5])
    ax_cur.set_ylim([-0.5,3.5])
    ax_cur.set_xlabel('',visible=False)
    ax_cur.tick_params(axis='both', which='major', labelsize=6)
    ax_cur.set_ylabel('',visible=False)
    ax_cur.set_title(sorts[i-1],fontsize=5)

    if i<11:
        ax_cur.set_xlim([0,1])
        ax_cur.set_xticks([0,1])
    else:
        r,l=ax_cur.get_xlim()
        ax_cur.set_xlim([0,l])




# violin_parts=ax.violinplot(data,positions=[0,1,2],showmedians=True,showextrema=False,points=100,widths=.9)
# for pc in violin_parts['bodies']:
#     pc.set_color('k')
# violin_parts['cmedians'].set_color('r')
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(sorts[0,7,9])
# ax.set_ylim([0,1])
fig.tight_layout(pad=0.01)
fig.savefig('./Best_score.png')
plt.close()

