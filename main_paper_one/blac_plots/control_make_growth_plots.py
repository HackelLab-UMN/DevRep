import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():


    for trial in ['iq1','iq2','iq3','sh1','sh2','sh3']:

        iq=pd.read_csv('./'+trial+'.csv',header=None)

        plate_cols=np.array(list(range(0,96,12)))

        if '1' in trial:
            blank_cols=[0]
        else:
            blank_cols=[0,1]
        blanks=[]
        for i in blank_cols:
            blanks.append(iq.iloc[:,plate_cols+i])
        blank=pd.concat(blanks).mean().mean()
        
        iq_blanked=iq-blank

        samples=[]
        if '1' in trial:
            sample_cols=[[1,1],[2,2],[6,7,8,9,10,11]]
        else:
            sample_cols=[[2,3],[4,5],[6,7,8,9,10,11]]
            
        for j in sample_cols:
            to_be_averaged=[]
            for k in j:
                to_be_averaged.append(iq_blanked.iloc[:,plate_cols+k].to_numpy())
            joined=np.array(to_be_averaged)
            averaged=np.mean(joined,axis=0)
            samples.append(averaged)

        lib=samples[2]
        if 'iq' in trial:
            conc_used=[7,5,4,2]
            conc_lables=['0 $\mu$g/mL', '100 $\mu$g/mL' , '333 $\mu$g/mL' , '1000 $\mu$g/mL']
        else:
            conc_used=[7,6,4,1]
            conc_lables=['0 $\mu$g/mL', '100 $\mu$g/mL' , '666 $\mu$g/mL' , '6666 $\mu$g/mL']




        time=np.array(list(range(0,len(iq),1)))*5

        for i in [0,1]:
            fig, ax = plt.subplots(1,1,figsize=[2,2],dpi=300)
            for conc in conc_used:

                iq_a=samples[i][:,conc]
                ax.plot(time,iq_a)
                # saturated=np.where(iq_a>0.35-blank)[0]
                # # print(saturated)
                # first_stop=len(iq_a)
                # if len(saturated)>0:
                #     first_stop=saturated[0]+1
                # print(first_stop*5)
                # ax.plot(time[:first_stop],iq_a[:first_stop])
                # ax.set_ylim([0,0.35-blank])
            ax.set_ylim([0,0.8])

            ax.set_xlim([0,1500])
            if 'iq' in trial:
                ax.set_xlim([0,500])
                ax.set_ylim([0,0.4])

            ax.set_xlabel('Time (min)',fontsize=6)
            ax.set_ylabel('OD600 (cell density)',fontsize=6)
            # if '3' in trial:
            #     ax.legend(conc_lables,title='[Amp]',title_fontsize=6,fontsize=6)
            ax.axhline(0.35-blank,color='black')
            ax.tick_params(axis='both', which='both', labelsize=6)
            fig.tight_layout()
            fig.savefig('./'+trial+'_control'+str(i)+'.png')
            plt.close()


if __name__=='__main__':
    main()