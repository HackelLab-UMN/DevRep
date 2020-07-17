import submodels_module as modelbank
import numpy as np
from itertools import combinations
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import load_format_data 
import pandas as pd


class results:
    def __init__(self):
        self.model_list=None
        self.compare_test=False

    def get_loss_list(self,model_list):
        model_loss_list,model_loss_std_list,model_name_list=[],[],[]
        if self.compare_test:
            exp_var=self.get_variance(model_list[0].testing_df)
        else: 
            exp_var=self.get_variance(model_list[0].training_df)

        for model in model_list:
            # model.get_best_trial()
            model_name_list.append(model.model_name)
            if self.compare_test:
                model_loss_list.append(model.model_stats['test_avg_loss'])
                model_loss_std_list.append(model.model_stats['test_std_loss'])
            else:
                model_loss_list.append(model.model_stats['cv_avg_loss'])
                model_loss_std_list.append(model.model_stats['cv_std_loss'])
        return model_loss_list,model_loss_std_list,model_name_list,exp_var 


    def save_loss_list(self,model_name_list,model_loss_list,model_loss_std_list):
        df=pd.DataFrame(np.transpose(np.array([model_name_list,model_loss_list,model_loss_std_list])))
        df.to_csv('./aty_best_arch_cv.csv')
    
    def plot_distribution(self,set_name,model_loss_list):
        control_model_loss,_=self.get_control()
        seq_model=self.get_best_seq_to_yield_simple()
        seq_model_cv_loss=seq_model.model_stats['cv_avg_loss']

        fig,ax = plt.subplots(1,1,figsize=[4.25,1.5],dpi=300)
        ax.hist(model_loss_list,bins=20,color='black',orientation='horizontal')
        ax.set_ylabel('CV Loss (MSE)',fontsize=6)
        ax.set_xlabel('# of Assay Combin.',fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.axhline(y=control_model_loss,label='Strain Only Control',color='red')
        ax.axhline(y=seq_model_cv_loss,label='OH Sequence Model',color='blue')
        ax.legend(fontsize=6,loc='center',framealpha=1)
        plt.tight_layout()
        fig.savefig('./'+set_name+'.png')

    def plot_bar(self,model_list,set_name):
        model_loss_list,model_loss_std_list,model_name_list,exp_var=self.get_loss_list(model_list)
        control_model_loss,control_model_loss_std=self.get_control()

        fig,ax = plt.subplots(1,1,figsize=[6,3],dpi=300)
        ax.bar(list(range(len(model_list)+2)),[exp_var,control_model_loss]+model_loss_list,tick_label=['exp_variance','control']+model_name_list)
        ax.set_ylabel('Model Loss')
        ax.tick_params(axis='both', which='major', labelsize=6)
        plt.tight_layout()
        fig.savefig('./'+set_name+'.png')

    def get_best_model(self,model_list,save=False,plot=False):
        model_loss_list,model_loss_std_list,model_name_list,exp_var=self.get_loss_list(model_list)
        if save:
            self.save_loss_list(model_name_list,model_loss_list,model_loss_std_list)
        if plot:
        	self.plot_distribution(plot,model_loss_list)
        best_index=np.argmin(np.array(model_loss_list))
        return model_list[best_index]

    def assay_to_yield_best_arch(self):
        self.compare_test=False
        self.get_control=self.get_assay_control

        a=[1,2,3,4,5,6,7,8,9,10]
        combin_list=[]
        for i in range(1,11):
            combin_list_temp=combinations(a,i)
            for j in combin_list_temp:
                combin_list.append(j)

        b_models=['ridge','forest','svm','fnn']
        # b_models=b_models[0:3]

        combin_list=combin_list[0:10]
        best_model_per_combin=[]
        for combin in combin_list:
            model_list=[]
            for arch in b_models:
                model_list.append(modelbank.assay_to_yield_model(combin,arch,1))
            best_model_per_combin.append(self.get_best_model(model_list))

        best_model=self.get_best_model(best_model_per_combin,save=True,plot='assay_to_yield_best_arch')
        print(best_model.model_name)

        # self.plot_distribution(best_model_per_combin,'assay_to_yield_best_arch')

    def get_best_seq_to_yield_simple(self):
        self.compare_test=False
        self.get_control=self.get_assay_control

        b_models=['ridge','forest','svm','fnn']
        model_list=[]
        for arch in b_models:
            model_list.append(modelbank.seq_to_yield_model(arch,1))

        best_model=self.get_best_model(model_list)

        return best_model

    def set_model_list(self,mode):
        mode_dict={
        'aty_best_arch':self.assay_to_yield_best_arch,
        }
        self.make_model_list=mode_dict[mode]
        self.make_model_list()

    def get_variance(self,df):
        exploded_df,_,_=load_format_data.explode_yield(df)
        return np.average(np.square(np.array(exploded_df['y_std'])))

    def get_assay_control(self):
        control_model=modelbank.control_to_yield_model('ridge',1)
        if self.compare_test:
            return control_model.model_stats['test_avg_loss'],control_model.model_stats['test_std_loss']
        else:
            return control_model.model_stats['cv_avg_loss'],control_model.model_stats['cv_std_loss']





def main():
    a=results()
    a.set_model_list('aty_best_arch')
    # a.set_model_list('aty_simple')


if __name__ == '__main__':
    main()