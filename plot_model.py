import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class model_plot():
    def colorbar(self,mappable):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        last_axes = plt.gca()
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.ax.tick_params(labelsize=4)
        plt.sca(last_axes)
        return cbar


    def __init__(self):
        self.fig, self.ax = plt.subplots(2,1,figsize=[1.25,2.5],dpi=300)

class x_to_yield_plot(model_plot):
    def add_axis(self):
        std_list=[0.591,0.603]
        for i in range(2):
            x=[-2,2]
            std=std_list[i]
            x_high=[y+std for y in x]
            x_low=[y-std for y in x]
            # self.ax[i].plot(x_low,x_low,'g--')
            # self.ax[i].plot(x_high,x_high,'g--')
            self.ax[i].fill_between(x,x_low,x_high,color='purple',alpha=0.25,label='Experimental Variance')
            # self.ax[i].plot(x,x,'r--')
            self.ax[i].tick_params(axis='both', which='major', labelsize=6)
            self.ax[i].set_xlabel("Predicted Yield'",fontsize=6)
            self.ax[i].set_ylabel("True Yield'",fontsize=6)
            self.ax[i].set_xlim([-2,2])
            self.ax[i].set_xticks([-2,-1,0,1,2])
            self.ax[i].set_yticks([-2,-1,0,1,2])
            self.ax[i].set_yticklabels(['-2','','0','','2'])
            self.ax[i].set_xticklabels(['-2','','0','','2'])
            # self.ax[i].set_xticks([])
            # self.ax[i].set_yticks([])
            self.ax[i].set_ylim([-2,2])
            # self.ax[i].legend(fontsize=6,bbox_to_anchor=(0.5, 1.25), loc='center',framealpha=1)
            self.ax[i].set_aspect('equal')


    def __init__(self,model):
        super().__init__()
        iq_data,sh_data=[[],[]],[[],[]]
        for pred,true,cell in zip(model.plotpairs_cv[1],model.plotpairs_cv[0],model.plotpairs_cv[2]):
            if cell[0]==1:
                iq_data[0].append(pred)
                iq_data[1].append(true)
            else:
                sh_data[0].append(pred)
                sh_data[1].append(true)

        # self.ax[0].scatter(iq_data[0], iq_data[1],s=16,marker='.',color='black',alpha=0.1,label='LysY/$I^q$')
        # self.ax[0].scatter(sh_data[0], sh_data[1],s=16,marker='.',color='black',alpha=0.1,label='SHuffle')

        self.ax[0].scatter(iq_data[0], iq_data[1],s=16,marker='.',color='maroon',alpha=0.25,label='LysY/$I^q$')
        self.ax[0].scatter(sh_data[0], sh_data[1],s=16,marker='.',color='orange',alpha=0.25,label='SHuffle')

        # self.ax[0].set_title('CV_MSE='+str(round(model.model_stats['cv_avg_loss'],3))+r'$\pm$'+str(round(model.model_stats['cv_std_loss'],3)))

        
        iq_data,sh_data=[[],[]],[[],[]]
        for pred,true,cell in zip(model.plotpairs_test[1],model.plotpairs_test[0],model.plotpairs_test[2]):
            if cell[0]==1:
                iq_data[0].append(pred)
                iq_data[1].append(true)
            else:
                sh_data[0].append(pred)
                sh_data[1].append(true)
        # self.ax[1].scatter(iq_data[0], iq_data[1],s=16,marker='.',color='black',alpha=0.1,label='LysY/$I^q$')
        # self.ax[1].scatter(sh_data[0], sh_data[1],s=16,marker='.',color='black',alpha=0.1,label='SHuffle')

        self.ax[1].scatter(iq_data[0], iq_data[1],s=16,marker='.',color='maroon',alpha=0.25,label='LysY/$I^q$')
        self.ax[1].scatter(sh_data[0], sh_data[1],s=16,marker='.',color='orange',alpha=0.25,label='SHuffle')

        # self.ax[1].set_title('Test_MSE='+str(round(model.model_stats['test_avg_loss'],3))+r'$\pm$'+str(round(model.model_stats['test_std_loss'],3)))

        self.add_axis()
        self.fig.tight_layout()

class x_to_assay_plot(model_plot):
    def add_axis(self):
        for i in range(2):
            self.ax[i].set_xlabel('Predicted Assay Score',fontsize=6)
            self.ax[i].set_ylabel('True Assay Score',fontsize=6)
            self.ax[i].set_xlim([0,1])
            self.ax[i].set_ylim([0,1])
            self.ax[i].set_xticks([0,0.5,1])
            self.ax[i].set_yticks([0,0.5,1])
            self.ax[i].set_yticklabels(['0','','1'])
            self.ax[i].set_xticklabels(['0','','1'])
            self.ax[i].set_aspect('equal')
            self.ax[i].tick_params(axis='both', which='major', labelsize=6)



    def __init__(self,model):
        super().__init__()
        _,_,_,img1=self.ax[0].hist2d(model.plotpairs_cv[1], model.plotpairs_cv[0], bins=(np.linspace(0,1,21), np.linspace(0,1,21)), cmap=plt.cm.jet,cmin=1)
        self.colorbar(img1)
        _,_,_,img2=self.ax[1].hist2d(model.plotpairs_test[1], model.plotpairs_test[0], bins=(np.linspace(0,1,21), np.linspace(0,1,21)), cmap=plt.cm.jet,cmin=1)
        self.colorbar(img2)

        self.add_axis()
        self.fig.tight_layout()