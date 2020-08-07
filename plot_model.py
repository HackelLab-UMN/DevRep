import numpy as np
import matplotlib as mpl
mpl.use('Agg')
## This sets a non-interactive backend gigving image outputs of constructed graphs
import matplotlib.pyplot as plt
## Module inteded fot progammic plot generation

class model_plot():
## A new object of type model_plot to create a blank uncharted plot
    def colorbar(self,mappable):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        last_axes = plt.gca()
        ## last_axes is the current axes of the given figure, this is the
        ## axes of the figure instantiated once the object was created. 
        ax = mappable.axes
        ## ax on the other hand is the axes of the figure passed in as input to
        ## this function
        fig = ax.figure
        ## Figure is a figure object of the previously mentioned ax variable
        divider = make_axes_locatable(ax)
        
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ## adds another axes to the right of the main axes to the right
        ## with a thickness of 5% of the axes and with distance between the
        ## new axes and orginal axes be 0.05 inch. Notice this is cax meaning that
        ## this will be the axes which will become a colorbar.
        cbar = fig.colorbar(mappable, cax=cax)
        ## Creates a colorbar for a given image, with mappable being the image
        ## where it is to be added and cax input the axes where the colorbar will
        ## be drawn. cbar is an Colorbar object for use in contour plots
        cbar.ax.tick_params(labelsize=4)
        ## This sets the ticks labels font size to a certain value
        plt.sca(last_axes)
        ## the above command sets the current axis of the figure to the variable last_axes defined above
        ##This function returns a colorbar to a given function. 
        return cbar

    def __init__(self):
        ## This function creates two attributes for each plot_model instance:
        ## fig is an object of type Figure while ax is a n-dimensional array
        ## The figure that is created contains two subplots
        self.fig, self.ax = plt.subplots(2,1,figsize=[1.6,3.2],dpi=300)


class x_to_yield_plot(model_plot):
## Class x_to_yield_plot inherits from the model_plot class shown above.
    
    def add_axis(self):
        std_list=[0.591,0.603]
        for i in range(2):
            x=[-2,2]
            std=std_list[i]
            x_high=[y+std for y in x]
            x_low=[y-std for y in x]
            # self.ax[i].plot(x_low,x_low,'g--')
            # self.ax[i].plot(x_high,x_high,'g--')
            self.ax[i].fill_between(x,x_low,x_high,color='green',alpha=0.25,label='Experimental Variance')
            ## Depending on the standard deviation, a high curve and a low curve is defined and the areas in between the 
            ## are shaded in green, the x and y range of the axis is [-2,2]. The top subplot has the std of 0.591 while the 
            ## bottom one has a std of 0.603
            # self.ax[i].plot(x,x,'r--')
            self.ax[i].tick_params(axis='both', which='major', labelsize=6)
            self.ax[i].set_xlabel('Predicted Yield',fontsize=6)
            self.ax[i].set_ylabel('True Yield',fontsize=6)
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
            ## The axis are adjusted and x and y labels are added to be 'Predicted Yield' and 'True Yield' respectively 
            ## The axes are adjusted so that they have a spacing of 1 unit between [-2,2], then adjust the axis scaling
            ## to ensure that it is equal. 


    def __init__(self,model):
        super().__init__()
        ## This inherits plot_model's class variable fig and ax along with its class method colorbar
        ## The input is of object type model prresent in model_module.py file
        ## As specified above fig is an object of type Figure while axe is a n-dimensional array
        iq_data,sh_data=[[],[]],[[],[]]
        ## Two lists are created one for IQ cells and one for Shffule cells
        for pred,true,cell in zip(model.plotpairs_cv[1],model.plotpairs_cv[0],model.plotpairs_cv[2]):
            if cell[0]==1:
                iq_data[0].append(pred)
                iq_data[1].append(true)
            else:
                sh_data[0].append(pred)
                sh_data[1].append(true)
        ## The corresponding plot pairs information is either placed in the sh_data or the iq_data list
        self.ax[0].scatter(iq_data[0], iq_data[1],s=16,marker='.',color='maroon',alpha=0.25,label='LysY/$I^q$')
        self.ax[0].scatter(sh_data[0], sh_data[1],s=16,marker='.',color='orange',alpha=0.25,label='SHuffle')
        ## A scatter plot is created for the true yield against the predicted yield in orange or maroon color depending on whether
        ## the shuffle cells or IQ cells are being tracked. This is done in the top subplot of the figure and axes attribute
        ## inherited from the model_plot class
        # self.ax[0].set_title('CV_MSE='+str(round(model.model_stats['cv_avg_loss'],3))+r'$\pm$'+str(round(model.model_stats['cv_std_loss'],3)))
        iq_data,sh_data=[[],[]],[[],[]]
        for pred,true,cell in zip(model.plotpairs_test[1],model.plotpairs_test[0],model.plotpairs_test[2]):
            if cell[0]==1:
                iq_data[0].append(pred)
                iq_data[1].append(true)
            else:
                sh_data[0].append(pred)
                sh_data[1].append(true)
        self.ax[1].scatter(iq_data[0], iq_data[1],s=16,marker='.',color='maroon',alpha=0.25,label='LysY/$I^q$')
        self.ax[1].scatter(sh_data[0], sh_data[1],s=16,marker='.',color='orange',alpha=0.25,label='SHuffle')
        ## A scatter plot is created for the true yield against the predicted yield in orange or maroon color depending on whether
        ## the shuffle cells or IQ cells are being tracked. This is done in the bottom subplot of the figure and axes attribute
        ## inherited from the model_plot class
        # self.ax[1].set_title('Test_MSE='+str(round(model.model_stats['test_avg_loss'],3))+r'$\pm$'+str(round(model.model_stats['test_std_loss'],3)))
        self.add_axis()
        ## The add_axis() function is called and set up the axis of the two subplots and the area of correlations
        ## in the graph and the x and y labels for each plot
        self.fig.tight_layout()
        ## The figure is then squeezed to fit within the given figure area. 


class x_to_assay_plot(model_plot):
    
    def add_axis(self):
        for i in range(2):
            self.ax[i].set_xlabel('Predicted Assay Score')
            self.ax[i].set_ylabel('True Assay Score')
            self.ax[i].set_xlim([0,1])
            self.ax[i].set_ylim([0,1])
            self.ax[i].set_xticks([0,0.5,1])
            self.ax[i].set_yticks([0,0.5,1])
            self.ax[i].set_yticklabels(['0','','1'])
            self.ax[i].set_xticklabels(['0','','1'])
            self.ax[i].set_aspect('equal')
            self.ax[i].tick_params(axis='both', which='major', labelsize=6)
        ## This function sets x and y axis labels for both the subplots to 'Predicted Assay Score' and 'True Assay Score' respectively
        ## Then set the domain and range of the x and y axis to be [0,1] respectively and the x and y ticks were placed every 0.5 units. 
        ## Then each subplot was scaled to ensure that each axis scaling is eqaul.


    def __init__(self,model):
        #This inherits plot_model's class variable fig and ax along with its class method colorbar
        ## The input is of object type model prresent in model_module.py file
        ## As specified above fig is an object of type Figure while axe is a n-dimensional array
        super().__init__()
        _,_,_,img1=self.ax[0].hist2d(model.plotpairs_cv[1], model.plotpairs_cv[0], bins=(np.linspace(0,1,21), np.linspace(0,1,21)), cmap=plt.cm.jet,cmin=1)
        ## In the top subplot a histogram plot is created with the plotpairs cross validated data from the model input as input values
        self.colorbar(img1)
        ## The colorbar() function is run on the object thereby returning a colorbar for the top histogram subplot. 
        _,_,_,img2=self.ax[1].hist2d(model.plotpairs_test[1], model.plotpairs_test[0], bins=(np.linspace(0,1,21), np.linspace(0,1,21)), cmap=plt.cm.jet,cmin=1)
        self.colorbar(img2)
        ## Similarly in the bottom subplot a histogram plot is created with the plotpairs testing data from the model input as input values. A colorbar is created 
        ## using the colorbar() function of model_plot class.
        self.add_axis()
        ## The add-axis() function is run to ensure that the axis are scaled to [0,1] and the appropriate labels are placed in each axis
        ## Finally  the figure is then squeezed to fit within the given figure area. 
        self.fig.tight_layout()

