import sys
import submodels_module as modelbank
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import load_format_data

def main():
    '''
    compare test performances when reducing training sample size. This version is for first paper, predicting yield from assays and one-hot encoded sequence. 
    '''
    ## A command line input is required when running this program. The integer input
    ## should be between 0-12.
    a=int(sys.argv[1])
    if a<4:
        b=0
        ## if the input is less than 4 then b value is set to 0
    elif a<8:
        a=a-4
        b=1
        ## if a is between 4-8 then the b value is set to 1 and a is reduced by 4
    elif a<12:
        a=a-8
        b=2
        ## if a is between 8-12 then the b value is set to 2 and a is reduced by 8
    elif a==12:
        b=3
        a=a-12
        ## if a is equal to 12 then the b value is set to 3 and a is set to 0. 
    else:
        print('incorrect toggle number')
        ## If the inout is out of bounds then an error message is printed. 
    arch_list=['ridge','svm','forest','fnn']
    ## A string list is created containing the names of the different regression models and stored as arch_list
    # size_list=[0.055,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    size_list=[0.7,0.8,0.9,1]
    ## A float list is created containing varying amounts of sample fractions and stored as size_list
    for size in size_list:
        ## each element in the size_list array, we check the value of the b value created in the above if-else
        ## statements and this dictates the kind of submodel_module.py object created
        ## if b = 0, then a seqandassay_to_yield_model object is created with an assay list of [1,8,10]
        ## a regression model dictated by the 'a' index of the arch_list and the size determined by the iteration of size_list
        if b==0:
            mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch_list[a],size)
        ## if b = 1, then a assay_to_yield_model object is created with an assay list of [1,8,10]
        ## a regression model dictated by the 'a' index of the arch_list and the size determined by the iteration of size_list
        elif b==1: #1,5,9,12
            mdl=modelbank.assay_to_yield_model([1,8,10],arch_list[a],size)
        ## if b = 2, then a seq_to_yield_model object is created with a regression model dictated by
        ## the 'a' index of the arch_list and the size determined by the iteration of size_list
        elif b==2: 
            mdl=modelbank.seq_to_yield_model(arch_list[a],size)
        ## if b = 3, then a control_to_yield_model object is created with a regression model dictated by
        ## the 'a' index of the arch_list and the size determined by the iteration of size_list
        elif b==3:
            mdl=modelbank.control_to_yield_model(arch_list[a],size)
            
        for seed in range(9): #no seed is seed=42
            ## For each element in the int range [0,9). The sample_seed class int to the element
            ## Then the trial data, model data and plots are updated to reflect the new sample_seed size
            mdl.change_sample_seed(seed)
            ## Then the best hyperparameters for the given model and seed size is determined using the cross_validate_model()
            ## function from the model object 
            mdl.cross_validate_model()
            ## Following this limit_test_set() function defined in the x_to_yield_model parent class to update the
            ## testing_df class dataframe to reflect the 1,8,10 assays.
            mdl.limit_test_set([1,8,10])
            ## Finally using the test_model() function from the model parent class  is run to
            ## train the model using the hyperparameters defined above and the training data to predict the testing dataset.
            mdl.test_model()
## The above for loops are used to train for different type of models for different types of regression models with different sample fraction sizes
# if __name__ == '__main__':
#     main()

arch_list=['ridge','svm','forest']
size_list=[0.055,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
## A string list arch_list is created, where each element is the class name of the regression models defined in the model_architecture program
## A float list size_list is also created with values in (0,1] where each element represent the different sample_fractions  
best_arch_list=[]
loss_per_model,std_per_model=[],[]
cv_loss_per_model,cv_std_per_model=[],[]
## Following this 5 empty list are created, one is a best_arch_list, the other four track the loss and standard deviation for the models
## for both the cross validated models and the other for everything.
for b in range(4):
    ## For each element in the int range of [0,3), this is stored to the object b.
    loss_per_size,std_per_size=[],[]
    cv_loss_per_size,cv_std_per_size=[],[]
    ## Four empty lists are created tracking the loss and standard deviation for each sample_fraction for both cross_validation and not
    for size in size_list:
        ## for each element in the size_list we create four objects:
        ## min_cv and min_test which are set to infinity and min_cv_std and min_std which are set to 0.
        min_cv,min_cv_std,min_test,min_std=np.inf,0,np.inf,0
        for arch in arch_list:
            ## Then for each element in the arch_list
            ## Depending on which outermost iterative loop(i.e what value of b) we are in the model object we are going to create
            ## from the submodels_module.py program 
            ## if b = 0, then a seqandassay_to_yield_model object is created with an assay list of [1,8,10]
            ## a regression model a sample_fraction determined by the iteration of arch_llist and size_list respectively. 
            if b==0:
                mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch,size)
            ## if b = 1, then a assay_to_yield_model object is created with an assay list of [1,8,10]
            ## a regression model a sample_fraction determined by the iteration of arch_llist and size_list respectively.
            elif b==1: #1,5,9,12
                mdl=modelbank.assay_to_yield_model([1,8,10],arch,size)
            ## if b = 2, then a seq_to_yield_model object is created with a regression model a sample_fraction
            ##  determined by the iteration of arch_llist and size_list respectively.
            elif b==2: 
                mdl=modelbank.seq_to_yield_model(arch,size)
            ## if b = 2, then a control_to_yield_model object is created with a ridge regression model and a sample_fraction
            ##  determined by the iteration of size_list.
            elif b==3:
                mdl=modelbank.control_to_yield_model('ridge',size)
        
            cur_cv_loss=[]
             cur_test_loss=[]
            cur_cv_loss.append(mdl.model_stats['cv_avg_loss'])
            cur_test_loss.append(mdl.model_stats['test_avg_loss'])
            ## Once the model object is created and stored on mdl, two new lists cur_cv_loss and cur_test_loss are created and the
            ## cv_avg_loss and test_avg_loss columns in the model_stats class dataframe are accessed and stored respectively.
            for seed in range(9):
                ## For each element in the int range of [0,9), the change_sample_seed() function of the x_to_yield_model parent class
                ## defined in the submodels_module.py program is run which changes the sample_seed class int to reflect the element and updates the
                ## trial data, model data and plots to reflect this change. 
                mdl.change_sample_seed(seed)
                cur_cv_loss.append(mdl.model_stats['cv_avg_loss'])
                cur_test_loss.append(mdl.model_stats['test_avg_loss'])
                ## From the newly updated model_stats class dataframe the cv_avg_loss and test_avg_loss columns are again stored in
                ## the cur_cv_loss and cur_test_loss lists.
            if np.average(cur_cv_loss)<min_cv:
                min_cv=np.average(cur_cv_loss)
                min_cv_std=np.std(cur_cv_loss)
                ## If the average of the cur_cv_loss list is less than the min_cv object created above
                ## Then the min_cv is set to the average of the cur_cv_loss while the min_cv_std is the
                ## standard deviation of the cur_cv_loss list. 
                if cur_test_loss[0]==np.inf:
                    print(mdl.model_name)
                    print(cur_test_loss)
                    ## If the first element of the cur_test_loss is equal to infinity then the model name and the cur_test_loss
                    ## list are printed. 
                min_test=np.average(cur_test_loss)
                min_std=np.std(cur_test_loss)
                best_arch=arch
                ## the average if the cur_test_loss is calculated and set to the min_test object while the standard
                ## deviation of the cur_test_loss is set to the min_std object and the arch iteration is set to best_arch object
        best_arch_list.append(best_arch)
        loss_per_size.append(min_test)
        std_per_size.append(min_std)
        cv_loss_per_size.append(min_cv)
        cv_std_per_size.append(min_cv_std)
        ## Then the above calculated best_arch, min_test, min_std, min_cv and min_cv_std are added to the best_arch_list
        ## loss_per_size, std_per_size, cv_loss_per_size and cv_std_per_size. 
    loss_per_model.append(loss_per_size)
    std_per_model.append(std_per_size)
    cv_loss_per_model.append(cv_loss_per_size)
    cv_std_per_model.append(cv_std_per_size)
    ## Then the above lists loss_per_size, std_per_size, cv_loss_per_size and cv_std_per_size are added to the lists
    ## loss_per_model, std_per_model, cv_loss_per_model and cv_std_per_model respectively. 

size_list=np.multiply(size_list,len(mdl.training_df))
## Following this each element of the size_list is multiplied by the length of the mdl class dataframe training_df.
control_model=modelbank.control_to_yield_model('ridge',1)
## A new control_to_yield_model() object os created from the submodel_module.py program, the object is instantiated with a
## ridge model regression and a sample fraction of 1. 
ontrol_model.limit_test_set([1,8,10])
## Then using the limit_test_set() function of the x_to_yield_model parent class the testing_df class dataframe is changed to reflect the 1,8,10 assay scores
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
## Then the explode_yield() function of the load_format_data.py program is run using the testing_df of the mdl object.
## The output is tored in exploded_df. The y_std column in exploded_df is accessed and the mean squarred is calculated and stored in exp_var
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
## Then the explode_yield() function of the load_format_data.py program is run on the training_df of the mdl object.
## The output is again stored in exploded_df. The y_std column of it is accessed and the mean squarred value of that column is store in cv_exp_var
exploded_df,_,_=load_format_data.explode_yield(control_model.training_df)
cv_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

## A Figure and Axis object is created and stored in fig and ax. A plot is created with two subplots in it.
fig,ax=plt.subplots(1,2,figsize=[4,2],dpi=300,sharey=True)
ax[0].errorbar(np.add(size_list,0),cv_loss_per_model[3],yerr=cv_std_per_model[3],label='Strain Only',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='red')
ax[0].errorbar(np.add(size_list,0),cv_loss_per_model[2],yerr=cv_std_per_model[2],label='OH Sequence',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='blue')
ax[0].errorbar(np.add(size_list,0),cv_loss_per_model[1],yerr=cv_std_per_model[1],label=r"$P_{PK37},G_{SH},\beta_{SH}$",marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='black')
ax[0].errorbar(np.add(size_list,0),cv_loss_per_model[0],yerr=cv_std_per_model[0],label=r"$Seq.&\ P_{PK37},G_{SH},\beta_{SH}$",marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='orange')
## Intially the updated size_list (X) is plotted against the fourth element of the cv_loss_per_model (Y) list with the fourth element of the cv_std_per_model acting as
## the error parameters for the y-data, the plot-line is labelled 'Strain Only'. This is then also done with the third,second and first element of the cv_loss_per_model
## along with the respective elements of the cv_std_per_model as errorbars and it is labelled 'OH Sequence', '$P_{PK37},G_{SH},\beta_{SH}$' and '$Seq.&\ P_{PK37},G_{SH},\beta_{SH}$'
## respectively.
ax[0].axhline(cv_exp_var,0,198,color='purple',linestyle='--',label='Experimental Variance')
## A horizontal line is constructed on the axis at the cv_exp_var position and labelled 'Experimental Variance'
ax[0].legend(fontsize=6,framealpha=1)
ax[0].tick_params(axis='both', which='major', labelsize=6)
ax[0].set_ylabel('CV Loss',fontsize=6)
ax[0].set_xlabel('Number of Training Sequences',fontsize=6)
ax[0].set_ylim([0.3,1])
## Intially a legend is created and the apperance of ticks are modified. Following that x and y labels are created and are 'Number of Training Sequence'
## and 'CV loss' respectively. Then the y axis is limited to range from [0.3,1]. The above changes are only done for the first subplot of the two c onstructed. 
# ax[0].axis('scaled')
ax[1].errorbar(np.add(size_list,0),loss_per_model[3],yerr=std_per_model[3],label='Strain Only',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='red')
ax[1].errorbar(np.add(size_list,0),loss_per_model[2],yerr=std_per_model[2],label='OH Sequence',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='blue')
ax[1].errorbar(np.add(size_list,0),loss_per_model[1],yerr=std_per_model[1],label='Assays',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='black')
ax[1].errorbar(np.add(size_list,0),loss_per_model[0],yerr=std_per_model[0],label='Sequence and Assays',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='orange')
## Intially the updated size_list (X) is plotted against the fourth element of the loss_per_model (Y) list with the fourth element of the std_per_model acting as
## the error parameters for the y-data, the plot-line is labelled 'Strain Only'. This is then also done with the third,second and first element of the loss_per_model
## along with the respective elements of the std_per_model as errorbars and it is labelled 'OH Sequence', 'Assays' and 'Sequence and Assays' respectively.
ax[1].axhline(exp_var,0,198,color='purple',linestyle='--',label='Experimental Variance')
## A horizontal line is constructed on the axis at the exp_var position and labelled 'Experimental Variance'
ax[1].tick_params(axis='both', which='major', labelsize=6)
ax[1].set_ylabel('$Test^2$ Loss',fontsize=6)
ax[1].set_xlabel('Number of Training Sequences',fontsize=6)
ax[1].set_ylim([0.3,1])
## Intially a legend is created and the apperance of ticks are modified. Following that x and y labels are created and are 'Number of Training Sequence'
## and '$Test^2$ Loss' respectively. Then the y axis is limited to range from [0.3,1]. The above changes are only done for the first subplot of the two c onstructed. 
# ax[1].axis('scaled')
fig.tight_layout()
fig.savefig('./changing_sample_size.png')
plt.close()
## First the figure is intially squeezed into the figure layout, then the figure is saved as a png file and saved under the name changing_sample_size.
## Then the figure window is closed. 

