import sys
import submodels_module as modelbank
from itertools import combinations
## The objects defined in the submodels_module are imported and accessed via the modelbank keyword

def main():
    toggle_no=int(sys.argv[1])
    ## This is the integer element entered in the command line prompt when running this program.
    ## The integer input should be within [0,5]
    ### use predictions of yield from model a to make a seq-to-(predicted)yield model
    c_models=['ridge','fnn','emb_fnn_maxpool','emb_fnn_flat','emb_rnn','emb_cnn']
    ## A string list is created with the different type of regression models stored in it and this is stored in the c_models lsit
    for j in range(1):
        ## Since the range(1) is just the integer  element 0, then the iterable j is only assigned to 0.
        assay_to_yield_model_no=j #for each saved model from a
        ## A new variable assay_to_yield_model_no is then also set to j value
        c=modelbank.seq_to_pred_yield_model([[1,8,10],'forest',1,assay_to_yield_model_no],[c_models[toggle_no],1])
        ## Following this a seq_to_pred_yield_model object from the submodels_module class is created with two list inputs to instantiate it
        ## The second list input has two elements, the first element is the model architecture in the 'toggle_no' index of the c_models liist.
        ## The second element is the sample fractions of the regression model. The second list is used to instantiate the class variables of this object
        ## while the first list is used to update the class variables to accurately the training and testing data to be used. 
        c.cross_validate_model()
        c.test_model()
        c.plot()
        ## First the cross_validate_model() function of the parent class model is run which determines the hyperparameters for this
        ## then the test_model() function of the parent class model is run which using the hyperparameters and training dataset trains
        ## the regression model. finally the plot() function also of the model parent class is run which constructs the figure from the
        ## given data and saves it.
        

if __name__ == '__main__':
    main()

## The above script is saved in the main() function which the above if clause runs