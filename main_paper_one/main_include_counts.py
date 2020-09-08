import sys
import submodels_module as modelbank
## The objects created in submodels_module is imported as modelbank. 
def main():
    '''
    compare test performances when reducing training sample size. This version is for first paper, predicting yield from assays and one-hot encoded sequence. 
    '''
    a=int(sys.argv[1])
    ## The second element in the command line input is stored as a integer value. The 
    ## acceptable integer value range is from [0,8)
    if a<4:
        b=0
        ## If the a value is less than 4, then the b value is set to 0
    elif a<8:
        a=a-4
        b=1
        ## If the a value is between 4 and 8, then the b value is set to 1
    else:
        print('incorrect toggle number')
        ## Finally a error is displayed if the a value is not within the integer range

    arch_list=['ridge','svm','forest','fnn']
    ## A string list is created with different strings corresponding to the name of the regression models constructed in the model_architecture.py script
    if b==0:
        mdl=modelbank.seqandassay_count_to_yield_model([1,8,10],arch_list[a],1)
        ## if the b value is 0 then the seqandassay_count_to_yield_model object in the
        ## submodels_module.py script. This object is instantiated with an integer list whoose elements
        ## are the assays scores used to build the regression model. In this case the assay scores
        ## used are 1,8,10. It is also instantiated with a element from the arch_list decided by the 'a'
        ## index of the arch_list. The last object required for instantiation is the sample fraction of 1.
        ## This object is stored in the mdl object
    elif b==1:
        mdl=modelbank.assay_count_to_yield_model([1,8,10],arch_list[a],1)
        ## if the b value is 1 then the assay_count_to_yield_model object in the
        ## submodels_module.py script. This object is instantiated with an integer list whoose elements
        ## are the assays scores used to build the regression model. In this case the assay scores
        ## used are 1,8,10. It is also instantiated with a element from the arch_list decided by the 'a'
        ## index of the arch_list. The last object required for instantiation is the sample fraction of 1.
        ## This object is stored in the mdl object

    mdl.cross_validate_model()
    mdl.limit_test_set([1,8,10])
    mdl.test_model()
    ## Initally the cross_validate_model() function of the parent model class is run
    ## This determines the hyperparameters for the regression model. Then the limit_test_set()
    ## function of the parent class x_to_yield_model is run to modify the testing_df class dataframe
    ## to reflect only the 1,8,10 assays scores. Finally the hyperparameters are used along with the 
    ## training dataset to train the regression model. 





if __name__ == '__main__':
    main()