import sys
import submodels_module as modelbank


def main():

    toggle_no=int(sys.argv[1])
    ## The integer input from the command line prompt is stored in the toggle_no variable. The input
    ## has to be between [0,3]. Any other input would cause an error
    a_models=['ridge','forest','svm','fnn']
    ## A string list is created called a_models with elements corresponding different regression models
    ## outlined in the model_architecture.py script
    a=modelbank.seqandassay_to_yield_model([1,8,10],a_models[toggle_no],1)
    ## Then a seqandassay_to_yield_model() object defined in the submodels_module.py script is created.
    ## It is instantiated with an integer list, a string and a float. The integer list corresponds to the
    ## assay scores that are going to be used to build the regression model. The string is the "toggle_no"
    ## element of the a_models list while the float corresponds to the sample fraction.
    a.cross_validate_model()
    a.test_model()
    a.plot()
    ## First the cross_validate_model() function of the parent class model is run which determines the hyperparameters for this
    ## then the test_model() function of the parent class model is run which using the hyperparameters and training dataset trains
    ## the regression model. finally the plot() function also of the model parent class is run which constructs the figure from the
    ## given data and saves it.

if __name__ == '__main__':
    main()