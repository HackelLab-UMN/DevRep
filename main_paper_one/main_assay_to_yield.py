import sys
import submodels_module as modelbank
from itertools import combinations


def main():
    ## a is a list of integers from 1-10 to represent each assay score.
    a=[1,2,3,4,5,6,7,8,9,10]
    combin_list=[]
    for i in range(1,11):
        combin_list_temp=combinations(a,i)
        for j in combin_list_temp:
            combin_list.append(j)
    ## Then each possible combination of the 10 different assays. is created and stored in an array format and then added to the empty combin_list.
    # the toggle no uses the supercomputers job array ID # to determine which assay combinaiton and architecture to train. 
    toggle_no=int(sys.argv[1])
    ## sys.argv[1] is the command line inputs and. this takes the input after the file name and assigns it to the toggle_no
    ## Depending on the magnitube of the toggle_no entered the regression model is decided.
    #determine the model architecture
    b_models=['ridge','forest','svm','fnn']
    if toggle_no<10000:
        arch=b_models[0]
        ## If the toggle_no is less than 10000 then a ridge regression is run
    elif toggle_no<20000:
        arch=b_models[1]
        toggle_no=toggle_no-10000
        ## If the toggle_no is less than 20000 then a random forest regression is run and the toggle_no is decreased by 10000
    elif toggle_no<30000:
        arch=b_models[2]
        toggle_no=toggle_no-20000
        ## If the toggle_no is less than 30000 then a support vector regression is run and the toggle_no is decreased by 20000
    elif toggle_no<40000:
        arch=b_models[3]
        toggle_no=toggle_no-30000
        ## If the toggle_no is less than 40000 then a feedforward neural network is run and the toggle_no is decreased by 30000


    b=modelbank.assay_to_yield_model(combin_list[toggle_no],arch,1)
    ## Then the assay_to_yield_model object in the submodels_module.py script is accessed and the object is 
    ## instantiated with the model architecture determined in the above if-elif-else statements and the list in toggle_no index of combin_list
    ## along with a sample fraction of 1.
    b.cross_validate_model()
    b.test_model()
    ## The cross_validate_models and test_model functions of the parent model class is run, these functions determine the hyperparamters
    ## and then using the hyperparamters train the dataset and make a test prediction
    b.plot()
    b.save_predictions()
    ## The plot() function from the parent model class and the save_predictions function from the parent x_to_yield_model class are run
    ## which creates a plot figure and saves it and it saves the model prediction in the datasets directory


if __name__ == '__main__':
    main()