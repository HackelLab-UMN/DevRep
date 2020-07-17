import sys
import submodels_module as modelbank
from itertools import combinations


def main():
    #creates a list of all pairwise combinations of the 10 assays
    a=[1,2,3,4,5,6,7,8,9,10]
    combin_list=[]
    for i in range(1,11):
        combin_list_temp=combinations(a,i)
        for j in combin_list_temp:
            combin_list.append(j)

    # the toggle no uses the supercomputers job array ID # to determine which assay combinaiton and architecture to train. 
    toggle_no=int(sys.argv[1])


    #determine the model architecture
    b_models=['ridge','forest','svm','fnn']
    if toggle_no<10000:
        arch=b_models[0]
    elif toggle_no<20000:
        arch=b_models[1]
        toggle_no=toggle_no-10000
    elif toggle_no<30000:
        arch=b_models[2]
        toggle_no=toggle_no-20000
    elif toggle_no<40000:
        arch=b_models[3]
        toggle_no=toggle_no-30000


    b=modelbank.assay_to_yield_model(combin_list[toggle_no],arch,1)
    b.cross_validate_model()
    b.test_model()
    b.plot()
    b.save_predictions() 


if __name__ == '__main__':
    main()