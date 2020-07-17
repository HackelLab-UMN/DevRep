import sys
import submodels_module as modelbank


def main():
    '''
    compare test performances when reducing training sample size. This version is for first paper, predicting yield from assays and one-hot encoded sequence. 
    '''


    a=int(sys.argv[1])
    if a<4:
        b=0
    elif a<8:
        a=a-4
        b=1
    else:
        print('incorrect toggle number')



    arch_list=['ridge','svm','forest','fnn']
    if b==0:
        mdl=modelbank.seqandassay_count_to_yield_model([1,8,10],arch_list[a],1)
    elif b==1:
        mdl=modelbank.assay_count_to_yield_model([1,8,10],arch_list[a],1)

    mdl.cross_validate_model()
    mdl.limit_test_set([1,8,10])
    mdl.test_model()





if __name__ == '__main__':
    main()