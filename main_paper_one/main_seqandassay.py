import sys
import submodels_module as modelbank


def main():

    toggle_no=int(sys.argv[1])

    a_models=['ridge','forest','svm','fnn']
    a=modelbank.seqandassay_to_yield_model([1,8,10],a_models[toggle_no],1)
    a.cross_validate_model()
    a.test_model()
    a.plot()

if __name__ == '__main__':
    main()