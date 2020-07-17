import sys
import submodels_module as modelbank
from itertools import combinations


def main():

    toggle_no=int(sys.argv[1])



    ### use predictions of yield from model a to make a seq-to-(predicted)yield model
    c_models=['ridge','fnn','emb_fnn_maxpool','emb_fnn_flat','emb_rnn','emb_cnn']
    for j in range(1):
        assay_to_yield_model_no=j #for each saved model from a
        c=modelbank.seq_to_pred_yield_model([[1,8,10],'forest',1,assay_to_yield_model_no],[c_models[toggle_no],1])
        c.cross_validate_model()
        c.test_model()
        c.plot()




if __name__ == '__main__':
    main()