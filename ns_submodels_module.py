

import submodels_module as mb
import numpy as np
import pickle
from functools import partial
from model_module import model
import load_format_data
import plot_model



class ns_seq_to_assay_model(mb.seq_to_assay_model):

    def __init__(self,s2a_params,model_nb=0): # default is zero b/c for now we are only doing one model.
        super().__init__(*s2a_params)
        self.cat_var_len=3 # [1,8,10]

        self.seq_embedding_model =None
        self.model_nb=model_nb

    def init_sequence_embeddings(self):
       # for i in self.nb_models: you can't do this: why?  b/c your loading different model parameters.
       # todo : add a another thing called model number. b/c each class can only be intilized for one model. or maybe i have an array or something.
       #  but it is model params. local to the class. im better off just pissing away the memory.
       #  truth fully i think you'll have to just have  single model for a class. so 3 classes.

        self._model.set_model(self.get_best_trial()['hyperparam'], xa_len=16,
                          # xa_len - 16
                          cat_var_len=3,  # cat_var_len - 3 [1,8,10]
                          lin_or_sig=self.lin_or_sig)  # need to build nn arch,
        self.load_model(0)  # load pkled sklearn model or weights of nn model , set z to zero.

        self.seq_embedding_model = self._model.get_seq_embeding_layer_model()

    def save_sequence_embeddings(self, df_list=None):
        # each model is already preloaded
        df=df_list.copy()
        'save sequence embeddings of model for '
        OH_matrix = np.eye(len(self.assays))
        x_a = self.get_input_seq(df)
        for i in np.arange(1): # only cat var for one assay to get sequence embedding
            cat_var = []
            for j in x_a:  # for each sequence add cat_var
                cat_var.append(OH_matrix[0].tolist())
            x = load_format_data.mix_with_cat_var(x_a, cat_var)
            seq_embedding_model = self._model.get_seq_embeding_layer_model()
            df_prediction = seq_embedding_model.predict([x])
            seq_emb_list = []
            for i in df_prediction:
                seq_emb_list.append([i])
            df.loc[:, 'learned_embedding_' + str(0)] = seq_emb_list # todo : change str to z

        return df



class ns_sequence_embeding_to_yield_model(mb.sequence_embeding_to_yield_model):
    def __init__(self,s2a_params,e2y_params):
        '''thhis is the nested sampling sequence embedding to yield model. it sacrifices memory
            in order to get higher performance. Also you can only save predictions with one model with each
            call to save_predictions. So you have to initilize a new class with every call to a new model.
        '''
        super().__init__(s2a_params,*e2y_params)
        self.model_no=s2a_params[-1]
    def init_e2y_model(self):
        self.load_model(self.model_no)

    def save_predictions(self, input_df_description=None):
        '''saves model predictions for nested sampling
            input_df_description : must be a dataframe with a column that contains the learned embedding as saved
            by ns_seq_to_assay_model.save_sequence_embeddings() as shown above.
        '''
        df=input_df_description.copy()
        x_a = self.get_input_seq(df, self.model_no)
        OH_matrix = np.eye(2)
        matrix_col = ['IQ_Average_bc', 'SH_Average_bc']
        for i in range(2):
            cat_var = []
            for j in x_a:
                cat_var.append(OH_matrix[i].tolist())
            x = load_format_data.mix_with_cat_var(x_a, cat_var)
            df_prediction = self._model.model.predict(x).squeeze().tolist()
            col_name = matrix_col[i]
            df.loc[:, col_name] = df_prediction
            col_name_std = matrix_col[i] + '_std'
            df.loc[:, col_name_std] = [0] * len(df_prediction)
         # if this a monte carlo sampling call , return the averaged predictions for the model
        return self.avg_prediction(df)



    def avg_prediction(self, df):
        'load predictions and add the two cell types yield together'
        predicted_iq_yield = df['IQ_Average_bc'].to_numpy()
        predicted_sh_yield = df['SH_Average_bc'].to_numpy()
        predicted_added_yield = np.sum([predicted_iq_yield, predicted_sh_yield], axis=0)
        return predicted_added_yield
