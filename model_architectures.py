from hyperopt import hp
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
##
class ridge_model():
    def __init__(self):
        ## This creates an object of instance ridge_model with a class variable parameter_space
        ## which is a dictionaray with the key being related to a value unifromly between -5 and 5  
        self.parameter_space={
        'alpha':hp.uniform('alpha', -5, 5)
        }
         
    def set_model(self,space,**kwargs):
        ## This function runs a Ridge regression with the bais of the regression specified in 
        ## space dictionary, under the 'alpha' key. space variable is a dictionary with the curretn set of hyper parameters.
        self.model=Ridge(alpha=10**space['alpha'])
        
    def fit(self,x,y):
        ## This function takes in training and target data and fits it in a ridge regression
        ## model specified in the set_model function. X corresponds to an array of training data
        ## y is an array of target values. 
        self.model.fit(x,y)

class forest_model():
    def __init__(self):
        ## This creates an object of instance forest_model with a class variable parameter_space
        ## which is a dictionary with n_estimators and max_depth a random integer in the defined space
        ## and max_features a random number choosen in a given space.
        self.parameter_space={
        'n_estimators':hp.quniform('n_estimators', 1, 500, 1),
        'max_depth':hp.quniform('max_depth', 1, 100, 1),
        'max_features':hp.uniform('max_features', 0, 1)
        }
        
    def set_model(self,space,**kwargs):
        ## This function runs a Random Forest regression with the features of the regression: no:of trees = n_estimators
        ## and tree depth = max_depth and max_features is the number of features to consider for the best split
        ## space variable is a dictionary with the curretn set of hyper parameters
        self.model=RandomForestRegressor(n_estimators=int(space['n_estimators']),max_depth=int(space['max_depth']),max_features=space['max_features'])
        
    def fit(self,x,y):
        ## This function takes in training and target data and fits it in a ridge regression
        ## model specified in the set_model function. X corresponds to an array of training data
        ## y is an array of target values.
        self.model.fit(x,y)

class svm_model():
    def __init__(self):
        ## This creates an object of instance svm_model with a class variable parameter_space
        ## which is a dictionary detialing the hyperparameters spaces necessary to run a suppourt vector regression
        self.parameter_space={
        'gamma':hp.uniform('gamma', -3, 3),
        'c':hp.uniform('c', -3, 3)
        }
        
    def set_model(self,space,**kwargs):
        ## This function runs a Epsilon-Suppourt Vector Regression with the necessary inputs from the class variable parameter_space
        ## A radial basis function kernel is run with a kernel coefficent determined by the gamma key in the space dictionary and a
        ## a reguralization parameter determined by the c key in the space dictionary
        self.model=SVR(gamma=10**space['gamma'],C=10**space['c'])
        
    def fit(self,x,y):
        ## This function takes in training data and runs a rbf - suppourt vector regression on the given data
        self.model.fit(x,y)

class nn():
    def __init__(self):
        ## This creates an object of instance nn with a class variable parameter_space which is dictionary
        ## detailing the hyperparameters spaces necessary to run a neural netowrk regression. 
        self.parameter_space={
        'epochs':hp.uniform('epochs', 0, 3),
        'batch_size':hp.uniform('batch_size',0.1,1),
        'dense_layers':hp.quniform('dense_layers',1,5,1),
        'dense_nodes_per_layer':hp.quniform('dense_nodes_per_layer',1,100,1),
        'dense_drop':hp.uniform('dense_drop',0.1,0.5)
        }
    
    def set_init_model(self,space,**kwargs):
        tf.keras.backend.clear_session()
        ## this clears the precious layers and variables used in Keras from previous models and resets the learning phase
        self.space=space
        ## creates a new class variable and sets it to the space dictionary input
        input_shape=kwargs['xa_len']+kwargs['cat_var_len']
        self.xa_len=kwargs['xa_len']
        self.lin_or_sig=kwargs['lin_or_sig']
        ## xa_len represents the length of the training data and lin_or_sig suggests whether a linear or sigmoidal regression is to be run
        self.inputs=tf.keras.Input(shape=(input_shape,))
        ## Input is used to instantiate a keras tensor flow

    def dense_layers(self):
        ## This function only works for child classes defined below. The child classes are defined below and are class that have
        ## nn in their name. 
        layers=int(self.space['dense_layers'])
        nodes=int(self.space['dense_nodes_per_layer'])
        ## The nodes and layers of the neural network are derived from the previously created self.space variable
        ## this is then assigned to temporary function variables called nodes and layers respectively
        dense_dropout=self.space['dense_drop']
        ## droput layer is the amount of neurons dropped to prevent overfitting the fraction of amount dropped is determined by the value to 
        ## the key assignment of dense_drop in self.space
        dense,drop=[[]]*layers,[[]]*layers
        drop[0]=tf.keras.layers.Dropout(rate=dense_dropout)(self.recombined)
        ## Applies the dropput to the input 
        ###following used for hidden layers
        dense[0]=tf.keras.layers.Dense(nodes,activation='relu')(drop[0])
        for i in range(1,layers):
            ## a densley connected NN layer is created and the respective layer is run through the droput function to drop some nodes
            drop[i]=tf.keras.layers.Dropout(rate=dense_dropout)(dense[i-1])
            dense[i]=tf.keras.layers.Dense(nodes,activation='relu')(drop[i])
        
        ###final output uses last dropout layer
        self.outputs=tf.keras.layers.Dense(1,activation=self.lin_or_sig)(drop[-1])  


    def set_end_model(self):
        ## Similar to the previous function this function only works for child classes defined below.
        self.model=tf.keras.Model(inputs=self.inputs,outputs=self.outputs)
        ## groups layers into an object, self.model, with training and interference features.
        self.model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError())
        ## configures the model for training with an adam algorithm optimizer and the loss is computed with the mean square of error btw labels and predictions

    def fit(self,x,y):
        self.epochs=int(10**self.space['epochs'])
        self.batch_size=int(len(x)*self.space['batch_size'])        
        self.model.fit(x,y,epochs=self.epochs,batch_size=self.batch_size,verbose=0)
        ## A model regression is run for a given set of training x and training y data, with the number of epochs to train the model
        ## specified in the self.space dictionary and the number of samples per gradient update specified in batch_size of self.space dictionary
        
        
class fnn(nn):
    def __init__(self):
        ## This inherits from the nn class and its respective class variables  
        super().__init__()
        pass

    def set_model(self,space,**kwargs):
        ## This constructs a model for a given hyperparameter space using the set_init_model, dense_layers and set_end_model function defined
        ## in the nn parent class
        self.set_init_model(space,**kwargs)
        self.recombined=self.inputs
        self.dense_layers()
        self.set_end_model()

    def get_seq_embeding_layer_model(self):
        print('this is just the regular OH encoding')
        return None

class emb_nn(nn):
    def __init__(self):
        ## This inherits from the nn class and its respective class variables, it adds the 'AA_emb_dim" key to the parameter_space dictionary
        super().__init__()
        self.parameter_space['AA_emb_dim']=hp.quniform('AA_emb_dim',1,20,1)

    def input_to_AA_emb(self):
        ## This function can only be run after the set_init_model is run so that the space, xa_len, lin_or_sig and inputs class variables are created
        emb_dim=int(self.space['AA_emb_dim'])
        self.input_seq=tf.keras.layers.Lambda(lambda x: x[:,:self.xa_len])(self.inputs)
        ## A new class variable input_seq is created as a layer object for the x function depednent on the xa_len
        self.AA_embed=tf.keras.layers.Embedding(21,emb_dim,input_length=16)(self.input_seq) #(batch size, seq len, embed size)
        ## Another class variable of type Embedding is created, where it takes in an input of dimension length 21 and the dimensions of the
        ## dense embedding equivalent to the emb_dim value previously specified with a constant input length of 16.

    def recombine_cat_var(self):
        ## This function can only be used in the child classes of emb_nn outlined below. 
        self.input_cat_var=tf.keras.layers.Lambda(lambda x: x[:,self.xa_len:])(self.inputs)
        ## this creates a self.inout_cat_var variable as a layer object similar to the self.input_seq object
        self.recombined=tf.keras.layers.concatenate([self.flat_seq,self.input_cat_var])
        ## a self.recombined class object is created to concatenate the given inputs, which in this case is the self.flat_seq and the previously created
        ## self.input_cat_var class objects. 

    def get_seq_embeding_layer_model(self):
        ## This function returns a model which has integerated the model input with the sequence embedding as output.and returns this object
        return tf.keras.Model(inputs=self.model.input,outputs=self.model.get_layer('seq_embedding').output)

    def reduce_to_linear_embedding(self):
        ## This changes the value of the 'dense_layers' key in the parameter_space dictionary to a choice Apply
        self.parameter_space['dense_layers']=hp.choice('dense_layers',[1])



class emb_fnn_maxpool(emb_nn):
    def __init__(self):
        ## This inherits from the emb_nn class and its respcetive class variables.
        super().__init__()
        pass

    def set_model(self,space,**kwargs):
        ## This constructs a model for a given hyperparameter space using the set_init_model, dense_layer and set_end_model function defined
        ## in the nn parent class along with the input_to_AA_emb and recombine_cat_var function from the emb_nn class. This also defines the flat_seq
        ## class variable.
        self.set_init_model(space,**kwargs)
        self.input_to_AA_emb()
        self.flat_seq=tf.keras.layers.GlobalMaxPool1D(name='seq_embedding')(self.AA_embed) #pool across sequence len, end with (batch size, embed size)
        self.recombine_cat_var()
        self.dense_layers()
        self.set_end_model()

class emb_fnn_maxpool_linear(emb_fnn_maxpool):
    def __init__(self):
        ## This inherits from the emb_fnn_maxpool class and its respcetive class variables but it changes the value to the 'dense_layers' key
        super().__init__()
        self.reduce_to_linear_embedding()

class emb_fnn_flat(emb_nn):
    def __init__(self):
        ## This inherits from the emb_nn class and its recpective class variables. 
        super().__init__()
        pass

    def set_model(self,space,**kwargs):
        ## This constructs a model for a given hyperparameter space using the set_init_model, dense_layer and set_end_model function defined
        ## in the nn parent class along with the input_to_AA_emb and recombine_cat_var function from the emb_nn class. This also defines the flat_seq
        ## class variable.
        self.set_init_model(space,**kwargs)
        self.input_to_AA_emb()
        self.flat_seq=tf.keras.layers.Flatten(name='seq_embedding')(self.AA_embed) #pool across sequence len, end with (batch size, embed size)
        self.recombine_cat_var()
        self.dense_layers()
        self.set_end_model()

class emb_rnn(emb_nn):
    def __init__(self):
        ## This inherits from the emb_rnn class and its respective class variables and it adds the new keys like 'units','input_dropout' and 'recurrent_dropout' 
        ## to the parameter_space dictionary 
        super().__init__()
        self.parameter_space['units']=hp.quniform('units',1,100,1)
        self.parameter_space['input_dropout']=hp.uniform('input_dropout',0.1,0.5)
        self.parameter_space['recurrent_dropout']=hp.uniform('recurrent_dropout',0.1,0.5)

    def set_model(self,space,**kwargs):
        ## creates temporary variables from the input hyperparameter dictionary
        units=int(space['units'])
        input_dropout=space['input_dropout']
        recurrent_dropout=space['recurrent_dropout']
        ## This constructs a model for a given hyperparameter space using the set_init_model, dense_layer and set_end_model function defined
        ## in the nn parent class along with the input_to_AA_emb and recombine_cat_var function from the emb_nn class. This also defines the flat_seq
        ## class variable.
        self.set_init_model(space,**kwargs)
        self.input_to_AA_emb()
        self.flat_seq=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=units,recurrent_dropout=recurrent_dropout,dropout=input_dropout),name='seq_embedding')(self.AA_embed)
        ## creates a bidirectional wrapper for a recurrrent neural network. 
        self.recombine_cat_var()
        self.dense_layers()
        self.set_end_model()

class small_emb_rnn(emb_rnn):
    def __init__(self):
        ## This inherits from the emb_rnn class and its recpective class variables and it changes the value to 'units' key in the parameter_space
        ## dictionary
        super().__init__()
        self.parameter_space['units']=hp.quniform('units',1,5,1)

class small_emb_rnn_linear(small_emb_rnn):
    def __init__(self):
        ## This inherits from the emb_rnn class and its recpective class variables and it changes the value to 'dense_layers' key in the parameter_space
        ## dictionary 
        super().__init__()
        self.reduce_to_linear_embedding()

class small_emb_atn_rnn(small_emb_rnn):
    def __init__(self):
        ## This inherits from the small_emb_rnn class and its rescpective class variables
        super().__init__()

    def set_model(self,space,**kwargs):
         ## creates temporary variables from the input hyperparameter dictionary
        units=int(space['units'])
        input_dropout=space['input_dropout']
        recurrent_dropout=space['recurrent_dropout']
        ## This constructs a model for a given hyperparameter space using the set_init_model, dense_layer and set_end_model function defined
        ## in the nn parent class along with the input_to_AA_emb and recombine_cat_var function from the emb_nn class. This also defines the flat_seq
        ## class variable.
        self.set_init_model(space,**kwargs)
        self.input_to_AA_emb()
        self.rnn_seq=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=units,recurrent_dropout=recurrent_dropout,dropout=input_dropout,return_sequences=True))(self.AA_embed)
        ## A bidirectional wrapper is created and stored in the rnn_seq class variable and is used in thie reccurent neural netowork model
        self.rnn_final= tf.keras.layers.Lambda(lambda t: t[:,-1])(self.rnn_seq)
        ## The experssion in the function is arbitarily placed in the rnn_seq wrapper
        self.atn_layer=tf.keras.layers.Attention()([self.rnn_seq,self.rnn_seq])
        ## Then a dot_product attention layer is created 
        self.flat_atn=tf.keras.layers.GlobalAveragePooling1D()(self.atn_layer)
        self.flat_seq=tf.keras.layers.Concatenate(name='seq_embedding')([self.rnn_final,self.flat_atn])
        ## The data is pooled together
        self.recombine_cat_var()
        self.dense_layers()
        self.set_end_model()

class emb_cnn(emb_nn):
    def __init__(self):
        ##this inherits from the emb_nn class and its recpective clqass variables and it adds the keys like "filters','kernel_size'
        ## and 'input_drop' to the parameter_space dictionary
        super().__init__()
        self.parameter_space['filters']=hp.quniform('filters',1,100,1)
        self.parameter_space['kernel_size']=hp.quniform('kernel_size',1,16,1) #needs to be updated for different length sequences
        self.parameter_space['input_drop']=hp.uniform('input_drop',0.1,0.5)

    def set_model(self,space,**kwargs):
        ## Creates temporary variables from the input hyperparameter dictionary
        filters=int(space['filters'])
        kernel_size=int(space['kernel_size'])
        input_drop=space['input_drop']
        ## This constructs a model for a given hyperparameter space using the set_init_model, dense_layer and set_end_model function defined
        ## in the nn parent class along with the input_to_AA_emb and recombine_cat_var function from the emb_nn class. This also defines the flat_seq
        ## class variable.
        self.set_init_model(space,**kwargs)
        self.input_to_AA_emb()
        self.input_drop=tf.keras.layers.Dropout(rate=input_drop)(self.AA_embed)
        ## A droput layer is set up in the AA_embedding with droput rate specified in the hyperparamete dictionary
        self.cov=tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,activation='relu')(self.input_drop)
        ## Creates a convolution kernel that is convoled with the input_drop layer with the filter and kernel size specification given in the
        ## input hyperparameter dictionary
        self.flat_seq=tf.keras.layers.GlobalMaxPool1D(name='seq_embedding')(self.cov)
        ## The data is pooled 
        self.recombine_cat_var()
        self.dense_layers()
        self.set_end_model()

class small_emb_cnn(emb_cnn):
    def __init__(self):
        ## This inherits from the emb_cnn class and its respective class variables and it changes the value to 'filters' key in the 
        ## self.parameter_space dictionary 
        super().__init__()
        self.parameter_space['filters']=hp.quniform('filters',1,10,1)


class small_emb_cnn_linear(small_emb_cnn):
    def __init__(self):
         ## This inherits from the small_emb_cnn class and its recpective class variables and it changes the value to 'dense_layers' key in the parameter_space
        ## dictionary 
        super().__init__()
        self.reduce_to_linear_embedding()


class small_emb_atn_cnn(small_emb_cnn):
    def __init__(self):
        ## This inherits from the small_emb_rnn class and its rescpective class variables
        super().__init__()

    def set_model(self,space,**kwargs):
        ## Creates temporary variables from the input hyperparameter dictionary
        filters=int(space['filters'])
        kernel_size=int(space['kernel_size'])
        input_drop=space['input_drop']
        ## This constructs a model for a given hyperparameter space using the set_init_model, dense_layer and set_end_model function defined
        ## in the nn parent class along with the input_to_AA_emb and recombine_cat_var function from the emb_nn class. This also defines the flat_seq
        ## class variable.
        self.set_init_model(space,**kwargs)
        self.input_to_AA_emb()
        
        self.input_drop=tf.keras.layers.Dropout(rate=input_drop)(self.AA_embed)
                ## A droput layer is set up in the AA_embedding with droput rate specified in the hyperparamete dictionary
        self.cov=tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,activation='relu')(self.input_drop)
        ## Creates a convolution kernel that is convoled with the input_drop layer with the filter and kernel size specification given in the
        ## input hyperparameter dictionary
        self.cov_flat=tf.keras.layers.GlobalMaxPool1D()(self.cov)
        self.atn_layer=tf.keras.layers.Attention()([self.cov,self.cov])
        ## Then a dot_product attention layer is created between the self.cov layer and itself.
        self.atn_flat=tf.keras.layers.GlobalMaxPool1D()(self.atn_layer)
        self.flat_seq=tf.keras.layers.Concatenate(name='seq_embedding')([self.cov_flat,self.atn_flat])
        ## Then the layers are pooled
        self.recombine_cat_var()
        self.dense_layers()
        self.set_end_model()

def get_model(model_architecture):
    ##input model_architecture is a string input into this function
    'call this to set model._model based upon model_architecture'
    ## the function variable model_switcher is a dictionary with 
    ## string key and class values. The function values are defined below in the code. 
    model_switcher= {
        'ridge': ridge_model(),
        'forest': forest_model(),
        'svm': svm_model(),
        'fnn': fnn(),
        'emb_fnn_flat': emb_fnn_flat(),
        'emb_fnn_maxpool': emb_fnn_maxpool(),
        'emb_fnn_maxpool_linear': emb_fnn_maxpool_linear(),
        'emb_rnn': emb_rnn(),
        'emb_cnn': emb_cnn(),
        'small_emb_rnn': small_emb_rnn(),
        'small_emb_cnn': small_emb_cnn(),
        'small_emb_atn_rnn': small_emb_atn_rnn(),
        'small_emb_atn_cnn': small_emb_atn_cnn(),
        'small_emb_rnn_linear': small_emb_rnn_linear(),
        'small_emb_cnn_linear': small_emb_cnn_linear()
        }
    ## This returns an object class value corresponding to the string key value input
    ## Suggestion, add a default value in return statement below in case given key value DNE
    if model_architecture in model_switcher.keys():
        return model_switcher.get(model_architecture)
    else:
        print("Please choose from the following model architectures:")
        print(list(model_switcher.keys()))
        return None
