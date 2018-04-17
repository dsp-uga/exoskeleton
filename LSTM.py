###########################################################################                                                                         #
# Responsible for pre-processing sensor data and building, training, &    #
# evaluating the model.                                                   #
#                                                                         #
# required input:                                                         #
###########################################################################

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import math

import argparse

from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense,BatchNormalization,LSTM,Input,GRU
from keras.callbacks import Callback

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

PROGRAMMATIC = True

###########################################################################
# A callback to give us the r-squared value for our models predictions    #
# during training.                                                        #
###########################################################################

class R2Callback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        X_test, y_test = self.test_data
        pred = self.model.predict(X_test)
        r2 = r2_score(pred,y_test)
        print('r2 score: {},\n'.format(r2))

###########################################################################
# Creates an RNN model.                                                   #
# Parameters:                                                             #
#     visible    : an initial layer of inputs                             #
#     layers     : a list of integers representing the number of cells in #
#                  each layer of the LSTM model                           #
#     activation : a list of activation functions for each layer of the   #
#                  LSTM model                                             #
#     dropouts   : a list of dropout values for each layer of the LSTM    #
#                  model                                                  #
###########################################################################

def makeRNN(num_layers,
            num_attributes,
            num_targets,
            layers,
            activations,
            dropouts,
            recurrent_dropouts,
            recurrent_initializer,
            kernel_initializer,
            visible,
            unit_forget_bias):
    if debug :
        print("making RNN")
    '''
    Mimicing the following:
    visible = ...
    extract = LSTM(30, activation="relu",dropout=0.5,return_sequences=True)(visible)
    extract = LSTM(20, activation="relu",dropout=0.5,return_sequences=True)(extract)
    extract = LSTM(10, activation="relu",dropout=0.5,return_sequences=True)(extract)
    extract = LSTM(5, activation="relu",dropout=0.5,return_sequences=True)(extract)
    '''
    info = zip(layers,dropouts,activations)
    for i in range(num_layers): #layer,dropout,activation in info:
        if debug: 
            print("layer: {},dropout: {}, activation: {}".format(layers[i],dropouts[i],activations[i]))
        visible = cell(layer = layers[i],
                       activation = activations[i], 
                       dropout = dropouts[i],
                       recurrent_dropout = recurrent_dropouts[i],
                       return_sequences=True, 
                       unit_forget_bias=unit_forget_bias,
                       kernel_initializer = kernel_initializer,
                       recurrent_initializer = recurrent_initializer,
                       lstm_flag = lstm_flag,
                       visible = visible)
    class1 = LSTM(num_targets,activation=None)(visible) if lstm_flag else GRU(num_targets,activation=None)(visible)
    output = Dense(num_targets, activation='linear')(class1)
    return output

###########################################################################
# Helper method.                                                          #
# Returns an LSTM layer or GRU layer depending on the flag.               #
# Needed because GRU layers do not take a unit_forget_bias parameter.     #
###########################################################################
def cell(layer,
         activation, 
         dropout,
         recurrent_dropout,
         unit_forget_bias,
         kernel_initializer,
         recurrent_initializer,
         lstm_flag,
         visible,
         return_sequences=True):
    if lstm_flag:
        return LSTM(units = layer,
                    activation = activation, 
                    dropout = dropout,
                    recurrent_dropout = recurrent_dropout,
                    return_sequences = return_sequences, 
                    unit_forget_bias = unit_forget_bias,
                    kernel_initializer = kernel_initializer,
                    recurrent_initializer = recurrent_initializer)(visible)
    else :
        return GRU(layer,
                   activation, 
                   dropout,
                   recurrent_dropout,
                   return_sequences, 
                   kernel_initializer,
                   recurrent_initializer,
                   lstm_flag)(visible)
    
###########################################################################
# Preprocesses an input file containing sensor data.                      #
# Parameters:                                                             #
#     filename : a file with some data that we want to pre-process.       #
#     scaler : an object that inherits from the sklearn classes           #
#              BaseEstimator and TransformerMixin, i.e. - an object of    #
#              type StandardScaler or MinMaxScaler                        #
###########################################################################

def preprocessing(filename,scaler):
    
    d = pd.read_csv(filename)
    
    #drop the first column -- sensor type
    d_drop = d.drop("Sensor type", axis=1)   

    if debug:
        print("Original \n{}".format(d))
        print ("Dropped Sensor Type \n{}".format(d_drop))
        
    d_matrix = d_drop.as_matrix()   # Using pandas again to convert the revised table into a matrix
    n_array = np.array(d_matrix) # now we use numpy to convert the pandas matrix into a Numpy array

    #WHY ARE WE DELING THE LAST ROW? 
    #We need an even number of rows of data.
    #So delete the last row if the revised row number is odd, otherwise cut last two rows
    n_array = n_array[:-1] if len(n_array) % 2 == 1 else n_array[:-2] 

    #Concatenate the EMG and Accerlaration rows for (nearly) the same time moments
    n_conc = np.concatenate((n_array[::2], n_array[1::2]), axis = 1)    

    #delete the second and third columns of each list in the matrix
    n_del = np.delete(n_conc, [1, 2], axis = 1)
    #Normalize the data using the supplied scaler
    n_del = scaler.fit(n_del).transform(n_del)

    #target for data t is [x,y,z] of t+1
    targets = targets = n_del[1:, 2:5]

    # no target for last example, so drop it.
    data = n_del[:-1] 
    
    if debug:
        print ("Delete odd rows \n{}".format(n_array[:16,:]))
        print ("Concatenated array \n{}".format(n_conc[:16,:]))
        print ("Delete Zero rows \n{}".format(n_del[:16,:]))
        print("n_del.shape: {}".format(n_del.shape))
        print("data.shape: {}".format(data.shape))
        
    return data, targets

###########################################################################
# Preprocess and collect a collection of files into a single file.        #
# Parameters:                                                             #
#     path : a path to the list of files that we want to collect.         #                                        
###########################################################################

def collect_data(fiels,debug):
    
    data    = np.empty((0,num_attributes)) #initializing empty array with proper shape
    targets = np.empty((0,num_targets   )) #empty array for targets
    
    for f in files:
        #preprocess this timeseries sample
        relative_path = f[f.rfind('/')+1:]
        processed_data, processed_targets = preprocessing(f,scaler)

        #append this timeseries to the dataset
        data    = np.append(data, processed_data, axis=0)
        targets = np.append(targets, processed_targets, axis=0)
        np.savetxt('Data/Processed/' + relative_path, processed_data, delimiter=',', fmt = '%f')

    if debug : print("data[0:10]: \n{}".format(data[0:10]))
    temp = list()
    for i in range(1,targets.shape[0]):
        #temp = targets[i]-targets[i-1]
        temp.append(targets[i]-targets[i-1])
    
    acceleration_targets = np.array(temp)
    data = data.reshape((data.shape[0],1,data.shape[1]))[1:]
    if debug : print("data[0:10]: \n{}".format(data[0:10]))
    
    return data,acceleration_targets

if __name__=="__main__":


    parser = argparse.ArgumentParser(description = "LSTM.py",
        epilog = "Use this program to train some LSTMs.",
        add_help = "How to use",
        prog = "python LSTM.py -l <layers> -d <dropouts> -a <activation> [OPTIONAL PARAMETERS]")

    parser.add_argument("-l", "--layers", nargs = "*", type = int, required = True,
        help = "The number of units to use in each layer. A list of ints.")

    parser.add_argument("-d", "--dropouts", nargs = "*", type = float, required = True,
        help = "The dropout values to use for each layer. A list of floats.")

    parser.add_argument("-v", "--activations", nargs = "*", required = True,
        help = "The activation functions to use for each layer.")

    parser.add_argument("-r", "--recurrent_dropouts", nargs = "*", required = True , type = float,
        help = "The recurrent dropout values for each layer. A list of floats.")

    parser.add_argument("-f", "--files", nargs = "*", required = True ,
        help = "A list of input files. NOTE: A path with path expansion a la \'\\*\' is acceptable. ")
        
    parser.add_argument("-e", "--epochs", type = int, default = 10,
        help = "The number of training epochs.")
    
    parser.add_argument("-g", "--GRU", action = "store_false",
        help = "Flag for whether to use GRUs instead of LSTMs. DEFAULT: False.")
    
    parser.add_argument("-k", "--kernel_initializer_bias", choices = ['lecun_uniform','glorot_normal','he_normal'], default='glorot_uniform',
        help = "The kernel initializer method. DEFAULT: glorot_uniform")

    parser.add_argument("-c", "--recurrent_initializer", choices = ['lecun_uniform','glorot_normal','he_normal'], default='orthogonal',
        help = "The kernel initializer method. DEFAULT: lecun_uniform")

    parser.add_argument("-u", "--unit_forget_bias", action = "store_false",
        help = "A flag to use a unit_forget_bias initialization. DEFAULT: True")

    parser.add_argument("-b", "--batch_size", default = 10,
        help = "The training batch size. DEFAULT: 10")

    parser.add_argument("-s", "--standard_scalar", action = 'store_false', 
        help = "The flag for whether to use a standard scalar or min-max scalar. DEFAULT: False, i.e. - use a min-max. ")

    parser.add_argument("-a", "--num_attributes", type = int, default = 6,
        help = "The number of attributes used to train the model. DEFAULT: 6")

    parser.add_argument("-t", "--num_targets", type = int, default = 3,
        help = "The number of target values the model will predict.")
    
    parser.add_argument("-z", "--debug", action="store_true",
        help = "The debug flag to run the program in debug mode. DEFAULT: False.")

    args = vars(parser.parse_args())

    layers                  = args['layers']
    dropouts                = args['dropouts']
    activations             = args['activations']
    recurrent_dropouts      = args['recurrent_dropouts']
    epochs                  = args['epochs']
    batch_size              = args['batch_size']
    unit_forget_bias        = args['unit_forget_bias']
    kernel_initializer_bias = args['kernel_initializer_bias']
    recurrent_initializer   = args['recurrent_initializer']
    lstm_flag               = args['GRU']
    standard_scaler         = args['standard_scalar']
    num_attributes          = args['num_attributes']
    num_targets             = args['num_targets']
    debug                   = args['debug']
    files                   = args['files']

    num_layers = len(layers)
    
    scaler = StandardScaler() if standard_scaler else MinMaxScaler()
    
    #collecting all the data and preprocess it
    print("Collecting and pre-processing data...")
    data,acceleration_targets = collect_data(files,debug)
    print("Done.")
    
    '''
    *************WHAT IS THIS FOR???******************
    input_data = list()
    for i in range(0,data.shape[0]):
        input_data.append([data[i]])
    input_data = np.array(data)
    '''
    print("Splitting the data into training and validation sets.")
    X_train, y_train, X_test, y_test = data[0:40000],acceleration_targets[0:40000],data[40000:],acceleration_targets[40000:]
    print("Done.")

    print("Specifying the model...")
    #https://machinelearningmastery.com/keras-functional-api-deep-learning/
    visible = Input(shape = (1,num_attributes))

    output = makeRNN(num_layers            = num_layers,
                     num_attributes        = num_attributes,
                     num_targets           = num_targets,
                     layers                = layers,
                     activations           = activations,
                     dropouts              = dropouts,
                     recurrent_dropouts    = recurrent_dropouts,
                     recurrent_initializer = recurrent_initializer,
                     unit_forget_bias      = unit_forget_bias,
                     kernel_initializer    = kernel_initializer_bias,
                     visible               = visible)
            
    model = Model(inputs=visible, outputs=output)
    
    print("Done.")
    # summarize layers
    print(model.summary())
    #sgd= SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error',optimizer="adam",metrics=["mae"])
    model.fit(X_train,
              y_train,
              verbose=1,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test,y_test),
              callbacks=[R2Callback((X_test,y_test))])
    
    print('X_test[:10]               : \n{}'.format(X_test[:10]))
    print('model.predict(X_test)[:10]: \n{})'.format(model.predict(X_test)[:10]))
    print('y_test[:10]               : \n{}'.format(y_test[:10]))

'''
    # feature extraction
        else:
        extract = LSTM(30, activation="relu",dropout=0.5,return_sequences=True)(visible)
        extract = LSTM(20, activation="relu",dropout=0.5,return_sequences=True)(extract)
        extract = LSTM(10, activation="relu",dropout=0.5,return_sequences=True)(extract)
        extract = LSTM(5, activation="relu",dropout=0.5,return_sequences=True)(extract)
        class11 = LSTM(3,activation=None)(extract)
        output1 = Dense(3, activation='linear')(class11)
        model = Model(inputs=visible, outputs=output1)

    if PROGRAMMATIC: 

'''
