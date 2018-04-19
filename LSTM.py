###########################################################################                                                                         #
# Responsible for pre-processing sensor data and building, training, &    #
# evaluating the model.                                                   #
#                                                                         #
# required input:                                                         #
###########################################################################

import pandas as pd
import numpy as np
import tensorflow as tf

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
DEBUG = True

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

######################################################################################
# Creates an RNN model.                                                              #
# Parameters:                                                                        #
#     num_attributes        : the number of attributes in the training data          #
#     num_targets           : the number of target values we want to predict         #
#     units                 : a list of integers representing the number of          #
#                             cells in each layer of the LSTM model                  #
#     activation            : a list of activation functions for each layer of the   #
#                             LSTM model                                             #
#     dropouts              : a list of dropout values for each layer of the LSTM    #
#                             model                                                  #
#     recurrent_dropouts    : a list of dropout values for each layer of the         #
#                             LSTM model                                             #
#     recurrent_initializer : initializer for the recurrent weight matrix            #
#     kernel_initializer    : initializer for the kernel weights matrix              #
#     unit_forget_bias      : add 1 to the bias of the forget gate at initialization #
#     batch_size            : the batch size                                         #
######################################################################################

def makeRNN(num_attributes,
            num_targets,
            units,
            activations,
            dropouts,
            recurrent_dropouts,
            recurrent_initializer,
            kernel_initializer,
            unit_forget_bias,
            batch_size = 1):
    if debug :
        print("making RNN")
    
    model = Sequential()

    ####################################################################
    # The first layer needs the batch_input_shape as a parameter,      #
    # and so we need to initialize it separately from the rest.        #
    ####################################################################
    
    if lstm_flag:
        model.add(LSTM(units = units[0],
                     activation = activations[0], 
                     dropout = dropouts[0],
                     recurrent_dropout = recurrent_dropouts[0],
                     return_sequences=True, 
                     unit_forget_bias=unit_forget_bias,
                     kernel_initializer = kernel_initializer,
                     recurrent_initializer = recurrent_initializer,
                     batch_input_shape = (batch_size,1,num_attributes)
                ))
    else:
        model.add(
                GRU(units = units[0],
                    activation = activations[0], 
                    dropout = dropouts[0],
                    recurrent_dropout = recurrent_dropouts[0],
                    return_sequences=True, 
                    kernel_initializer = kernel_initializer,
                    recurrent_initializer = recurrent_initializer,
                    batch_input_shape = (batch_size,1,num_attributes)
                ))
    
    info = zip(units[1:],dropouts[1:],activations[1:],recurrent_dropouts[1:])
    for i in info: #layer,dropout,activation in info:
        if debug: 
            print("units: {},dropout: {}, activation: {}".format(i[0],i[1],i[2]))
        if lstm_flag: 
            model.add(
                LSTM(units = i[0],
                     activation = i[2], 
                     dropout = i[1],
                     recurrent_dropout = i[3],
                     return_sequences=True, 
                     unit_forget_bias=unit_forget_bias,
                     kernel_initializer = kernel_initializer,
                     recurrent_initializer = recurrent_initializer,
                ))
        else:
            model.add(
                GRU(units = units[i],
                    activation = activations[i], 
                    dropout = dropouts[i],
                    recurrent_dropout = recurrent_dropouts[i],
                    return_sequences=True, 
                    kernel_initializer = kernel_initializer,
                    recurrent_initializer = recurrent_initializer,
                ))

    #Adding a single lstm cell on top with the correct number of outputs
    if lstm_flag: model.add(LSTM(num_targets,
                                 activation = None,
                                 stateful   = True))
    
    else : model.add(GRU(num_targets,
                         activation = None,
                         stateful   = True))
    
    model.add(Dense(num_targets,
                    activation = 'linear'))
    return model

######################################################################################
# Split the data into testing and validation sets, making sure to return an even     #
# number of batches in the result.                                                   #
# Parameters:                                                                        #
#     X          : the feature data                                                  #
#     y          : the target data                                                   #
#     batch_size : the batch size                                                    #
######################################################################################

def testTrainSplit(X,y,batch_size):
    X = X[0:int(np.floor(data.shape[0]/batch_size))*batch_size]
    y = y[0:int(np.floor(data.shape[0]/batch_size))*batch_size]
    l = int(np.floor(X.shape[0] * (2/3)))
    return X[:l,:],y[:l,:],X[l:,:],y[l:,:]

######################################################################################
# Get the data out of the files and in a nice format.                                #
# Parameters:                                                                        #
#     files : a list of files with the data in it                                    #
#     debug : a debug flag.                                                          #
######################################################################################

def getData(files,debug):
    
    data    = np.empty((0,num_attributes)) #initializing empty array with proper shape
    targets = np.empty((0,num_targets   )) #empty array for targets
    batches = np.array([])
    
    for f in files:
        #grab the relative pathname and we can save the data for later.
        relative_path = f[f.rfind('/')+1:]
        
        full_data = np.loadtxt(f,delimiter = ',')

        ###########################################################
        # The first three columns are targets, the rest features, #
        # and we're predicting the difference in each feature     #
        # from one time point to the next.                        #
        ###########################################################
        
        processed_data    = full_data[:-1,3:]
        processed_targets = full_data[1:,:3]-full_data[:-1,:3]

        batch = processed_data.shape[0]

        if debug :
            print("full_data.shape: {}".format(full_data.shape))
            print("processed_data.shape: {}".format(processed_data.shape))
            print("processed_target.shape: {}".format(processed_targets.shape))
                  
        data    = np.append(data, processed_data, axis=0)
        targets = np.append(targets, processed_targets, axis=0)
        batches = np.append(batches,[batch])

        np.savetxt('Data/Processed/' + relative_path, processed_data, delimiter=',', fmt = '%f')

    if debug :
        print("data[0:10]: \n{}".format(data[0:10]))
        print("targets[0:10]: \n{}".format(targets[0:10]))
        print("batches: \n{}".format(batches))
    return data,targets,batches


if __name__=="__main__":


    parser = argparse.ArgumentParser(description = "LSTM.py",
        epilog = "Use this program to train some LSTMs.",
        add_help = "How to use",
        prog = "python LSTM.py -n <units> -d <dropouts> -v <activation> -r <recurrent_dropours> -f <files> [OPTIONAL PARAMETERS]")

    parser.add_argument("-n", "--units", nargs = "*", type = int, required = True,
        help = "The number of units to use in each layer. A list of ints.")

    parser.add_argument("-d", "--dropouts", nargs = "*", type = float, required = True,
        help = "The dropout values to use for each layer. A list of floats.")

    parser.add_argument("-v", "--activations", nargs = "*", required = True,
        help = "The activation functions to use for each layer.")

    parser.add_argument("-r", "--recurrent_dropouts", nargs = "*", required = True , type = float,
        help = "The recurrent dropout values for each layer. A list of floats.")
    
    parser.add_argument("-f", "--files", nargs = "*", required = True ,
        help = "A list of input files. NOTE: A path with path expansion a la \'\\*\' is acceptable. ")

    parser.add_argument("-i", "--time_steps", type=int, default = 300,
        help = "The number of time steps to use in the TBPTT algorithm. DEFAULT: 300")
    
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

    parser.add_argument("-s", "--standard_scalar", action = 'store_false', 
        help = "The flag for whether to use a standard scalar or min-max scalar. DEFAULT: False, i.e. - use a min-max. ")

    parser.add_argument("-a", "--num_attributes", type = int, default = 8,
        help = "The number of attributes used to train the model. DEFAULT: 6")
        
    parser.add_argument("-t", "--num_targets", type = int, default = 3,
        help = "The number of target values the model will predict.")
    
    parser.add_argument("-z", "--debug", action="store_true",
        help = "The debug flag to run the program in debug mode. DEFAULT: False.")

    parser.add_argument("-b", "--batch_size", default = 1, type = int,
        help = "The debug flag to run the program in debug mode. DEFAULT: False.")

    parser.add_argument("-q","--quiet",action="store_true",
        help = "Flag to run TensorFlow in quiet mode.")
    
    args = vars(parser.parse_args())

    units                   = args['units']
    dropouts                = args['dropouts']
    activations             = args['activations']
    recurrent_dropouts      = args['recurrent_dropouts']
    epochs                  = args['epochs']
    time_steps              = args['time_steps']
    unit_forget_bias        = args['unit_forget_bias']
    kernel_initializer_bias = args['kernel_initializer_bias']
    recurrent_initializer   = args['recurrent_initializer']
    lstm_flag               = args['GRU']
    standard_scaler         = args['standard_scalar']
    num_attributes          = args['num_attributes']
    num_targets             = args['num_targets']
    debug                   = args['debug']
    files                   = args['files']
    batch_size              = args['batch_size']
    quiet                   = args['quiet']

    if quiet :
        print("working in quiet TF mode.")
        tf.logging.set_verbosity(tf.logging.ERROR)

    scaler = StandardScaler() if standard_scaler else MinMaxScaler()
    
    print("Collecting and pre-processing data...")
    data,targets,samples = getData(files,debug)
    print("Done.")
    print("Specifying the model...")

    model = makeRNN(num_attributes        = num_attributes,
                    num_targets           = num_targets,
                    units                 = units,
                    activations           = activations,
                    dropouts              = dropouts,
                    recurrent_dropouts    = recurrent_dropouts,
                    recurrent_initializer = recurrent_initializer,
                    unit_forget_bias      = unit_forget_bias,
                    kernel_initializer    = kernel_initializer_bias,
                    batch_size            = batch_size)
    
    print("Done.")
    print(model.summary())
    
    model.compile(loss='mean_squared_error',optimizer="adam",metrics=["mae"])

    #########################################################################
    # Training by hand is a little ugly.                                    #
    # We run each epoch separately, and inside of it we'll run the data     #
    # through the model on a forward pass in batches.                       #
    # We'll reset the state of the net on each new sample.                  #
    #########################################################################
    
    for epoch in range(epochs):
        #j is going to allow us to distinguish each sample from the last
        j = 0
        
        print("training epoch {}...".format(epoch))

        tr_mse = 0
        tr_mae = 0
        
        te_mse = 0
        te_mae = 0

        #samples is a list of the lengths of each of our samples
        for sample_size in samples:

            print("training on a new sample...")
            
            if debug :
                print("samle_size:".format(sample_size))
                
            #X will have shape (sample_size,num_features)
            #y will have shape (samlpe_size,num_targets)
            X = data[j:j+int(sample_size),:]
            y = targets[j:j+int(sample_size),:]

            if debug :
                print("X.shape: {}".format(X.shape))
                print("y.shape: {}".format(y.shape))

            #aggregate the sample indicator
            j += int(sample_size)

            #testTrainSplit should return data with an even number of batches in it. 
            X_train, y_train, X_test, y_test = testTrainSplit(X,y,batch_size)

            x_tr_m,x_tr_n = X_train.shape
            y_tr_m,y_tr_n = y_train.shape
            x_te_m,x_te_n = X_test.shape
            y_te_m,y_te_n = y_test.shape

            #k let's us print out a progress bar. May not need it.
            k = 1

            ###################################################################
            # now we train the model on this sample, one batch at a time...   #
            # x_tr_m/batch_size gives us the number of batches in this sample #
            ###################################################################
            for i in range(int(x_tr_m/batch_size)):

                ################################################################
                # We have to reshape the data to [batch_size,1,num_attributes] #
                # and the targets to [batch_size,1,num_targets]                #
                ################################################################
                X_tr_batch = X_train[i*batch_size:(i+1)*batch_size].reshape(batch_size,1,x_tr_n)
                y_tr_batch = y_train[i*batch_size:(i+1)*batch_size].reshape(batch_size,y_tr_n)
                
                if debug :
                    print("sample[0].shape: {}".format(sample[0].shape))
                    print("sample[1].shape: {}".format(sample[1].shape))
                    print("x_tr.shape: {}".format(x_tr.shape))
                    print("y_tr.shape: {}".format(y_tr.shape))
                mse, mae = model.train_on_batch(X_tr_batch,y_tr_batch)
                tr_mse += mse 
                tr_mae += mae
                if i%150 == 0 :
                    print("-"*k +">",end="\r")
                    k += 1
            print("\n")

            ##################################################
            # Make sure to reset the state after each sample #
            ##################################################
            
            model.reset_states()

            #Rese k for an accurate progress bar. 
            k=1

            print("testing on new sample...")
            
            ###################################################################
            # now we test the model on this sample, one batch at a time...    #
            # x_te_m/batch_size gives us the number of batches in this sample #
            ###################################################################
            for i in range(int(np.floor(x_tr_m/batch_size))):

                ################################################################
                # We have to reshape the data to [batch_size,1,num_attributes] #
                # and the targets to [batch_size,1,num_targets]                #
                ################################################################
                X_te_batch = X_train[i*batch_size:(i+1)*batch_size].reshape(batch_size,1,x_te_n)
                y_te_batch = y_train[i*batch_size:(i+1)*batch_size].reshape(batch_size,y_te_n)
                
                mse, mae = model.train_on_batch(X_te_batch,y_te_batch)
                
                te_mse += mse 
                te_mae += mae

                if i%150 == 0 :
                    print("-"*k +">",end="\r")
                    k += 1

            ##################################################
            # Make sure to reset the state after each sample #
            ##################################################
            
            model.reset_states()

            
            print("_______________________")
            print("mse_training: {}".format(tr_mse / x_tr_m))
            print("mae_training: {}".format(tr_mae / x_tr_m))
            print("_______________________")
            print("mse_testing: {}".format(te_mse / x_te_m))
            print("mae_testing: {}".format(te_mae / x_te_n))
            print("_______________________")

