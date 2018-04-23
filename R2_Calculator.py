###########################################################################                                                                         #
# Responsible for pre-processing sensor data and building, training, &    #
# evaluating the model.                                                   #
#                                                                         #
# required input:                                                         #
###########################################################################

import pandas as pd
import numpy as np

import tensorflow as tf
from keras import backend as K

num_cores = 4
CPU=1
GPU=0

num_CPU = 1
num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads = num_cores,\
                        inter_op_parallelism_threads = num_cores,\
                        allow_soft_placement         = True,\
                        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU}
                        )
session = tf.Session(config=config)
K.set_session(session)

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

import os
import sys

DEBUG = False

###########################################################################
#                               R2Callback                                #
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
#                             makeRNN()                                              #
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
    if DEBUG :
        print("making RNN")

    model = Sequential()

    ####################################################################
    # The first layer needs the batch_input_shape as a parameter,      #
    # and so we need to initialize it separately from the rest.        #
    ####################################################################

    if lstm_flag:
        model.add(LSTM(units               = units[0],
                     activation            = activations[0],
                     dropout               = dropouts[0],
                     recurrent_dropout     = recurrent_dropouts[0],
                     return_sequences      = True,
                     unit_forget_bias      = unit_forget_bias,
                     kernel_initializer    = kernel_initializer,
                     recurrent_initializer = recurrent_initializer,
                     batch_input_shape     = (batch_size,1,num_attributes)
                ))
    else:
        model.add(
                GRU(units                 = units[0],
                    activation            = activations[0],
                    dropout               = dropouts[0],
                    recurrent_dropout     = recurrent_dropouts[0],
                    return_sequences      = True,
                    kernel_initializer    = kernel_initializer,
                    recurrent_initializer = recurrent_initializer,
                    batch_input_shape     = (batch_size,1,num_attributes)
                ))

    info = zip(units[1:],dropouts[1:],activations[1:],recurrent_dropouts[1:])
    for i in info: #layer,dropout,activation in info:
        if DEBUG:
            print("units: {},dropout: {}, activation: {}".format(i[0],i[1],i[2]))
        if lstm_flag:
            model.add(
                LSTM(units                 = i[0],
                     activation            = i[2],
                     dropout               = i[1],
                     recurrent_dropout     = i[3],
                     return_sequences      = True,
                     unit_forget_bias      = unit_forget_bias,
                     kernel_initializer    = kernel_initializer,
                     recurrent_initializer = recurrent_initializer,
                ))
        else:
            model.add(
                GRU(units                 = i[0],
                    activation            = i[2],
                    dropout               = i[1],
                    recurrent_dropout     = i[3],
                    return_sequences      = True,
                    kernel_initializer    = kernel_initializer,
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
#                         testTrainSplit()                                           #
# Split the data into testing and validation sets, making sure to return an even     #
# number of batches in the result.                                                   #
# Parameters:                                                                        #
#     X          : the feature data                                                  #
#     y          : the target data                                                   #
#     batch_size : the batch size                                                    #
######################################################################################

def testTrainSplit(X,y,batch_size):
    batches = np.floor(data.shape[0]/batch_size)
    split = int(np.floor(batches * (1/2))) * int(batch_size)
    return X[:split,:],y[:split,:],X[split:,:],y[split:,:]

######################################################################################
#                             getData()                                              #
# Get the data out of the files and in a nice format.                                #
# Parameters:                                                                        #
#     files : a list of files with the data in it                                    #
#     DEBUG : a DEBUG flag.                                                          #
######################################################################################

def getData(files):

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

        if DEBUG :
            print("full_data.shape: {}".format(full_data.shape))
            print("processed_data.shape: {}".format(processed_data.shape))
            print("processed_target.shape: {}".format(processed_targets.shape))

        data    = np.append(data, processed_data, axis=0)
        targets = np.append(targets, processed_targets, axis=0)
        batches = np.append(batches,[batch])

        np.savetxt('Data/Processed/' + relative_path, processed_data, delimiter=',', fmt = '%f')

    if DEBUG :
        print("data[0:10]: \n{}".format(data[0:10]))
        print("targets[0:10]: \n{}".format(targets[0:10]))
        print("batches: \n{}".format(batches))
    return data,targets,batches

######################################################################################
#                              Make some constants                                   #
# These constants will allow us to more clearly access and modify summary statistics #
######################################################################################

TRAIN_LOSS = 0
TRAIN_ACC  = 1
TEST_LOSS  = 2
TEST_ACC   = 3

#########################################################################################################
#                                    Evaluate Model                                                     #
# Use the stats across training epochs (best_stats) and the stats across samples for this epoch         #
# (sample stats) to ind out whether the most recent training epoch made a better model.                 #
# If so, save the model, update the best_stats, and return the update flag.                             #
#########################################################################################################

def evaluate_model(model,best_stats,overall_stats,model_name):
    if DEBUG :
        print('from evaluate model...')
        print("best_stats: \n{}\noverall_stats: \n{}\n".format(best_stats,overall_stats))
    updated = False
    for stat_place in [TRAIN_LOSS,TRAIN_ACC,TEST_LOSS,TEST_ACC]:
        diff = best_stats[stat_place] - overall_stats[stat_place]
        if diff > .01 :
            updated = True
            if save_mode : model.save("../data/DMproject/Models/{}/{}.hf".format(model_name,stat_place))
            print("best: {}, new: {}, diff: {}".format(best_stats   [stat_place],
                                                       overall_stats[stat_place],
                                                       diff))
            best_stats[stat_place] = overall_stats[stat_place]
            print("Saving a good training loss model: {}".format(overall_stats[stat_place]))
    if DEBUG :
        print('from evaluate, best_stats: \n{}'.format(best_stats))
    return updated

##################################################################################################################################
#                                                     __main__                                                                   #
##################################################################################################################################
if __name__=="__main__":

    parser = argparse.ArgumentParser(description = "LSTM.py",
        epilog = "Use this program to calculate some R2 values.",
        add_help = "How to use",
        prog = "python LSTM.py -f <files> [OPTIONAL PARAMETERS]")

    parser.add_argument("-f", "--files", nargs = "*", required = True ,
        help = "A list of input files. NOTE: A path with path expansion a la \'\\*\' is acceptable. ")

    parser.add_argument("-b", "--batch_size", default = 1, type = int,
        help = "The batch size.")

    parser.add_argument("-S","--save_mode",action="store_false",
        help = "Flag to run the mdoel in \"no save\" mode.")

    args = vars(parser.parse_args())


    files                   = args['files']
    batch_size              = args['batch_size']
    save_mode               = args['save_mode']

    num_attributes = 8
    num_targets    = 3

    DEBUG = True
    
    ##############################################
    #              Get the data                  #
    ##############################################
    print("Collecting and pre-processing data...")
    data,targets,samples = getData(files)
    num_samples = len(samples)
    print("Done.")

    
    '''
    ################################################################
    #                        Make the net                          #
    ################################################################
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


    model.compile(loss='mean_squared_error',optimizer="adam",metrics=["mae"])

    ###########################################################
    #          Make some statistics placeholders              #
    # These allow us to track whether our model is improving  #
    # and collect the statistics across epochs
    ###########################################################

    epochs_without_update = 0
    converged = False
    best_stats = [1000000] * 4
    stats_list = [np.array([])] * 4

    '''

    ###################################################
    # Create the iterator for the directory structure #
    ###################################################

    R2s        = []
    models     = []
    bestR2     = 1
    best_model = 'None_None'

    #################################################################
    # Each sample potentially has a different length from the last. #
    # j is going to allow us to distinguish them from one another.  #
    #################################################################
    j = 0

    ##############################################################
    # Train the model one sample at a time, resetting the models #
    # state after each sample.                                   #
    # samples is a list of the lengths of each of our samples    #
    ##############################################################
    for sample_size in samples:

        print('predicting for a new sample...')
        #####################################
        # Get the data for this sample      #
        #####################################
        X = data[j:j+int(sample_size),:]
        y = targets[j:j+int(sample_size),:]

        ##################################
        # aggregate the sample indicator #
        ##################################
        j += int(sample_size)

        ##################################################
        # Split the data into testing and training sets. #
        # NOTE: testTrainSplit should return data        #
        #       with an even number of batches in it.    #
        ##################################################
        X_train, y_train, X_test, y_test = testTrainSplit(X,y,batch_size)

        training_time_steps, num_features = X_train.shape
        num_targets  = y_train.shape[1]
        testing_time_steps = X_test.shape[0]


        ####################################################
        # Traverse the directory structure - dropout specs #
        ####################################################

        stuff = glob.iglob('../data/DMproject/Models/*')
          
        print("Done.")
        
        for model_architecture in stuff:

            print('predicting for a new architecture...')

            ######################################################################
            # Load each of the four model types and predict with each one.       #
            ######################################################################
            
            for model_type in [TRAIN_LOSS,TRAIN_ACC,TEST_LOSS,TEST_ACC]:

                model = keras.load_model("{}/{}.hf".format(model_arcjhitecture,model_type))
                print(model.summary())
                predict_values = np.array([]*3).reshape(0,3)

                ###################################################################
                # now we test the model on this sample, one batch at a time...    #
                # x_te_m/batch_size gives us the number of batches in this sample #
                ###################################################################

                num_testing_batches = int(testing_time_steps/batch_size)
                for i in range(num_testing_batches):

                    ################################################################
                    # We have to reshape the data to [batch_size,1,num_attributes] #
                    # and the targets to [batch_size,1,num_targets]                #
                    ################################################################
                    
                    X_te_batch = X_test[i*batch_size:(i+1)*batch_size].reshape(batch_size,1,num_features)
                    y_te_batch = y_test[i*batch_size:(i+1)*batch_size].reshape(batch_size,num_targets)
                    
                    predictions = model.predict_on_batch(X_te_batch)
                    predict_values = np.append(predict_values,[predictions])                  

                r2 = r2_score(y_test,predict_values)
                R2s.append(r2)
                models.append('{}_{}'.format(model_architecture,model_type))
                if best_r2 > r2 :
                    best_model = ('{}{}'.format(model_architecture,model_type))
                    best_r2    = r2

        data = pd.DataFrame(np.R2s,index=models)
        data.to_csv('R2s.csv')
