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
import random

from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense,BatchNormalization,LSTM,Input,GRU
from keras.callbacks import Callback

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

import os
import sys

DEBUG = False
DEBUGG = False
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
    if debug :
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
        if debug:
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
    batches = np.floor(X.shape[0]/batch_size)
    split = int(np.floor(batches * (1/2))) * int(batch_size)
    last = int(batches) * int(batch_size)
    trainX,trainY,testX,testY = X[:split],y[:split:],X[split:last,:],y[split:last,:]
    if DEBUGG :
        print('test_train_split')
        print('X.shape: {},y.shape: {}'.format(X.shape,y.shape))
        print('batches: {}, split: {}, last: {}, batch_size: {}'.format(batches,split,last,batch_size))
        print('trainX.shape: {},trainY.shape: {},testX.shape: {},testY.shape: {}'.format(trainX.shape,trainY.shape,testX.shape,testY.shape))
    return X[:split,:],y[:split,:],X[split:last,:],y[split:last,:]

######################################################################################
#                             getData()                                              #
# Get the data out of the files and in a nice format.                                #
# Parameters:                                                                        #
#     files : a list of files with the data in it                                    #
#     debug : a debug flag.                                                          #
######################################################################################

#def getData(files,debug,scaler):
def getData(files,debug,scaler,batch_size):    

    train = []
    test  = []
    data    = np.empty((0,num_attributes)) #initializing empty array with proper shape
    batches = np.array([])

    for f in files:

        #grab the relative pathname and we can save the data for later.
        relative_path = f[f.rfind('/')+1:]
        dat = np.loadtxt(f,delimiter = ',')[:,:-1]
        data = np.append(data,dat,axis=0)

        batch = dat.shape[0]

        batches = np.append(batches,[batch])

        if DEBUGG :
            print('f: {}'.format(f))
            print("dat.shape: {}, batch: ".format(dat.shape,batch))
            print('batches: {}'.format(batches))
            print('data.shape:{}'.format(data.shape))
    ###########################################################
    # The first three columns are targets, the rest features, #
    # and we're predicting the difference in each feature     #
    # from one time point to the next.                        #
    ###########################################################

    print('from getData(), data.shape: {}'.format(data.shape))
    data = scaler.fit_transform(data)
    
    #X = data[:-1,:]
    #y = data[1:,:3]-data[:-1,:3]


    j = 0 
    for i,batch in enumerate(batches) :
        dat = data[j:j+int(batch),:]
        X,y = dat[:-1,:],dat[1:,:3]-dat[:-1,:3]

        if DEBUGG :
            print('before testTrainSplit')
            print('j: {}, batch: {}'.format(j,batch))
            print('dat.shape: {}'.format(dat.shape))
            

        trainX,trainY,testX,testY = testTrainSplit(X,y,batch_size)

        if DEBUGG :
            print('after testTrainSplit')
            print('trainX.shape: {}, trainY.shape: {}, testX.shape: {},testY.shape: {}'.format(trainX.shape,trainY.shape,testX.shape,testY.shape))
        train += [(trainX,trainY)]
        test  += [(testX,testY)]
        j += int(batch)
            
    #return X,y,batches
    random.shuffle(train)
    random.shuffle(test)
    return train,test

######################################################################################
#                              Make some constants                                   #
# These constants will allow us to more clearly access and modify summary statistics #
######################################################################################

TRAIN_LOSS = 0
TEST_LOSS  = 1
TRAIN_R2   = 2
TEST_R2    = 3
#########################################################################################################
#                                    Evaluate Model                                                     #
# Use the stats across training epochs (best_stats) and the stats across samples for this epoch         #
# (sample stats) to ind out whether the most recent training epoch made a better model.                 #
# If so, save the model, update the best_stats, and return the update flag.                             #
#########################################################################################################

def evaluate_model(model,best_stats,epoch_stats,model_name):
    if DEBUG :
        print('from evaluate model...')
        print("best_stats: \n{}\nepoch_stats: \n{}\n".format(best_stats,epoch_stats))
    updated = False
    diff = np.array(best_stats) - np.array(epoch_stats)
    for stat_place in [TRAIN_LOSS,TEST_LOSS,TRAIN_R2,TEST_R2]:
        new_best_loss = stat_place in [TRAIN_LOSS,TEST_LOSS] and diff[stat_place] > .001
        new_best_acc  = stat_place in [TRAIN_R2  ,TEST_R2  ] and .001 > diff[stat_place] 
        if new_best_loss or new_best_acc :
            updated = True
            if save_mode : model.save("../data/DMproject/Models/{}/{}.hf".format(model_name,stat_place))
            print("best: {}, new: {}, diff: {}".format(best_stats   [stat_place],
                                                       epoch_stats[stat_place],
                                                       diff))
            best_stats[stat_place] = epoch_stats[stat_place]
            print("Saving a {} model: {}".format(stat_place,epoch_stats[stat_place]))
    if DEBUG :
        print('from evaluate, best_stats: \n{}'.format(best_stats))
    return updated

##################################################################################################################################
#                                                     __main__                                                                   #
##################################################################################################################################
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

    parser.add_argument("-a", "--num_attributes", type = int, default = 11,
        help = "The number of attributes used to train the model. DEFAULT: 6")

    parser.add_argument("-t", "--num_targets", type = int, default = 3,
        help = "The number of target values the model will predict.")

    parser.add_argument("-z", "--debug", action="store_true",
        help = "The debug flag to run the program in debug mode. DEFAULT: False.")

    parser.add_argument("-b", "--batch_size", default = 1, type = int,
        help = "The debug flag to run the program in debug mode. DEFAULT: False.")

    parser.add_argument("-q","--quiet",action="store_true",
        help = "Flag to run TensorFlow in quiet mode.")

    parser.add_argument("-R","--repeat", required = True,
        help="The repeat number for running this particular model configuration.")

    parser.add_argument("-S","--save_mode",action="store_false",
        help = "Flag to run the mdoel in \"no save\" mode.")

    args = vars(parser.parse_args())

    units                   = args['units']
    dropouts                = args['dropouts']
    activations             = args['activations']
    recurrent_dropouts      = args['recurrent_dropouts']
    epochs                  = args['epochs']
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
    save_mode               = args['save_mode']
    repeat                  = args['repeat']
    #The scaler that we will use later
    scaler = RobustScaler() if standard_scaler else MinMaxScaler()

    ####################################################
    # Making the model name                            #
    ####################################################
    neuron_string = ''
    for n in units:
        neuron_string += "_{}".format(str(n))
    
    drop_string = ''
    for d in dropouts:
        drop_string += "_{}".format(str(d))

    rdrop_string = ''
    for rd in recurrent_dropouts:
        rdrop_string += "_{}".format(str(rd))

    model_name = "M{}{}{}".format(neuron_string,drop_string,rdrop_string)
    ######################################################################
    #             Creating the directory structure                       #
    ######################################################################
    try:
        original_umask = os.umask(0)
        os.makedirs("../data/DMproject/Models/{}".format(model_name),mode=0o777)
    except :
        print("Couldn't make directory: ../data/DMproject/Models/{}".format(model_name))
        print(sys.exc_info()[0])
    finally:
        os.umask(original_umask)

    try:
        original_umask = os.umask(0)
        os.makedirs("../data/DMproject/Results/{}".format(model_name),mode=0o777)
    except :
        print("Couldn't make directory: Data/Results/{}".format(model_name))
        print(sys.exc_info()[0])
    finally:
        os.umask(original_umask)

    ##############################################
    #              Get the data                  #
    ##############################################
    print("Collecting and pre-processing data...")
    #data,targets,samples = getData(files,debug,scaler)
    train,test = getData(files,debug,scaler,batch_size)
    #num_samples = len(samples)
    num_training_samples = len(train)
    num_testing_samples  = len(test)
    print("Done.")
    print("Specifying the model...")

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

    print("Done.")
    print(model.summary())

    model.compile(loss='mean_squared_error',optimizer="adam")

    ###########################################################
    #          Make some statistics placeholders              #
    # These allow us to track whether our model is improving  #
    # and collect the statistics across epochs
    ###########################################################

    epochs_without_update = 0
    converged = False
    best_stats = [1000000,1000000,-1000000,-1000000]
    stats_list = [np.array([])] * 4

    #########################################################################
    #                       Train the Model                                 #
    #                                                                       #
    # Training by hand is a little ugly.                                    #
    # We run each epoch separately, and inside of it we'll run the data     #
    # through the model on a forward pass in batches.                       #
    # We'll reset the state of the net on each new sample.                  #
    #########################################################################
    epoch = 0
    while not converged:

        epoch += 1
        ##################################################################################
        # We'll collect our predictions on each batch and use them to calculate an TEST_R2    #
        # score for each epoch. We'll also need to collect the y_true values for batches #
        ##################################################################################
        train_y_pred = np.array([]*3).reshape(0,3)
        train_y_true  = np.array([]*3).reshape(0,3)
        
        test_y_pred = np.array([]*3).reshape(0,3)
        test_y_true  = np.array([]*3).reshape(0,3)
        
        ##################################################################################
        # We'll use epoch_stats to aggregate our results for this epoch across samples   #
        ##################################################################################
        epoch_stats = [0]*4
        print("training epoch {}...".format(epoch))

        ##############################################################
        # Train the model one sample at a time, resetting the model  #
        # state after each sample.                                   #
        ##############################################################
        for sample in train:
            X_train,y_train = sample

            ##############################################################################
            # We'll use train_loss to aggregate our results for this particular sample    #
            ##############################################################################
            train_loss = 0
            print("training on a new sample...")

            ##############################################################################
            # We'll want these feature shapes for lining up arrays later.                #
            ##############################################################################
            training_time_steps, num_features = X_train.shape
            num_targets  = y_train.shape[1]
            
            ##############################
            # This is for a progress bar #
            ##############################
            k = 1

            #######################################################################
            # now we train the model on this sample, one batch at a time...       #
            # time_steps/batch_size gives us the number of batches in this sample #
            #######################################################################
            num_training_batches = int(training_time_steps / batch_size)
            if DEBUGG : print("num_trainig_batches: {}".format(num_training_batches))

            for i in range(int(num_training_batches)):

                ################################################################
                # We have to reshape the data to [batch_size,1,num_attributes] #
                # and the targets to [batch_size,num_targets]                #
                ################################################################
                X_tr_batch = X_train[i*batch_size:(i+1)*batch_size].reshape(batch_size,1,num_features)
                y_tr_batch = y_train[i*batch_size:(i+1)*batch_size].reshape(batch_size,num_targets)

                mse = model.train_on_batch(X_tr_batch,y_tr_batch)
                train_loss += mse

                #Print the progress bar.
                if i%150 == 0 :
                    print("-"*k +">",end="\r")
                    k += 1

                if DEBUGG :
                    print("for this batch, mse: {}".format(mse))
                    print('after training on batch, test_loss: {}'.format(train_loss))
            print("\n")
            
            ##############################################################################################
            # Average out the test loss for this testing sample and add it to the epoch stats            #
            ##############################################################################################
            sample_train_loss = train_loss / num_training_batches
            epoch_stats[TRAIN_LOSS] += sample_train_loss 
            if DEBUGG :
                print('after most recent training sample, epoch_stats[TRAIN_LOSS]: {}'.format(epoch_stats[TRAIN_LOSS]))
                
            ##############################################################################################
            # Make sure to reset the state and update the epoch_stats after each training sample.        #
            # NOTE : Our metrics are each averaged across every time point in the batch, so we want to   #
            #        make sure that our sample stats are each averaged across the training batches also. #
            ##############################################################################################
            model.reset_states()

        
        ##############################################################
        # Test the model one sample at a time, resetting the model  #
        # state after each sample.                                   #
        ##############################################################
        for sample in test:
            X_test,y_test = sample
            
            ##############################################################################
            # We'll use test_loss to aggregate our results for this particular sample #
            ##############################################################################
            test_loss = 0
            print("testing on a new sample...")

            ##############################################################################
            # We need these feature shapes for lining up batches and other nparrays.     #
            ##############################################################################
            testing_time_steps, num_features = X_test.shape
            num_targets  = y_test.shape[1]

            if DEBUGG :
                print('testing_time_steps: {}, num_features: {}, num_targets: {}'.format(testing_time_steps,num_features,num_targets))
            ##############################
            # This is for a progress bar #
            ##############################
            k = 1

            #######################################################################
            # now we test the model on this sample, one batch at a time...       #
            # time_steps/batch_size gives us the number of batches in this sample #
            #######################################################################
            num_testing_batches = int(testing_time_steps / batch_size)
            if DEBUG : print("num_testig_batches: {}".format(num_testing_batches))

            for i in range(int(num_testing_batches)):

                ################################################################
                # We have to reshape the data to [batch_size,1,num_attributes] #
                # and the targets to [batch_size,1,num_targets]                #
                ################################################################
                X_te_batch = X_test[i*batch_size:(i+1)*batch_size].reshape(batch_size,1,num_features)
                y_te_batch = y_test[i*batch_size:(i+1)*batch_size].reshape(batch_size,num_targets)

                mse = model.test_on_batch(X_te_batch,y_te_batch)
                test_loss += mse

                #Print the progress bar.
                if i%150 == 0 :
                    print("-"*k +">",end="\r")
                    k += 1
            print("\n")

            ##############################################################################################
            # Make sure to reset the state and update the epoch_stats after each training sample.        #
            # NOTE : Our metrics are each averaged across every time point in the batch, so we want to   #
            #        make sure that our sample stats are each averaged across the training batches also. #
            ##############################################################################################
            model.reset_states()
            ##############################################################################################
            # Average out the test loss for this testing sample and add it to the epoch stats            #
            ##############################################################################################
            sample_test_loss = test_loss / num_testing_batches
            epoch_stats[TEST_LOSS] += sample_test_loss 
            if DEBUGG :
                print('after most recent training sample, epoch_stats[TEST_LOSS]: {}'.format(epoch_stats[TEST_LOSS]))


        ##############################################################
        # Make some predictions                                      #
        ##############################################################

        for training_sample in train:

            X_train,y_train = sample
            
            training_time_steps, num_features = X_train.shape
            num_targets  = y_train.shape[1]
            num_training_batches = int(training_time_steps / batch_size)

            pred = model.predict_on_batch(X_te_batch)
            pred.reshape(batch_size,num_targets)
            
            train_y_pred = np.append(train_y_pred,pred      ,axis=0)
            train_y_true = np.append(train_y_true,y_te_batch,axis=0)
            
        for testing_sample in test:
            
            X_test,y_test = sample
            
            testing_time_steps, num_features = X_test.shape
            num_targets  = y_test.shape[1]
            num_testing_batches = int(testing_time_steps / batch_size)

            pred = model.predict_on_batch(X_te_batch)
            pred.reshape(batch_size,num_targets)
            
            test_y_pred = np.append(test_y_pred,pred      ,axis=0)
            test_y_true = np.append(test_y_true,y_te_batch,axis=0)
            
        ##########################################################################################
        # Collect and aggregate the stats for the epoch                                          #
        # Note: The stats were calculated as an average for each sample, so we want to make sure #
        #       that the stats for the epoch are an average across the samples as well.          #
        ##########################################################################################
        
        test_r2  = r2_score(test_y_true ,test_y_pred)
        train_r2 = r2_score(train_y_true,train_y_pred)
        
        epoch_stats[TRAIN_LOSS] /= num_training_samples 
        epoch_stats[TEST_LOSS]  /= num_testing_samples
        epoch_stats[TRAIN_R2] = train_r2
        epoch_stats[TEST_R2]  = test_r2
        
        if DEBUGG :
            for stat_place in [TRAIN_LOSS,TEST_LOSS,TRAIN_R2,TEST_R2]:
                print('after averaging batch results, stat_place: {}, epoch_stats[stat_place]: {}'.format(stat_place,epoch_stats[stat_place]))

        if DEBUGG :
            print('before updating: stats_list: \n{}\nepoch stats: \n{}'.format(stats_list,epoch_stats))
        for stat_place in [TRAIN_LOSS,TEST_LOSS,TRAIN_R2,TEST_R2]:
            stats_list[stat_place] = np.append(stats_list[stat_place],[epoch_stats[stat_place]])
        if DEBUGG :
            print('after updating: stats_list: \n{}\nepoch stats: \n{}'.format(stats_list,epoch_stats))

        updated = evaluate_model(model,best_stats,epoch_stats,model_name)

        if not updated :
            epochs_without_update += 1
        else :
            epochs_without_update = 0
        if epochs_without_update == 10 : converged = True
    
    '''
    #########################################################################
    #                       Train the Model                                 #
    #                                                                       #
    # Training by hand is a little ugly.                                    #
    # We run each epoch separately, and inside of it we'll run the data     #
    # through the model on a forward pass in batches.                       #
    # We'll reset the state of the net on each new sample.                  #
    #########################################################################
    epoch = 0
    while not converged:

        epoch += 1

        #################################################################
        # Each sample potentially has a different length from the last. #
        # j is going to allow us to distinguish them from one another.  #
        #################################################################
        j = 0

        ##################################################################################
        # We'll use epoch_stats to aggregate our results for this epoch across samples   #
        ##################################################################################
        epoch_stats = [0]*3
        print("training epoch {}...".format(epoch))

        ##############################################################
        # Train the model one sample at a time, resetting the models #
        # state after each sample.                                   #
        # samples is a list of the lengths of each of our samples    #
        ##############################################################
        for sample_size in samples:

            ##############################################################################
            # We'll use sample_stats to aggregate our results for this particular sample #
            ##############################################################################
            sample_stats = [0]*3
            print("training on a new sample...")
            if DEBUGG :
                print("samle_size: {}".format(sample_size))

            #####################################
            # Get the data for this sample      #
            #####################################
            X = data[j:j+int(sample_size),:]
            y = targets[j:j+int(sample_size),:]

            if DEBUG :
                print("X.shape: {}".format(X.shape))
                print("y.shape: {}".format(y.shape))

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

            ##############################
            # This is for a progress bar #
            ##############################
            k = 1

            #######################################################################
            # now we train the model on this sample, one batch at a time...       #
            # time_steps/batch_size gives us the number of batches in this sample #
            #######################################################################
            num_training_batches = int(training_time_steps / batch_size)
            if DEBUG : print("num_trainig_batches: {}".format(num_training_batches))

            for i in range(int(num_training_batches)):

                ################################################################
                # We have to reshape the data to [batch_size,1,num_attributes] #
                # and the targets to [batch_size,1,num_targets]                #
                ################################################################
                X_tr_batch = X_train[i*batch_size:(i+1)*batch_size].reshape(batch_size,1,num_features)
                y_tr_batch = y_train[i*batch_size:(i+1)*batch_size].reshape(batch_size,num_targets)

                mse = model.train_on_batch(X_tr_batch,y_tr_batch)
                sample_stats[TRAIN_LOSS] += mse

                #Print the progress bar.
                if i%150 == 0 :
                    print("-"*k +">",end="\r")
                    k += 1

                if DEBUGG :
                    print("for this batch, mse: {}".format(mse))
                    for stat_place in [TRAIN_LOSS,TEST_LOSS]:
                        print('after training on batch, stat_place: {}, sample_stats[stat_place]: {}'.format(stat_place,sample_stats[stat_place]))
            print("\n")

            ##############################################################################################
            # Make sure to reset the state and update the epoch_stats after each training sample.        #
            # NOTE : Our metrics are each averaged across every time point in the batch, so we want to   #
            #        make sure that our sample stats are each averaged across the training batches also. #
            ##############################################################################################
            model.reset_states()

            #Reset k for an accurate progress bar. 
            k=1

            print("testing on new sample...")

            preds = np.array([]*3).reshape(0,3)
            ###################################################################
            # now we test the model on this sample, one batch at a time...    #
            # x_te_m/batch_size gives us the number of batches in this sample #
            ###################################################################
            num_testing_batches = int(testing_time_steps/batch_size)
            if DEBUG :
                print('testing_time_steps: {}'.format(testing_time_steps))
            for i in range(num_testing_batches):

                ################################################################
                # We have to reshape the data to [batch_size,1,num_attributes] #
                # and the targets to [batch_size,1,num_targets]                #
                ################################################################
                X_te_batch = X_test[i*batch_size:(i+1)*batch_size].reshape(batch_size,1,num_features)
                y_te_batch = y_test[i*batch_size:(i+1)*batch_size].reshape(batch_size,num_targets)

                mse  = model.test_on_batch(X_te_batch,y_te_batch)
                pred = model.predict_on_batch(X_te_batch)
                
                pred.reshape(batch_size,3)
                
                preds = np.append(preds,pred,axis=0)
                
                sample_stats[TEST_LOSS] += mse

                #Print the progress bar
                if i%150 == 0 :
                    print("-"*k +">",end="\r")
                    k += 1

                if DEBUGG :
                    print("for this batch, mse: {}".format(mse))
                    for stat_place in [TRAIN_LOSS,TEST_LOSS]:
                        print('after training on batch, stat_place: {}, sample_stats[stat_place]: {}'.format(stat_place,sample_stats[stat_place]))
            ##############################################################################################
            # Make sure to reset the state and update the epoch_stats after each testing  sample.        #
            # NOTE : Our metrics are each averaged across every time point in the batch, so we want to   #
            #        make sure that our sample stats are each an average of the metrics over all of the  #
            #        the testing batches also.  #
            ##############################################################################################
            model.reset_states()

            
            r2 = r2_score(y_test,np.array(preds))
            
            epoch_stats[TEST_LOSS] += sample_stats[TEST_LOSS] / num_testing_batches
            epoch_stats[R2] = r2
            epoch_stats[TRAIN_LOSS] += sample_stats[TRAIN_LOSS]/num_training_batches
        
            if DEBUGG :
                for stat_place in [TRAIN_LOSS,TEST_LOSS,R2]:
                    print('after averaging batch results, stat_place: {}, epoch_stats[stat_place]: {}'.format(stat_place,epoch_stats[stat_place]))

        ##########################################################################################
        # Collect and aggregate the stats for the epoch                                          #
        # Note: The stats were calculated as an average for each sample, so we want to make sure #
        #       that the stats for the epoch are an average across the samples as well.          #
        ##########################################################################################

        if DEBUGG : print('num_sample: {}'.format(num_samples))
        for stat in [TRAIN_LOSS, TEST_LOSS]:
            epoch_stats[stat] /= num_samples

        if DEBUGG :
            print('before updating: stats_list: \n{}\nepoch stats: \n{}'.format(stats_list,epoch_stats))
        for stat_place in [TRAIN_LOSS,TEST_LOSS,R2]:
            stats_list[stat_place] = np.append(stats_list[stat_place],[epoch_stats[stat_place]])
        if DEBUGG :
            print('after updating: stats_list: \n{}\nepoch stats: \n{}'.format(stats_list,epoch_stats))

        updated = evaluate_model(model,best_stats,epoch_stats,model_name)

        if not updated :
            epochs_without_update += 1
        else :
            epochs_without_update = 0
        if epochs_without_update == 10 : converged = True
    '''
    stats_list         = np.array(stats_list)
    print('stats_list.shape: {}'.format(stats_list.shape))
    summary_data       = stats_list.T
    summary_data_frame = pd.DataFrame(summary_data,columns=['tr_mse','te_mse','r2_train','r2_test'])

    print(summary_data_frame)
    if save_mode: summary_data_frame.to_csv("../data/DMproject/Results/{}/{}.csv".format(model_name,repeat),index=False)
