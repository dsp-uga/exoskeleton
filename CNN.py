###########################################################################                                                                         #
# Responsible for pre-processing sensor data and building, training, &    #
# evaluating the model. Preprocessing code has been taken from LSTM       # 
#                                                                         #
#                                                                         #
###########################################################################

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

#necessary improts
import os
import sys
import random
import warnings



from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input,Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint


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



num_attributes=11
num_targets=3

###########################################################################                                                                         #
# Responsible for forcing Keras run on CPU.                               #
#                                                                         #
# required input:                                                         #
###########################################################################



import tensorflow as tf
from keras import backend as K

num_cores = 4
num_CPU = 0
num_GPU = 1

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


######################################################################################
#                             getData()                                              #
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
        print(f)
        #grab the relative pathname and we can save the data for later.
        relative_path = f[f.rfind('/')+1:]

        full_data = np.loadtxt(f,delimiter = ',')

        ###########################################################
        # The first three columns are targets, the rest features, #
        # and we're predicting the difference in each feature     #
        # from one time point to the next.                        #
        ###########################################################

        processed_data    = full_data
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
    scalar_train = MinMaxScaler()
    scalar_train = scalar_train.fit(data)
    data = scalar_train.transform(data)
    scalar_targets = MinMaxScaler()
    scalar_targets = scalar_targets.fit(targets)
    targets= scalar_targets.transform(targets)
    
    if debug :
        print("data[0:10]: \n{}".format(data[0:10]))
        print("targets[0:10]: \n{}".format(targets[0:10]))
        print("batches: \n{}".format(batches))
    return data,targets,batches


######################################################################################
#                             split_array()                                          #
# splits data into chunks , each array will be of same sizest.                       #
# Parameters:                                                                        #
#     files : a list of files with the data in it                                    #
#     debug : a debug flag.                                                          #
######################################################################################


def split_array(x,split_size=8,final_shape=8,test_only=False):
    counter = 0
    block = list();
    temp = list();
    print(x.shape)
    for i in range(0,x.shape[0]):
        block.append(x[i])
        counter = counter + 1
        if counter == split_size:
            block_list = list();
            for i in range(0,final_shape):
                block_list.append(block[-1])
            block = np.array(block_list)
            block = np.resize(block_list,(final_shape,final_shape))
            temp.append(block)
            counter = 0
            block = list();
    return np.array(temp)

files=["/home/crumpler/DMproject-master/Data/Clean/layton_1.1.csv"]

###########################################################################                                                                         #
# Splits the data to image slice and generats mask required by the symbol #
#  branch                                                                 #
#                                                                         #
###########################################################################
data,targets,samples = getData(files,True)
print(targets.shape)
data = split_array(data,8)
data = np.array(data)
targets = split_array(targets,8,8)
last_value = targets[:,7,:]
last_value = last_value[:,0:3]
targets = np.array(targets)
data =data.reshape((data.shape[0],data.shape[1],data.shape[2],1))
targets =targets.reshape((targets.shape[0],targets.shape[1],targets.shape[2],1))
targets_mask = np.ma.masked_where(targets >0, targets).mask
print(data.shape)
print("Targets:{}".format(last_value.shape))
print("Targets_mask:{}".format(targets_mask.shape))
print(targets_mask.shape)
print(np.count_nonzero(targets_mask))




#####################################################################################
#                            generate Unet Model                                    #
# Produces a unet model depending on the optimizer function                         #
# Parameters:                                                                       #
#     optimizer : the optimizer that needs to be selected                           #
#     width: the width parameter to be selected                                     #
#     height: the height of slice to be selected                                    #
#https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277?scriptVersionId=2164855 #
#Copied the basic unet structure , used our own filter depth and additionally       #
#modified UNET structure to add two output branches  and two loss function          #
#####################################################################################

def get_unet(optimizer="adagrad",width=8,height=8):
    # Build U-Net model
    inputs = Input((width, height, 1))
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (p3)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
    c9 = Dropout(0.2)(c9)
    outputs = Conv2D(1, (1, 1), activation='linear',name="magnitudes") (c9)
    outputs_1 = Conv2D(1, (1, 1), activation='sigmoid',name="symbols") (c9)

    model = Model(inputs=[inputs], outputs=[outputs,outputs_1])
    model.compile(optimizer=optimizer, loss=["mae","binary_crossentropy"],metrics=['accuracy','mse'])
    model.summary()
    return model
model = get_unet()



​
###########################################################################                                                                         #
# Print the actual and predicted target values                            #
#                                                                         #
###########################################################################
​
result = model.predict(data)[0]
values = np.squeeze(result)
data_entry = list()
data_value=list()
for i in range(0,values.shape[0]):
    data_entry.append(np.round(values[i][7][0],3))
    data_value.append(np.round(targets[i][7][0][0],3))
    print("Actual:{},{},{}".format(values[i][7][0],values[i][7][1],values[i][7][2]))
    print("Target: {},{},{}".format(targets[i][7][0][0],targets[i][7][1][0],targets[i][7][2][0]))

print(r2_score(data_entry,data_value)) #print the r2 score


