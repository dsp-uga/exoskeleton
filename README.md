# Acceleration prediction
----
This project is a continuation of work done here: https://github.com/minimum-LaytonC/DMproject
(though no data, and little to no code from that project is used here)

We attempt to predict next state acceleration of the human wrist from current observations of acceleration, rotation, magnetic field, and electrical activity of the biceps and triceps. Our data comes from normal human activity, collected with a Raspberry Pi taped to my arm powered by a battery in my pocket.

In the previous project simpler algorithms were applied to a more limited dataset. The previous dataset included only acceleration and a single EMG on the biceps, and activity was limited to a fixed arm position with only one axis of motion. Here activity is unrestricted.

## Dependencies
This project uses the following packages, which you will need to install and configure in order to use this repository:

* [SciPy](https://scipy.org/install.html)
* [SciKitLearn](http://scikit-learn.org/stable/install.html)
* [Tensorflow](https://www.tensorflow.org/install/)
* [Keras](https://keras.io)

## Running an RNN Model
To test the software, navigate to the root directory. Inside of the root directory is a Data/ directory, where the data on which the model was trained is located. There is also the main driver of the repository, LSTM.py.

To see all of the available command line arguments for LSTM.py, run the following command:

    python LSTM.py -h

The required command line arguments include:

* -n : The number of units per layer
* -d : A list of layer dropout levels for each layer
* -r : A list of recurrent dropout levels for each layer
* -f : A list of input data files to use to train the model.
* -o : A path in which to place the output of the model results.
* -v : A list of activation functions to be used on each layer of the RNN.

Note that the length of the lists of units per layer, dropout levels, recurrent dropout levels, and activation functions should all be the same length.

With that in mind the following command will train, test, and record the model and it's epoch statistics:

    python LSTM.py -n 512 512 -d .4 .3 -v relu relu -r .6 .3 -f Data/* -g -b 50 -o ../Data

## Running a CNN Model

## Running an RL model

## License

## Contributors
