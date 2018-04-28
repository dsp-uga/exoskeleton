from adxl345 import ADXL345
#from itg3200.ITG3200 import ITG3200
import sys, time
import zmq
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from keras.models import load_model


#################################################
# saving on ctrl-c, exit_gracefully framework code found here:
# https://stackoverflow.com/questions/18114560/python-catch-ctrl-c-command-prompt-really-want-to-quit-y-n-resume-executi
#################################################
import signal
def exit_gracefully(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    print("saving model")
    model.save("rl_model.hdf5")
    print("model saved. exiting.")

    sys.exit(0)

original_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, exit_gracefully)


#################################################
# initializing accelerometer and zeroMQ subscriber
#################################################
accl = ADXL345()

port = "5556"

# Socket to talk to server
context = zmq.Context()
context.setsockopt(zmq.CONFLATE,1) #only keep most recent message
context.setsockopt(zmq.SUBSCRIBE, b"1000")

socket = context.socket(zmq.SUB)

print("connecting to port...")
socket.connect ("tcp://192.168.43.164:" + port)


#################################################
# Keras reinforcement learning model
# heavily based on this tutuorial:
# http://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
# sections of the training loop are adapted from there as well
#################################################
print("defining model...")
import os.path
if(os.path.isfile("rl_model.hdf5")):
    model = load_model("rl_model.hdf5")
else:
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, 15)))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

y = 0.95
eps = 1
decay_factor = 0.999
eps *= decay_factor
actions = np.array([-2, -1, 0, 1, 2])


#################################################
# Motor control
#################################################
print("setting up motor control...")
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)

pwmPin = 14
dirPin = 15

GPIO.setup(pwmPin,GPIO.OUT)
GPIO.setup(dirPin,GPIO.OUT)

pwm = GPIO.PWM(14, 1000)
pwm.start(0)

def act(action):
    if action == 2:
        GPIO.output(dirPin, GPIO.HIGH)
        pwm.ChangeDutyCycle(50)
    if action == 1:
        GPIO.output(dirPin, GPIO.HIGH)
        pwm.ChangeDutyCycle(25)
    if action == 0:
        pwm.ChangeDutyCycle(0)
    if action == -1:
        GPIO.output(dirPin, GPIO.LOW)
        pwm.ChangeDutyCycle(25)
    if action == -2:
        GPIO.output(dirPin, GPIO.LOW)
        pwm.ChangeDutyCycle(50)

print("entering loop.")
i = 0
first = True
selected = []
data = np.array([]).reshape((0,14))
while(True):
    dstring = socket.recv()[5:]
    axes = accl.get_axes()
    acc = [axes['x'], axes['y'], axes['z']]
    #g = list(gyro.read_data())
    #a = [-1] #robot's action, to be implemented

    robot_data = [i if i is not None else 0 for i in acc]

    dstring = socket.recv()[5:]
    human_data = np.fromstring(dstring, dtype=np.float64)

    datapoint = np.append(human_data, robot_data)
    data = np.append(data, datapoint.reshape((1,14)), axis=0)
    datapoint = (datapoint-np.mean(data,axis=0))*np.std(data)
    #print(datapoint)


    # select action either randomly or using predicted reward
    if np.random.random() < eps:
	# sampling from a normal distribution centered at 2 (0 movement action)
        a = np.round(np.clip(np.random.normal(2, 1, 1), 0, 4)).astype(int)[0]

    else:
        # predict reward from 5 available actions
        state_actions = np.append(np.tile(datapoint,(5,1)), actions.reshape((5,1)), axis=1)
        preds = model.predict(state_actions)
        #print(preds)
        a = np.argmax(preds)
        selected.append(actions[a])
    # apply selected action
    act(actions[a])

    i+=1

    if(i%10 == 0):
        print("\nactions:  "+str(actions[a])+"    epsilon:  "+str(eps))#+"    avg_act:  "+str(np.mean(selected)))
        print("human_data:\t" + str(human_data[:3]) + "\nrobot_data:\t"+str(robot_data[:3]))
        if(i%10000==0):
            print("\n\nsaving model")
            model.save("rl_model.hdf5")

    if(not first):
        accl_error = human_data[:2] - robot_data[:2]
        r = -(accl_error.dot(accl_error) + np.abs(actions[a])*3)
        # learning to map from state-actions to reward
        new_state_actions = np.append(np.tile(previous_datapoint,(5,1)), actions.reshape((5,1)), axis=1)
        target = r + y * np.max(model.predict(new_state_actions))
        state_actions = np.append(np.tile(datapoint,(5,1)), actions.reshape((5,1)),axis=1)
        target_vec = model.predict(state_actions)
        target_vec[a] = target

        model.fit(state_actions, target_vec.reshape(5), epochs=2, verbose=0)



    else:
        first = False
    eps*=decay_factor
    previous_datapoint = datapoint
    time.sleep(0.01)
