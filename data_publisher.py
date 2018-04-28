# dealing with 'None's:
# https://stackoverflow.com/questions/14060894/replacing-nones-in-a-python-array-with-zeroes

# publishing with zmq:
# http://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html
# https://stackoverflow.com/questions/38256936/last-message-only-option-in-zmq-subscribe-socket

from adxl345 import ADXL345
from hmc5883l.HMC5883L import HMC5883L
from itg3200.ITG3200 import ITG3200
from Adafruit_ADS1x15 import ADS1115
import numpy as np
import time
import zmq

# creating publishing socket
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:" + port)
topic = "1000"

# initializing sensors
accl = ADXL345()
gyro = ITG3200()
comp = HMC5883L()
adc = ADS1115()

numEMGs = 2

while(1):
    axes = accl.get_axes()
    a = [axes['x'], axes['y'], axes['z']]
    g = list(gyro.read_data())
    c = list(comp.read_data())
    e = [adc.read_adc(i, gain=1) for i in range(numEMGs)]
    
    datapoint = [i if i is not None else 0 for i in a+g+c+e]
    datapoint = np.array(datapoint)

    print(datapoint)
    socket.send(topic + " " + datapoint.tostring())
    time.sleep(0.001)

