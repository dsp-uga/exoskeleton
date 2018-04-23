# dealing with 'None's:
# https://stackoverflow.com/questions/14060894/replacing-nones-in-a-python-array-with-zeroes

from adxl345 import ADXL345
from hmc5883l.HMC5883L import HMC5883L
from itg3200.ITG3200 import ITG3200
from Adafruit_ADS1x15 import ADS1115
import numpy as np
import time


# initializing sensors
accl = ADXL345()
gyro = ITG3200()
comp = HMC5883L()
adc = ADS1115()

numEMGs = 2
start = time.time()

while(1):
    axes = accl.get_axes()
    a = [axes['x'], axes['y'], axes['z']]
    g = list(gyro.read_data())
    c = list(comp.read_data())
    e = [adc.read_adc(i, gain=1) for i in range(numEMGs)]
    t = [time.time() - start]

    datapoint = [i if i is not None else 0 for i in a+g+c+e+t]
    datapoint = np.array(datapoint).reshape((1,12))

    print(datapoint)
    with open("recording.csv", "ab") as f:
        np.savetxt(f, datapoint, delimiter=",")
    time.sleep(0.001)

