import sys
import zmq
import numpy as np
import time

port = "5556"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)
    
if len(sys.argv) > 2:
    port1 =  sys.argv[2]
    int(port1)

# Socket to talk to server
context = zmq.Context()
context.setsockopt(zmq.CONFLATE,1) #only keep most recent message
context.setsockopt(zmq.SUBSCRIBE, "1000")

socket = context.socket(zmq.SUB)

print("connecting to port...")
socket.connect ("tcp://192.168.43.164:" + port)

if len(sys.argv) > 2:
    socket.connect ("tcp://localhost:%s" % port1)

print("listening...")
while True:
    dstring = socket.recv()[5:]
    data = np.fromstring(dstring, dtype=np.float64)
    print(data)
    time.sleep(0.02)

