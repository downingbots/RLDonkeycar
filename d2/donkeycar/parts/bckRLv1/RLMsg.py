'''
# makes JSON classes for each message
 https://docs.python.org/3.3/library/json.html#encoders-and-decoders

ETHERNET


The address needs to be unique for each Raspberry Pi. And the gateway is not needed if there is no router with internet access connected.

So it looks something like this.

auto eth0
iface eth0 inet static
    address 192.168.12.7
    netmask 255.255.255.0
    gateway 192.168.12.254

https://www.raspberrypi.org/learning/networking-lessons/lessons/
http://zeromq.org/  => message queues
https://wiki.python.org/moin/TcpCommunication

https://stackoverflow.com/questions/49294156/python-3-6-zeromq-pyzmq-asyncio-pub-sub-hello-world
https://openwsn.atlassian.net/wiki/spaces/OW/pages/113475628/Sending+Receiving+messages+on+the+EventBus+using+ZMQ


'''
# Task worker
# Connects PULL socket to tcp://localhost:5557
# Collects workloads from ventilator via that socket
# Connects PUSH socket to tcp://localhost:5558
# Sends results to sink via that socket
#
# Author: Lev Givon <lev(at)columbia(dot)edu>

import sys
import time
import zmq
import numpy as np
from PIL import Image
import donkeycar as dk
import pickle


from threading import Thread
import os
# dependency, not in stdlib
from netifaces import interfaces, ifaddresses, AF_INET



class MessageServer():

  def __init__(self, portid = 5557, nonblocking = False):
    global receiver, blocking, poll
    context = zmq.Context()
    blocking = (not nonblocking)

    if (blocking):
      # Socket to receive messages on
      receiver = context.socket(zmq.PULL)
      # receiver = context.socket(zmq.REP)
      addr = "tcp://*:%d" % portid
      receiver.bind(addr)
    else:
      global msgcnt, result, weights, weight_cnt
      msgcnt = -1
      result = cfg.EMERGENCY_STOP * 2
      weights = None
      weight_cnt = 0
      ctx = zmq.Context.instance()
      receiver = context.socket(zmq.PULL)
      addr = "tcp://*:%d" % portid
      listen_thread = Thread(target=self.listen, args=(addr,))
      listen_thread.start()
      
  def recv_msgtype(self ):
    global receiver

    msg_type = int(receiver.recv_string())
    return msg_type

  def get_msgcnt_result(self):
    global msgcnt, result
    return msgcnt, result

  def get_weight_cnt(self):
    global weight_cnt
    return weight_cnt

  def get_weights(self):
    global weights
    return weights

  def listen(self, addr):
    global msgcnt, result, weights, weight_cnt
    global receiver

    """listen for messages in thread """
    print(addr)
    receiver.bind(addr)

    while True:
        try:
            # print("LSNR BLOCK")
            mtype = self.recv_msgtype()
            # print("LSNR UNBLOCK : mtype = %d" % mtype)
            if mtype == cfg.MSG_RESULT:
              msgcnt, result = self.recvmsg_result()
              # print("Got msgcnt %d result %f" % (msgcnt, result))
            elif mtype == cfg.MSG_WEIGHTS:
              weightcnt, weights = self.recvmsg_weights()
              weight_cnt += 1 # must be set after weights global variable
              print("Got weightcnt %d" % weight_cnt)
              # print(weights)
            else:
              print("ERROR: got mtype %d" % mtype)
        except (KeyboardInterrupt, zmq.ContextTerminated):
            break


  def recvmsg_get_weights(self):
    # no additional content
    return None

  def recvmsg_weights(self):
    # weight_array = int(receiver.recv_string())
    # np.array2string(w)
    # weights = np.array((weight_array)).tolist()
    weightcnt = int(receiver.recv_string())
    pickledweights = receiver.recv()
    weights = pickle.loads(pickledweights)
    return weightcnt, weights

  def recvmsg_angle_throttle_roi(self):
    imgid = int(receiver.recv_string())
    angle = float(receiver.recv_string())
    throttle = float(receiver.recv_string())
    pickledimg = receiver.recv()
    roi = pickle.loads(pickledimg)
    return angle, throttle, roi

  def recvmsg_reward_roi(self):
    imgid = int(receiver.recv_string())
    pickledimg = receiver.recv()
    roi = pickle.loads(pickledimg)
    return roi

  def recvmsg_result(self):
    msgcnt = int(receiver.recv_string())
    result = float(receiver.recv_string())
    return msgcnt, result


class MessageClient():

  # def initclient()
  def __init__(self, portid = 5557):
    global sender , receiver, cfg
    cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))
    context = zmq.Context()
    # Socket to send messages to
    sender = context.socket(zmq.PUSH)
    addr = "tcp://localhost:%d" % portid
    sender.connect(addr)

  def sendmsg_get_weights(self):
    global cfg
    # no additional content
    sender.send_string("%d" % cfg.MSG_GET_WEIGHTS)
    return True

  def sendmsg_weights(self, weightcnt, weights):
    global cfg
    sender.send_string("%d" % cfg.MSG_WEIGHTS)
    sender.send_string("%d" % weightcnt)
    # w = np.array(weights)
    # msg = w.to_string()
    # sender.send_string(msg) 
    # print("weights")
    # print(weights)
    msg = pickle.dumps(weights)
    sender.send(msg) 
    return True

  def sendmsg_angle_throttle_roi(self,imgid, angle, throttle, roi):
    global cfg
    sender.send_string("%d" % cfg.MSG_ANGLE_THROTTLE_ROI)
    sender.send_string("%d" % imgid)
    sender.send_string("%f" % angle)
    sender.send_string("%f" % throttle)
    msg = pickle.dumps(roi)
    sender.send(msg) 
    return True

  def sendmsg_reward_roi(self,imgid, roi):
    global cfg
    sender.send_string("%d" % cfg.MSG_REWARD_ROI)
    sender.send_string("%d" % imgid)
    msg = pickle.dumps(roi)
    sender.send(msg) 
    return True

  def sendmsg_result(self, msgcnt, result):
    global cfg
    sender.send_string("%d" % cfg.MSG_RESULT)
    sender.send_string("%d" % msgcnt)
    sender.send_string("%f" % result)
    return True
