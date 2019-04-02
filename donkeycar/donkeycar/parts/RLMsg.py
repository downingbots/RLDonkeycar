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
from datetime import datetime

from threading import Thread
import os
# dependency, not in stdlib
from netifaces import interfaces, ifaddresses, AF_INET



class MessageServer():

  def __init__(self, portid = 5557, nonblocking = False):
    global cfg

    cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))

    self.DBG = False
    context = zmq.Context()
    blocking = (not nonblocking)
    self.do_send_weights = False
    self.emergency_stop_val = False
    self.same_values = True

    if (blocking):
      print("Blocking MsgSvr")
      # Socket to receive messages on
      # self.receiver = context.socket(zmq.ROUTER)
      # self.receiver = context.socket(zmq.PAIR)
      self.receiver = context.socket(zmq.PULL)
      # self.receiver = context.socket(zmq.REP)
      # addr = "tcp://*:%d" % portid
      addr = "tcp://*:%d" % portid
      self.receiver.bind(addr)
    else:
      print("Nonblocking MsgSvr")
      self.result = cfg.EMERGENCY_STOP * 2
      self.weights = None
      self.weight_cnt = 0
      self.imgid = -1
      self.state = None
      self.predicted_angle = None
      self.predicted_throttle = None 
      self.actual_angle = None
      self.actual_throttle = None 
      self.opencv_reward = None
      self.roi = None
      self.smean = [0,0,0]
      self.ymean = [0,0,0]
      self.wmean = [0,0,0]
      self.svar  = [0,0,0]
      self.yvar  = [0,0,0]
      self.wvar  = [0,0,0]
      self.scnt  = 0
      self.ycnt  = 0
      self.wcnt  = 0
      self.lw    = -1

      ctx = zmq.Context.instance()
      # self.receiver = context.socket(zmq.ROUTER)
      # self.receiver = context.socket(zmq.PAIR)
      self.receiver = context.socket(zmq.PULL)
      # self.receiver = context.socket(zmq.REP)
      addr = "tcp://*:%d" % portid
      listen_thread = Thread(target=self.listen, args=(addr,))
      listen_thread.start()
      
  def recv_msgtype(self):
    msg_type = int(self.receiver.recv_string())
    return msg_type

  def get_weight_cnt(self):
    return self.weight_cnt

  def get_weights(self):
    self.DBG = True
    return self.weights


  def listen(self, addr):

    """listen for messages in thread """
    print(addr)
    self.receiver.bind(addr)

    self.DBG = True
    # self.DBG = False
    while True:
        try:
            # if self.DBG:
            #   print("LSNR BLOCK")
            mtype = self.recv_msgtype()
            # if self.DBG:
            #   print("LSNR UNBLOCK : mtype = %d" % mtype)
            if mtype == cfg. MSG_STATE_ANGLE_THROTTLE_REWARD_ROI:
              self.recvmsg_state_angle_throttle_reward_roi()
              if self.DBG:
                print("Got msgcnt %d result %f" % (self.imgid, self.result))
            elif mtype == cfg.MSG_WEIGHTS:
              self.weightcnt, self.weights = self.recvmsg_weights()
              # just make weightcnt not equal to weightcnt in message 
              # must be set after weights global variable
              # weightcnt can be any increasing count from recvmsg_weights
              self.weight_cnt = self.weightcnt + 1 
              if self.DBG:
                print("Got weightcnt %d" % self.weight_cnt)
                # print(self.weights)
            elif mtype == cfg.MSG_GET_WEIGHTS:
              self.recvmsg_get_weights()
              print("MSG_GET_WEIGHTS")
              # print(self.Model.get_weights())
              # self.MsgClnt.sendmsg_weights(roi_cnt, self.Model.get_weights())
              # Model.clear_session()
            elif mtype == cfg.MSG_EMERGENCY_STOP:
              self.recvmsg_emergency_stop()
            else:
              print("ERROR: got mtype %d" % mtype)
        except (KeyboardInterrupt, zmq.ContextTerminated):
            break


  def recvmsg_get_weights(self):
    same = True
    local_wcnt = int(self.receiver.recv_string())
    if self.wcnt != local_wcnt:
      self.wcnt = local_wcnt
      same = False
    for i in range(3):
      local_wvar = float(self.receiver.recv_string())
      if self.wvar[i] != local_wvar:
        self.wvar[i] = local_wvar
        same = False
    for i in range(3):
      local_wmean = float(self.receiver.recv_string())
      if self.wmean[i] != local_wmean:
        self.wmean[i] = local_wmean
        same = False
    local_ycnt = int(self.receiver.recv_string())
    if self.ycnt != local_ycnt:
      self.ycnt = local_ycnt
      same = False
    for i in range(3):
      local_yvar = float(self.receiver.recv_string())
      if self.yvar[i] != local_yvar:
        self.yvar[i] = local_yvar
        same = False
    for i in range(3):
      local_ymean = float(self.receiver.recv_string())
      if self.ymean[i] != local_ymean:
        self.ymean[i] = local_ymean
        same = False
    local_scnt = int(self.receiver.recv_string())
    if self.scnt != local_scnt:
      self.scnt = local_scnt
      same = False
    for i in range(3):
      local_svar = float(self.receiver.recv_string())
      if self.svar[i] != local_svar:
        self.svar[i] = local_svar
        same = False
    for i in range(3):
      local_smean = float(self.receiver.recv_string())
      if self.smean[i] != local_smean:
        self.smean[i] = local_smean
        same = False

    local_lw = float(self.receiver.recv_string())
    if self.lw != local_lw:
      self.lw = local_lw
      same = False
    self.do_send_weights = True
    self.same_values = same
    return same

  def same_donkey_state(self):
    return self.same_values

  def get_donkey_state(self):
    return self.wcnt, self.wvar, self.wmean, self.ycnt, self.yvar, self.ymean, self.scnt, self.svar, self.smean, self.lw

  def send_weights(self):
    # no additional content
    return self.do_send_weights 

  def weights_sent(self):
    self.do_send_weights = False

  def recvmsg_weights(self):
    # weight_array = int(self.receiver.recv_string())
    # np.array2string(w)
    # weights = np.array((weight_array)).tolist()
    print("recvmsg_weights")
    weightcnt = int(self.receiver.recv_string())
    pickledweights = self.receiver.recv()
    weights = pickle.loads(pickledweights)
    return weightcnt, weights

  def recvmsg_state_angle_throttle_reward_roi(self):
    tmpimgid = int(self.receiver.recv_string())
    self.state = int(self.receiver.recv_string())
    DUMMY_VALUE = -1000
    self.predicted_angle = float(self.receiver.recv_string())
    if self.predicted_angle == DUMMY_VALUE:
      self.predicted_angle = None
    self.predicted_throttle = float(self.receiver.recv_string())
    if self.predicted_throttle == DUMMY_VALUE:
      self.predicted_throttle = None
    self.actual_angle = float(self.receiver.recv_string())
    self.actual_throttle = float(self.receiver.recv_string())
    self.opencv_reward = float(self.receiver.recv_string())
    if self.opencv_reward == DUMMY_VALUE:
      self.opencv_reward = None
    pickledimg = self.receiver.recv()
    self.roi = pickle.loads(pickledimg)
    # assign self.imgid last for thread synch reasons
    self.imgid = tmpimgid
    if tmpimgid % 100 == 0:
      dt_object = datetime.now()
      print("imgid ", imgid, " timestamp ", dt_object)

  def get_imgid(self):
    return self.imgid

  def get_state_angle_throttle_reward_roi(self):
    return self.imgid, self.state, self.predicted_angle, self.predicted_throttle, self.actual_angle, self.actual_throttle, self.opencv_reward, self.roi

  # used by secondary Control2
  def recvmsg_roi(self):
    # print("recvmsg_roi")
    # imgid = int(self.receiver.recv_string())
    pickledimg = self.receiver.recv()
    roi = pickle.loads(pickledimg)
    rl_ready = False
    if self.receiver.recv_string() == "True":
      rl_ready = True
    return roi, rl_ready

  def recvmsg_angle_throttle_reward(self):
    imgid = int(self.receiver.recv_string())
    predicted_angle = float(self.receiver.recv_string())
    predicted_throttle = float(self.receiver.recv_string())
    trial_angle = float(self.receiver.recv_string())
    trial_throttle = float(self.receiver.recv_string())
    trial_reward = float(self.receiver.recv_string())
    return predicted_angle, predicted_throttle, trial_angle, trial_throttle, trial_reward

  def recvmsg_result(self):
    msgcnt = int(self.receiver.recv_string())
    result = float(self.receiver.recv_string())
    # print("recvmsg_result %d" % msgcnt)
    return msgcnt, result

  def recvmsg_emergency_stop(self):
    msgcnt = int(self.receiver.recv_string())
    print("EMERGENCY_STOP %d" % msgcnt)
    self.emergency_stop_val = True
    return msgcnt

  def check_emergency_stop(self):
    if self.emergency_stop_val:
      self.emergency_stop_val = False
      return True
    return False

class MessageClient():

  # def initclient()
  def __init__(self, portid = 5557, nonblocking = False):
    global cfg
    self.DBG = False
    cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))
    context = zmq.Context()
    # Socket to send messages to
    # self.sender = context.socket(zmq.PAIR)
    if nonblocking:
      self.sender = context.socket(zmq.PUSH)
    else:
      self.sender = context.socket(zmq.PUSH)
      # self.sender = context.socket(zmq.REQ)
    addr = "tcp://localhost:%d" % portid
    # addr = "tcp://%s" % portid
    self.sender.connect(addr)

  def sendmsg_get_weights(self, wcnt, wvar, wmean, ycnt, yvar, ymean, scnt, svar, smean, lw):
    global cfg
    # no additional content
    if self.DBG:
      print("sendmsg_get_weights")
    self.sender.send_string("%d" % cfg.MSG_GET_WEIGHTS)
    self.sender.send_string("%d" % wcnt)
    for i in range(3):
      self.sender.send_string("%f" % wvar[i])
    for i in range(3):
      self.sender.send_string("%f" % wmean[i])
    self.sender.send_string("%d" % ycnt)
    for i in range(3):
      self.sender.send_string("%f" % yvar[i])
    for i in range(3):
      self.sender.send_string("%f" % ymean[i])
    self.sender.send_string("%d" % scnt)
    for i in range(3):
      self.sender.send_string("%f" % svar[i])
    for i in range(3):
      self.sender.send_string("%f" % smean[i])
    self.sender.send_string("%f" % lw)
    return True

  def sendmsg_weights(self, weightcnt, weights):
    global cfg
    self.DBG = True
    self.sender.send_string("%d" % cfg.MSG_WEIGHTS)
    self.sender.send_string("%d" % weightcnt)
    # w = np.array(weights)
    # msg = w.to_string()
    # self.sender.send_string(msg) 
    # print("weights")
    # print(weights)
    msg = pickle.dumps(weights)
    if self.DBG:
      print("sendmsg_weights")
    self.sender.send(msg) 
    return True

  def sendmsg_state_angle_throttle_reward_roi(self,imgid, state, predicted_angle, predicted_throttle, actual_angle, actual_throttle, prev_reward, roi):
    global cfg
    self.sender.send_string("%d" % cfg.MSG_STATE_ANGLE_THROTTLE_REWARD_ROI)
    self.sender.send_string("%d" % imgid)
    if imgid % 100 == 0:
      dt_object = datetime.now()
      print("imgid ", imgid, " timestamp ", dt_object)
    self.sender.send_string("%d" % state)
    DUMMY_VALUE = -1000.0
    if predicted_angle is None:
      self.sender.send_string("%f" % DUMMY_VALUE)
    else:
      self.sender.send_string("%f" % predicted_angle)
    if predicted_throttle is None:
      self.sender.send_string("%f" % DUMMY_VALUE)
    else:
      self.sender.send_string("%f" % predicted_throttle)
    self.sender.send_string("%f" % actual_angle)
    self.sender.send_string("%f" % actual_throttle)
    if prev_reward is None:
      self.sender.send_string("%f" % DUMMY_VALUE)
    else:
      self.sender.send_string("%d" % prev_reward)
    msg = pickle.dumps(roi)
    # if self.DBG:
    #   print("sendmsg_state_angle_throttle_reward_roi")
    self.sender.send(msg) 
    return True

  # control1 and control2
  def sendmsg_roi(self, roi, rl_ready):
    global cfg
    # print("sendmsg_roi")
    self.sender.send_string("%d" % cfg.MSG_ROI)
    # self.sender.send_string("%d" % imgid)
    msg = pickle.dumps(roi)
    self.sender.send(msg)
    # if self.DBG:
    #   print("sendmsg_angle_throttle_roi")
    if rl_ready:
      self.sender.send_string("True")
    else:
      self.sender.send_string("False")
    return True


  # used by secondary Control2
  def sendmsg_angle_throttle_reward(self, predicted_angle, predicted_throttle, trial_angle, trial_throttle):
    global cfg
    # print("sendmsg_angle_throttle")
    self.sender.send_string("%d" % cfg.MSG_ANGLE_THROTTLE)
    # self.sender.send_string("%d" % imgid)
    self.sender.send_string("%f" % predicted_angle)
    self.sender.send_string("%f" % predicted_throttle)
    self.sender.send_string("%f" % trial_angle)
    self.sender.send_string("%f" % trial_throttle)
    self.sender.send_string("%d" % trial_reward)
    # if self.DBG:
    #   print("sendmsg_angle_throttle")
    return True

  def sendmsg_reward_roi(self,imgid, roi):
    global cfg
    self.sender.send_string("%d" % cfg.MSG_REWARD_ROI)
    self.sender.send_string("%d" % imgid)
    # if imgid % 100 = 0:
    #   dt_object = datetime.fromtimestamp(timestamp)
    #   print("imgid ", imgid, " timestamp ", dt_object)
    msg = pickle.dumps(roi)
    self.sender.send(msg) 
    # if self.DBG:
    #   print("sendmsg_reward_roi")
    return True

  def sendmsg_result(self, msgcnt, result):
    global cfg
    # if self.DBG:
    #   print("sendmsg_result")
    self.sender.send_string("%d" % cfg.MSG_RESULT)
    self.sender.send_string("%d" % msgcnt)
    self.sender.send_string("%f" % result)
    return True

  def sendmsg_emergency_stop(self, msgcnt):
    global cfg
    # if self.DBG:
    #   print("sendmsg_result")
    print("sendmsg EMERGENCY_STOP %d" % msgcnt)
    self.sender.send_string("%d" % cfg.MSG_EMERGENCY_STOP)
    self.sender.send_string("%d" % msgcnt)
    return True

