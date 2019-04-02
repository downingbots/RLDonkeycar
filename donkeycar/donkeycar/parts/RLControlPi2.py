'''
ControlPi2 runs NN and loads weights for ControlPi1.

Tensorflow can deadlock when loading weights in a thread.
So, ControlPi2 runs in the foreground and acts like a NN RPC server
for ControlPi.py.

Not currently used when using a single categorical output for Angle-Throttle.

'''

import os
import numpy as np
from donkeycar.parts.RLMsg import MessageServer, MessageClient
from donkeycar.parts.RLKeras import KerasRLContinuous
from donkeycar.parts.RLOpenCV import ThrottleBase, HoughBundler, LaneLines
import donkeycar as dk

class RLServer2():
    def __init__(self):
      global MsgSvr, MsgClnt, Model, cfg
      global ll, TB

      cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))

      TB = ThrottleBase()
      ll = LaneLines(TB)
      MsgSvr = MessageServer(portid = cfg.PORT_CONTROLPI2RL)
      MsgClnt = MessageClient(portid = cfg.PORT_CONTROLPI2)
      Model = KerasRLContinuous(LaneLine=ll)

      roi_cnt = 0
      angle = 0
      throttle = 0
      weights_init = False
      DBG = True
      nncnt = 0
      wcnt = 0
      prev_img = None

      while True:
        # if DBG:
        #   print("about to block")
        msgtype = MsgSvr.recv_msgtype()
        if DBG:
          print("MType %d" % msgtype)
        if msgtype == cfg.MSG_ROI:
          img, RL_ready = MsgSvr.recvmsg_roi()
          if RL_ready:
            nncnt += 1
            print("MSG_ROI %d" % nncnt)
          # if prev_img is not None and np.array_equal(img, prev_img):
          #   print("same image")
          # else:
          #   print("diff image")
          # prev_img = img
           
          predicted_angle, predicted_throttle, trial_angle, trial_throttle, trial_reward = Model.trial_run(img, RL_ready)
          MsgClnt.sendmsg_angle_throttle_reward( predicted_angle, predicted_throttle, trial_angle, trial_throttle, trial_reward)
          print("sent throttle %f angle %f" % (throttle, angle))
        elif msgtype == cfg.MSG_WEIGHTS:
          wcnt += 1
          weights_init = True
          weight_cnt, weights = MsgSvr.recvmsg_weights()
          print("MSG_WEIGHTS %d" % wcnt)
          # print(MsgSvr.get_weights())
          Model.set_weights(weights)
          # angle, throttle = Model.trial_run(img, False)
          # no previous img ; send dummy angle/throttle; 
          # opencv should be used by controlpi
          predicted_angle = 0
          predicted_throttle = 0
          trial_angle = 0
          trial_throttle = 0
          trial_reward = 0
          MsgClnt.sendmsg_angle_throttle_reward( predicted_angle, predicted_throttle, trial_angle, trial_throttle, trial_reward)
          print("sent %d throttle %f angle %f " % (wcnt, throttle, angle))
          # Model.clear_session()
          # Model.load('/home/ros/d2/models/rlpilot')
        else:
          print("Unknown MType %d" % msgtype)

RLS = RLServer2()
