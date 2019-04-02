'''
Reinforcement Learning Pi (RLPi)

RLPi will:
  - runs RL on ControlPi data as fast as can.
    - cache ROI(n) with steering/throttle from ControlPi
    - Run RL when receive ROI(n+1) from ControlPi
    - acks with step # when ready for next message 
  - save Model parameters periodically or on-demand 
  - send over Model parameters upon ControlPi request

ControlPi will send over:
  - Run data:
    - send ROI(n) to rpi2 with computed steering/throttle
    - send ROI(n+1) to rpi2 (separate message)
  - Request for Model

Note: RLPi has no control over DK car.
'''

import os
from donkeycar.parts.RLMsg import MessageServer, MessageClient
from donkeycar.parts.RLKeras import KerasRLCategorical
from donkeycar.parts.RLOpenCV import ThrottleBase, HoughBundler, LaneLines
import donkeycar as dk

class RLServer():
    def __init__(self):
      global MsgSvr, MsgClnt, Model, cfg
      global ll, TB

      cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))

      TB = ThrottleBase()
      ll = LaneLines(TB)
      MsgSvr = MessageServer(portid = cfg.PORT_RLPI)
      MsgClnt = MessageClient(portid = cfg.PORT_CONTROLPI)
      Model = KerasRLCategorical(LaneLine=ll)
      try:
        #load keras model
        Model.load('/home/ros/d2/models/rlpilot')
      except:
        print("No model to load")

      roi_cnt = 0
      trial_angle = 0 
      trial_throttle = 0 
      trial_roi = None

      DBG = False
      while True:
        # roi_cnt += 1
        if DBG:
          print("about to block")
        msgtype = MsgSvr.recv_msgtype()
        if DBG:
          print("MType %d" % msgtype)
        if msgtype == cfg.MSG_ANGLE_THROTTLE_ROI:
          trial_angle, trial_throttle, trial_roi = MsgSvr.recvmsg_angle_throttle_roi()
          if DBG:
            print("MSG_ANGLE_THROTTLE_ROI")
        elif msgtype == cfg.MSG_REWARD_ROI:
          reward_roi = MsgSvr.recvmsg_reward_roi()
          if DBG:
            print("MSG_REWARD_ROI")
          roi_cnt += 1
          reward = Model.rl_fit(roi_cnt, trial_roi, reward_roi, trial_angle, trial_throttle, os.path.expanduser(cfg.RL_MODEL_PATH))
          MsgClnt.sendmsg_result(roi_cnt, reward)
          # print("sent roicnt %d reward %f" % (roi_cnt, reward))
        elif msgtype == cfg.MSG_GET_WEIGHTS:
          MsgSvr.recvmsg_get_weights()
          print("MSG_GET_WEIGHTS")
          # print(Model.get_weights())
          MsgClnt.sendmsg_weights(roi_cnt, Model.get_weights())
          # Model.clear_session()
          # Model.load('/home/ros/d2/models/rlpilot')
          DBG = True

RLS = RLServer()
