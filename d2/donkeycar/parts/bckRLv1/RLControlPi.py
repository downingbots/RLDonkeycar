'''
------------------
Step 1: initial training
ControlPi
  - Runs CV2 line following
  - get reward
  - send ROI(n) to rpi2 with steering/throttle
  - send ROI(n+1) to rpi2
  - controls car
  - stops if outside track

RLPi
  - receives ROI(n), thottle, steering 
  - receives ROI(N+1) reward
  - trains as fast as possible
  - drops on floor what it can't handle
  - no car control

------------------
Step 2: RLPi to ControlPi  Model transfer

RPI2 sends ControlPi the model after X (thousands
   30sec/sec x 20frames x 10 laps = 6000 images trained
ControlPi continues to control car until model is ready

------------------

Step 3: NN takes control of ControlPi

ControlPi:
  - increase FPS hertz
  - NN takes control from CV2
  - typically uses current model to control car
  - when get READY from RPI2:
    -- Do NN to get steering/throttle
    -- randomly change steering/throttle by small delta (GPS)
    -- sends ROI(n),steering/throttle to RPI2
    -- sends ROI(n+1) to rpi2 

  - random delta throttle: +2 / +1/  0/ -1
                           10%/50%/30%/10%
    random delta steering: -2 / -1/  0/ +1/ +2
      with distribution:   10% 20% 40% 20% 10%
    note: 12% of 0 delta throttle/0 delta steering.
            Just run with reward anyway
    if steering already at +-Max:
            increase odds of steering to start at Max

RLPi:
  - send Ready to ControlPi
  - receives ROI(n) from rpi1 with GPS steering/throttle
  - receives ROI(n+1) from rpi1 
  - computes ROI(n+1) reward
     -- if out of bounds send "Stop" to RPI1
  - feedback/train model

------------------
Step 4: RLPi to ControlPi  Model transfer

RPI2 sends revised model to RPI1 after:
  - X (1000) random images
  - or if stopped 
ControlPi continues to control car until model is ready
ControlPi switches to new model params

------------------
'''
import donkeycar as dk
import keras
import os
from donkeycar.parts.RLOpenCV import ThrottleBase
from donkeycar.parts.RLKeras import KerasRLCategorical
from donkeycar.parts.RLMsg import MessageClient, MessageServer
from donkeycar.parts.RLOpenCV import HoughBundler, LaneLines


class ControlPi():

    '''
    def rl_copy_model(self, model):
      model_copy = keras.models.clone_model(model)
      model_copy.set_weights(model.get_weights())
      return model_copy
    '''

    def run_by_state(self, img):
      global TB, last_model_update, cfg
      global MsgClnt, MsgSvr, model_state
      global RLPiState, RLPi_msgcnt, RLPi_weightcnt
      global KRLC, imgid

      if TB.emergencyStop():
        print("EMERGENCY STOP TRUE")
        # eventually run_opencv will detect vanishing point and unset bit
        VP, VP2, angle, throttle = self.run_opencv(img)
        if VP:
          print("1a th %f angle %f" % (throttle, angle))
          TB.setEmergencyStop(False)
        elif VP2:
          # idealy would set throttle high for one iteration
          throttle = throttle  
          angle = -1   # assume car to right of middle line
          print("1b th %f angle %f" % (throttle, angle))
          TB.setEmergencyStop(False)
        else:
          print("NO VANISHING POINTS")
          angle = 0
          throttle = 0
          # print("1c th %f angle %f" % (throttle, angle))
      elif  RLPi_msgcnt < cfg.SWITCH_TO_NN:
        # if model_state != cfg.STATE_NN:
          # self.setFPSHz(10):
        # self.setFPSHz(10):
        model_state = cfg.STATE_OPENCV
        VP, VP2, angle, throttle = self.run_opencv(img)
        # print("2 th %f angle %f" % (throttle, angle))
      elif  RLPi_msgcnt >= cfg.SWITCH_TO_NN:
        if model_state == cfg.STATE_NN:
          if RLPiState == cfg.RLPI_READY1:
            # run with random changes
            angle, throttle = self.run_nn(img, True)
            if (cfg.THROTTLE_CONSTANT > 0):
              throttle = cfg.THROTTLE_CONSTANT
            print("3 th %f angle %f" % (throttle, angle))
          else:
            # run with just the model
            angle, throttle = self.run_nn(img, False)
            print("4 th %f angle %f" % (throttle, angle))
        elif model_state == cfg.STATE_OPENCV:
          MsgClnt.sendmsg_get_weights()
          VP, VP2, angle, throttle = self.run_opencv(img)
          model_state = cfg.STATE_MODEL_TRANSFER_STARTED
          # print("5 th %f angle %f" % (throttle, angle))
        elif model_state == cfg.STATE_MODEL_TRANSFER_STARTED :
          VP, VP2, angle, throttle = self.run_opencv(img)
          # print("6 th %f angle %f" % (throttle, angle))
        if  RLPi_msgcnt - last_model_update % cfg.UPDATE_NN == 0:
          MsgClnt.sendmsg_get_weights()
          last_model_update =  RLPi_msgcnt
          # TB.resetThrottleInfo()
          # angle, throttle = self.run_nn(img)
      # print("7 th %f" % throttle)

      # MType = MsgSvr.recv_msgtype()
      msgcnt, result = MsgSvr.get_msgcnt_result()
      weightcnt = MsgSvr.get_weight_cnt()
      # print("Mcnt = %d Weightcnt %d modelst %d RLPI = %d" % (msgcnt, weightcnt, model_state, RLPiState))
      # print("Mcnt = %d result %d RLPI = %d" % (msgcnt, result,RLPiState))
      if RLPiState == cfg.RLPI_READY1:
        minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
        print("minth %f maxth %f batadj %f th %f" % (minthrottle, maxthrottle, battery_adjustment, throttle))
        if (maxthrottle > 0):
          # print("angle %f " % (angle))
          # reward_throttle = throttle - minthrottle - battery_adjustment 
          reward_throttle = throttle
          MsgClnt.sendmsg_angle_throttle_roi(imgid, angle, reward_throttle, img)
          print("MsgClnt.sendmsg_angle_throttle_roi; set to READY2")
          RLPiState = cfg.RLPI_READY2
        else:
          print("MAXTHROT ERR: minth %f maxth %f batadj %f th %f" % (minthrottle, maxthrottle, battery_adjustment, throttle))
      elif RLPiState == cfg.RLPI_READY2:
        MsgClnt.sendmsg_reward_roi(imgid, img)
        print("MsgClnt.sendmsg_reward_roi; set to WAITING")
        RLPiState = cfg.RLPI_WAITING
      # if MType == cfg.MSG_RESULT:
      if msgcnt != RLPi_msgcnt:
        RLPi_msgcnt = msgcnt
        # trigger new weight request
        last_model_update = RLPi_msgcnt - cfg.UPDATE_NN + 1
        # print("RL %d RESULT %d" % (msgcnt, result))
        RLPiState = cfg.RLPI_READY1
        if result == cfg.EMERGENCY_STOP:
          if cfg.DISABLE_EMERGENCY_STOP:
            pass
          else:
            print("EMERGENCY STOP SET")
            TB.resetThrottleInfo()
            TB.setEmergencyStop(True)
            if msgcnt >= cfg.SWITCH_TO_NN:
              if RLPi_msgcnt - last_model_update > (cfg.UPDATE_NN / 2):
                print("WEIGHT UPDATE REQUEST")
                MsgClnt.sendmsg_get_weights()
                last_model_update =  RLPi_msgcnt
            angle = 0
            throttle = 0
      if weightcnt != RLPi_weightcnt:
        RLPi_weightcnt = weightcnt
        print("RECEIVED WEIGHTS")
        weights = MsgSvr.get_weights() 
        if weights is None:
          print("None weights")
        else:
          print("Not None weights")
        KRLC.set_weights(weights)
        if model_state == cfg.STATE_MODEL_TRANSFER_STARTED: 
          print("MODEL TRANSFER COMPLETED")
          # self.setFPSHz(20)
          last_model_update = RLPi_msgcnt
          model_state = cfg.STATE_NN 
      imgid += 1
      return angle, throttle



    def run_opencv(self, img):
        global steering, throttle, steering_hist, throttle_hist, speed 
        global angle, conf, ll, TB

        VP = False
        VP2 = False
        mode = cfg.MODE_STEER_THROTTLE
        simplecl, lines, roi = ll.process_img(img)
        complexsteering = 0
        complexthrottle = 0
        # if simplecl is None and lines is not None and roi is not None:
        if mode == cfg.MODE_COMPLEX_LANE_FOLLOW and lines is not None and roi is not None:
            # although slower, let's see what happens when we
            # now run complex lrclines to gather global history, extra info 
            # discard steering and throttle
            complexsteering, complexthrottle = ll.lrclines(lines,roi)
            VP = ll.is_vanishing_point()
            VP2 = ll.is_vanishing_point2()
        if simplecl is not None:
          pos = 4
          conf = cfg.MAX_ACCEL
          conf, steering, throttle = ll.setSteerThrottle(pos, None, simplecl, None, conf)
        elif mode == cfg.MODE_COMPLEX_LANE_FOLLOW and complexthrottle > 0:
          # fall back to complex lane following
          steering = complexsteering 
          throttle = complexthrottle
        else:
          steering = 0
          throttle = 0

        # print("STEER %f THROT %f" % (steering, throttle))
        return VP, VP2, steering, throttle 

    def setFPSHz(hz):
      # needs to be done as a reurn parameter
      pass

    def run_nn(self, img, RL_ready):
      global KRLC, TB
      if TB.throttleCheckInProgress():
        TB.setMinMaxThrottle(img)
      minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
      angle, throttle = KRLC.trial_run(img, RL_ready)
      # angle, throttle is returned in float to control pi
      print("run_nn trial_run angle %f throttle %f minth %f batadj %f" % (angle, throttle, minthrottle, battery_adjustment))
      # return angle, (throttle+minthrottle+battery_adjustment)
      return angle, throttle

    def run(self, img):
      return self.run_by_state(img)

    def __init__(self):
      global model_state, KRLC, cfg, TB, last_model_update
      global imgid, MsgSvr, MsgClnt, ll
      global RLPiState, RLPi_msgcnt, RLPi_weightcnt

      RLPi_msgcnt = -1
      RLPi_weightcnt = 0
      last_model_update =  0
      cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))
      TB = ThrottleBase()
      ll = LaneLines(TB)
      KRLC = KerasRLCategorical(LaneLine = ll)
      imgid = 0
      # self.setFPSHz(10)
      # cfg.STATE_EMERGENCY_STOP = 0
      model_state = cfg.EMERGENCY_STOP
      MsgClnt = MessageClient(portid=cfg.PORT_RLPI)
      MsgSvr = MessageServer(portid=cfg.PORT_CONTROLPI, nonblocking = True)
      RLPiState = cfg.RLPI_READY1

