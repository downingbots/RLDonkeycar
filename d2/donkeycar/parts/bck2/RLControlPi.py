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
from donkeycar.parts.RLOpenCV import ThrottleBase
from donkeycar.parts.RLKeras import KerasRLCategorical
from donkeycar.parts.RLMsg import MessageClient, MessageServer
from donkeycar.parts.RLOpenCV import HoughBundler, LaneLines


class ControlPi():

    def rl_copy_model(self, model):
      model_copy = keras.models.clone_model(model)
      model_copy.set_weights(model.get_weights())
      return model_copy

    def run_by_state(self, img):
      global state, TB, last_model_update, cfg
      global MsgClnt, MsgSvr, RLPiState, model_state
      global KRLC, imgid

      if TB.emergencyStop():
        # eventually run_opencv will detect vanishing point and unset bit
        angle, throttle = self.run_opencv(img)
        angle, throttle = 0,0
      elif  RLPi_msgcnt < cfg.SWITCH_TO_NN:
        # if state != cfg.STATE_NN:
          # self.setFPSHz(10):
        # self.setFPSHz(10):
        state = cfg.STATE_OPENCV
        angle, throttle = self.run_opencv(img)
      elif  RLPi_msgcnt >= cfg.SWITCH_TO_NN:
        if model_state == cfg.STATE_NN:
          angle, throttle = self.run_nn(img)
        elif state == cfg.STATE_OPENCV:
          MsgClnt.sendmsg_get_model()
          angle, throttle = self.run_opencv(img)
          model_state = cfg.STATE_MODEL_TRANSFER_STARTED
        elif model_state == cfg.STATE_MODEL_TRANSFER_STARTED :
          angle, throttle = self.run_opencv(img)
        if  RLPi_msgcnt - last_model_update % cfg.UPDATE_NN == 0:
          MsgClnt.sendmsg_get_model()
          last_model_update =  RLPi_msgcnt
          # TB.resetThrottleInfo()
          # angle, throttle = self.run_nn(img)

      # MType = MsgSvr.recv_msgtype()
      global RLPi_msgcnt, RLPi_weightcnt
      msgcnt, result = MsgSvr.get_msgcnt_result()
      weightcnt, weights = MsgSvr.get_weights()
      # print("Mcnt = %d result %d RLPI = %d" % (msgcnt, result,RLPiState))
      if RLPiState == cfg.RLPI_READY1:
        minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
        print("minth %f maxth %f batadj %f th %f" % (minthrottle, maxthrottle, battery_adjustment, throttle))
        if (maxthrottle > 0):
          print("angle %f " % (angle))
          # reward_throttle = throttle - minthrottle - battery_adjustment 
          reward_throttle = throttle
          MsgClnt.sendmsg_angle_throttle_roi(imgid, angle, reward_throttle, img)
          print("MsgClnt.sendmsg_angle_throttle_roi")
          RLPiState = cfg.RLPI_READY2
      elif RLPiState == cfg.RLPI_READY2:
        MsgClnt.sendmsg_reward_roi(imgid, img)
        print("MsgClnt.sendmsg_reward_roi")
        RLPiState = cfg.RLPI_WAITING
      # if MType == cfg.MSG_RESULT:
      if msgcnt != RLPI_msgcnt:
        RLPi_msgcnt = msgcnt
        print("RL %d RESULT %d" % (msgcnt, result))
        RLPiState = cfg.RLPI_READY1
        if result == cfg.EMERGENCY_STOP:
          print("EMERGENCY STOP")
          TB.resetThrottleInfo()
          TB.setEmergencyStop(True)
          if msgcnt >= cfg.SWITCH_TO_NN:
            MsgClnt.sendmsg_get_model()
          angle, throttle = 0,0
      if weightcnt != RLPI_weightcnt:
        RLPi_weightcnt = weightcnt
        print("RECEIVED WEIGHTS")
        KRLC.set_weights(weights)
        if model_state == cfg.MODEL_TRANSFER_STARTED: 
          print("MODEL TRANSFER STARTED")
          # self.setFPSHz(20)
          last_model_update = RLPi_msgcnt
          model_state = cfg.STATE_NN 
      imgid += 1
      return angle, throttle



    def run_opencv(self, img):
        global steering, throttle, steering_hist, throttle_hist, speed 
        global angle, conf, ll, TB

        mode = cfg.MODE_SIMPLE_LINE_FOLLOW 
        simplecl, lines, roi = ll.process_img(img)
        if mode == cfg.MODE_COMPLEX_LANE_FOLLOW and lines is not None and len(lines) > 0 and roi is not None:
          steering, throttle = ll.lrclines(lines,roi)
        elif mode == cfg.MODE_SIMPLE_LINE_FOLLOW:
          # although slower, let's see what happens when we
          # now run complex lrclines to gather global history, extra info 
          complexsteering = 0
          complexthrottle = 0
          # if simplecl is None and lines is not None and roi is not None:
          if lines is not None and roi is not None:
            # discard steering and throttle
            complexsteering, complexthrottle = ll.lrclines(lines,roi)
          if simplecl is not None:
            pos = 4
            conf = 10
            conf, steering, throttle = ll.setSteerThrottle(pos, None, simplecl, None, conf)
          elif complexthrottle > 0:
            # fall back to complex lane following
            steering = complexsteering 
            throttle = complexthrottle
          else:
            steering = 0
            throttle = 0
        else:
          steering = 0
          throttle = 0
        print("STEER %f THROT %f" % (steering, throttle))
        return steering, throttle 

    def setFPSHz(hz):
      # needs to be done as a reurn parameter
      pass

    def run_nn(self, img):
      global KRLC, TB
      if TB.throttleCheckInProgress():
        self.setMinMaxThrottle(img)
      minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
      angle, throttle = KRLC.trial_run(img)
      return angle, (throttle+minthrottle+battery_adjustment)

    def run(self, img):
      return self.run_by_state(img)

    def __init__(self):
      global state, KRLC, cfg, TB
      global imgid, MsgSvr, MsgClnt, ll
      global RLPiState, RLPi_msgcnt, RLPi_weightcnt

      RLPi_msgcnt = -1
      RLPi_weightcnt = 0
      cfg = dk.load_config(config_path='/home/ros/donkeycar/donkeycar/parts/RLConfig.py')
      TB = ThrottleBase()
      ll = LaneLines(TB)
      KRLC = KerasRLCategorical(LaneLine = ll)
      imgid = 0
      # self.setFPSHz(10)
      # cfg.STATE_EMERGENCY_STOP = 0
      state = cfg.EMERGENCY_STOP
      MsgClnt = MessageClient(portid=cfg.PORT_RLPI)
      MsgSvr = MessageServer(portid=cfg.PORT_CONTROLPI, nonblocking = True)
      RLPiState = cfg.RLPI_READY1

