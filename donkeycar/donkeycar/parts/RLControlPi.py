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
import os, sys
from threading import Thread
from donkeycar.parts.RLOpenCV import ThrottleBase
from donkeycar.parts.RLKeras import KerasRLContinuous
from donkeycar.parts.RLMsg import MessageClient, MessageServer
from donkeycar.parts.RLOpenCV import HoughBundler, LaneLines


class ControlPi():

    '''
    def rl_copy_model(self, model):
      model_copy = keras.models.clone_model(model)
      model_copy.set_weights(model.get_weights())
      return model_copy
    '''

    def run_wsgi(self):
        from donkeycar.management.base import Sim

        SS = Sim()
        SS.reconnect()
        '''
        args = sys.argv[:]
        args[0] = "--type"
        args[1] = "rl"
        SS.run(args)
        '''
  
        ''' 
    # need to run outside run_by_state
    def set_weights(weights, img):
        global KRLC, cfg
        global last_model_update, model_state, RLPiState, RLPi_msgcnt

        KRLC.set_weights(weights, "control")
        # run once to load model into memory
        img_arr = img_arr.reshape((1,) + img.shape)
        KRLC.predict(img_arr) 

        last_model_update = RLPi_msgcnt
        model_state = cfg.STATE_NN
        RLPiState = cfg.RLPI_READY1
        ''' 


    def run_by_state(self, img):
      global TB, last_model_update, cfg
      global MsgClnt, MsgSvr, model_state
      global RLPiState, RLPi_msgcnt, RLPi_weightcnt
      global KRLC, imgid
      # global thrd

      DBG = False
      need_wsgi = False
      # print("RBS")
      cur_model_state = model_state
      if TB.emergencyStop():
        # currently only for sim to reset car
        print("EMERGENCY STOP TRUE")
        # eventually run_opencv will detect vanishing point and unset bit
        predicted_throttle = None
        predicted_angle = None
        TB.setEmergencyStop(False)
        return cfg.SIM_EMERGENCY_STOP, cfg.SIM_EMERGENCY_STOP 
        VP, VP2, trial_angle, trial_throttle = self.run_opencv(img)
        minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
        trial_throttle = max(throttle - minthrottle - battery_adjustment,0)
        if VP:
          print("1a th %f angle %f" % (trial_throttle, trial_angle))
          TB.setEmergencyStop(False)
        elif VP2:
          # idealy would set throttle high for one iteration
          trial_throttle = trial_throttle  
          trial_angle = -1   # assume car to right of middle line
          print("1b th %f angle %f" % (trial_throttle, trial_angle))
          TB.setEmergencyStop(False)
        else:
          print("NO VANISHING POINTS")
          predicted_angle = 0
          predicted_throttle = 0
          trial_angle = 0
          trial_throttle = 0
      elif  RLPi_msgcnt < cfg.SWITCH_TO_NN:
        # if model_state != cfg.STATE_NN:
          # self.setFPSHz(10):
        # self.setFPSHz(10):
        model_state = cfg.STATE_OPENCV
        predicted_angle = None
        predicted_throttle = None
        VP, VP2, trial_angle, trial_throttle, trial_reward = self.run_opencv(img)
        minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
        trial_throttle = max(trial_throttle - minthrottle - battery_adjustment,0)
        print("%d C th %f angle %f" % (RLPi_msgcnt, trial_throttle, trial_angle))
      elif  RLPi_msgcnt >= cfg.SWITCH_TO_NN:
        if model_state == cfg.STATE_NN:
          DBG = True
          need_wsgi = True
          if MsgSvr.check_emergency_stop():
            return cfg.SIM_EMERGENCY_STOP, cfg.SIM_EMERGENCY_STOP 
          if RLPiState == cfg.RLPI_READY1:
            # run with random changes
            predicted_angle, predicted_throttle, trial_angle, trial_throttle = self.run_nn(img, True)
            trial_reward = None
            if (cfg.THROTTLE_CONSTANT > 0):
              trial_throttle = cfg.THROTTLE_CONSTANT
            print("3 th %f angle %f" % (trial_throttle, trial_angle))
            RLPiState = cfg.RLPI_READY2  # alternate betw random and predicted
          else:
            # run with just the model
            predicted_angle, predicted_throttle, trial_angle, trial_throttle = self.run_nn(img, False)
            trial_reward = None
            print("4 th %f angle %f state %d" % (trial_throttle, trial_angle, RLPiState))
            RLPiState = cfg.RLPI_READY1  # alternate betw random and predicted
        elif model_state == cfg.STATE_PARTIAL_NN:
          # PARTIAL NN is required when using Control2 process
          DBG = True
          need_wsgi = True
          if RLPiState == cfg.RLPI_READY1:
            # run with just the model (no random changes)
            predicted_angle, predicted_throttle, trial_angle, trial_throttle = self.run_nn(img, False)
            trial_reward = None
            if (cfg.THROTTLE_CONSTANT > 0):
              trial_throttle = cfg.THROTTLE_CONSTANT
            print("8 th %f angle %f" % (trial_throttle, trial_angle))
            cur_model_state = cfg.STATE_NN
          if RLPiState == cfg.RLPI_READY2:
            # run with openCV
            predicted_throttle = None
            predicted_angle = None
            VP, VP2, trial_angle, trial_throttle, trial_reward = self.run_opencv(img)
            minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
            trial_throttle = max(trial_throttle - minthrottle - battery_adjustment,0)
            print("9 th %f angle %f state %d" % (trial_throttle, trial_angle, RLPiState))
            cur_model_state = cfg.STATE_OPENCV
          if  RLPi_msgcnt - cfg.SWITCH_TO_NN > cfg.PARTIAL_NN_CNT:
            model_state = cfg.STATE_NN

        elif model_state == cfg.STATE_OPENCV:
          wcnt, wvar, wmean, ycnt, yvar, ymean, scnt, svar, smean, lw, minth = ll.getDonkeyState()
          MsgClnt.sendmsg_get_weights(wcnt, wvar, wmean, ycnt, yvar, ymean, scnt, svar, smean, lw)
          predicted_throttle = None
          predicted_angle = None
          VP, VP2, trial_angle, trial_throttle, trial_reward = self.run_opencv(img)
          minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
          predicted_throttle = 0
          predicted_angle = 0
          trial_throttle = max(trial_throttle - minthrottle - battery_adjustment,0)
          model_state = cfg.STATE_MODEL_TRANSFER_STARTED
          print("5 th %f angle %f" % (trial_throttle, trial_angle))
        elif model_state == cfg.STATE_MODEL_WEIGHTS_SET:
          # run opencv until NN completely ready and in memory
          predicted_throttle = None
          predicted_angle = None
          VP, VP2, trial_angle, trial_throttle, trial_reward = self.run_opencv(img)
          minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
          trial_throttle = max(trial_throttle - minthrottle - battery_adjustment,0)
          print("6 th %f angle %f" % (trial_throttle, trial_angle))
        elif model_state == cfg.STATE_MODEL_PREPARE_NN:
          # processed below in same long-running thread
          print("RECEIVED WEIGHTS")
          model_state = cfg.STATE_MODEL_WEIGHTS_SET
          weightcnt = MsgSvr.get_weight_cnt()
          weights = MsgSvr.get_weights()
          if weights is None:
            print("None weights")
          if cfg.PORT_CONTROLPI2 is None:
            print("local set weights")
            predicted_angle = 0
            predicted_throttle = 0
            trial_angle = 0
            trial_throttle = 0
            trial_reward = 0
            KRLC.set_weights(weights, model = "actor")
            img_arr = img.reshape((1,) + img.shape)
            predicted_angle, predicted_throttle = KRLC.predicted_actions(img_arr) # set up KRLC in memory while Waiting
          else:
            global MsgClnt2, MsgSvr2
            print("MsgClnt2.sendmsg_weights")
            MsgClnt2.sendmsg_weights(weightcnt, weights)
            predicted_angle, predicted_throttle, trial_angle, trial_throttle, trial_reward = MsgSvr2.recvmsg_angle_throttle_reward()
          print("6b th %f angle %f" % (trial_throttle, trial_angle))
          print("MODEL TRANSFER COMPLETED")
          last_model_update = RLPi_msgcnt
          if cfg.PORT_CONTROLPI2 is None:
            model_state = cfg.STATE_NN
          else:
            model_state = cfg.STATE_PARTIAL_NN
            RLPiState = cfg.RLPI_READY1
        elif model_state == cfg.STATE_MODEL_TRANSFER_STARTED :
          # run opencv until NN completely ready and in memory
          predicted_throttle = None
          predicted_angle = None
          VP, VP2, trial_angle, trial_throttle, trial_reward = self.run_opencv(img)
          minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
          trial_throttle = max(trial_throttle - minthrottle - battery_adjustment,0)
          print("6c th %f angle %f" % (trial_throttle, trial_angle))
        if  RLPi_msgcnt - last_model_update - cfg.UPDATE_NN >= 0 and model_state != cfg.STATE_MODEL_TRANSFER_STARTED:
          print("update NN %d %d %d" % (RLPi_msgcnt, last_model_update, model_state)) 
          wcnt, wvar, wmean, ycnt, yvar, ymean, scnt, svar, smean, lw, minth = ll.getDonkeyState()
          MsgClnt.sendmsg_get_weights(wcnt, wvar, wmean, ycnt, yvar, ymean, scnt, svar, smean, lw)
          last_model_update =  RLPi_msgcnt

      # Detect and Process Asynchronous Messages
      # only latest msgcnt_result processed
      # msgcnt, result = MsgSvr.get_msgcnt_result()
      weightcnt = MsgSvr.get_weight_cnt()
      if DBG and weightcnt != RLPi_weightcnt:
        print("Weightcnt %d expected weightcnt %d modelst %d RLPI = %d" % (weightcnt, RLPi_weightcnt, model_state, RLPiState))
        # print("Mcnt = %d result %d RLPI = %d" % (msgcnt, result,RLPiState))
      # in DDQN, only RLPiState is READY1
      if RLPiState == cfg.RLPI_READY1 or RLPiState == cfg.RLPI_READY2:
        minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
        print("minth %f maxth %f batadj %f th %f" % (minthrottle, maxthrottle, battery_adjustment, trial_throttle))
        if (maxthrottle > 0 and imgid > cfg.INIT_STEER_FRAMES):
          MsgClnt.sendmsg_state_angle_throttle_reward_roi(imgid, cur_model_state, predicted_angle, predicted_throttle, trial_angle, trial_throttle, trial_reward, img)
          # print("MsgClnt.sendmsg_angle_throttle_roi: a %f t %f r %d" % (trial_angle, trial_throttle, trial_reward))
          # RLPiState = cfg.RLPI_READY2
        else:
          print("MAXTHROT ERR: minth %f maxth %f batadj %f th %f" % (minthrottle, maxthrottle, battery_adjustment, trial_throttle))
      adjust_throttle = True
      if weightcnt != RLPi_weightcnt:
        RLPi_weightcnt = weightcnt
        model_state = cfg.STATE_MODEL_PREPARE_NN
        print("New state: PREPARE_NN")

      imgid += 1
      RLPi_msgcnt = imgid
      minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
      if adjust_throttle:
        trial_throttle = trial_throttle + minthrottle + battery_adjustment
      if DBG:
        print("final a %f t %f" % ( trial_angle, trial_throttle))
      return trial_angle, trial_throttle

    def run_opencv(self, img):
        global steering, throttle, steering_hist, throttle_hist, speed 
        global angle, conf, ll, TB

        VP = False
        VP2 = False
        mode = cfg.MODE_STEER_THROTTLE
        complexsteering = 0
        complexthrottle = 0
        Lines = None
        wLines = None
        yLines = None
        simplecl = None

        if cfg.USE_COLOR:
          line_color_simple, line_color_yellow, line_color_white = ll.get_line_color_info()

        if cfg.USE_COLOR and line_color_yellow is not None and line_color_white is not None and line_color_simple is not None:
          simplecl, wlines, ylines, roi = ll.process_img_color(img)
          currline = []
          curlline = []
          curcline = []
          if wlines is not None:
            currline, curlline, dummycurcline = ll.lrcsort(wlines)
          if ylines is not None:
            dummycurrline, dummycurlline, curcline = ll.lrcsort(ylines)
          if mode == cfg.MODE_COMPLEX_LANE_FOLLOW and roi is not None:
            complexsteering, complexthrottle = ll.lrclines(currline, curlline, curcline, roi)
            VP = ll.is_vanishing_point()
            VP2 = ll.is_vanishing_point2()
        else:
          simplecl, lines, roi = ll.process_img(img)
          if lines is not None and roi is not None:
            currline, curlline, curcline = ll.lrcsort(lines)
            if mode == cfg.MODE_COMPLEX_LANE_FOLLOW and lines is not None and roi is not None:
              currline, curlline, curcline = ll.lrcsort(lines)
              if cfg.USE_COLOR:
                # img used to compute line colors
                complexsteering, complexthrottle = ll.lrclines(currline, curlline, curcline, img)
              else:
                # roi only used for debugging
                complexsteering, complexthrottle = ll.lrclines(currline, curlline, curcline, roi)
              VP = ll.is_vanishing_point()
              VP2 = ll.is_vanishing_point2()
              if VP:
                ll.vp_confirmed()

        # print("1 simplecl: ", simplecl)
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

        ####
        # compute reward
        ####
        laneWidth, pixPerFrame, donkeyvpangle, bestll, bestcl, bestrl, width, curpos = ll.get_map_data()

        if simplecl is not None:
          # width/2 is mid-pixel
          dist_from_center = abs(ll.closestX(simplecl) - (width/2))
          print("1 dist_from_center %f (simplecl)" % dist_from_center)
        # elif lines is not None and wlines is not None and ylines is not None:
        elif bestcl is not None or bestrl is not None or bestll is not None:
          if bestcl is not None:
            dist_from_center = abs(ll.closestX(bestcl) - (width/2))
            print("2 dist_from_center %f" % dist_from_center)
          elif bestll is not None and bestrl is not None:
            dist_from_center = abs((ll.closestX(bestll) + ll.closestX(bestrl))/2 - (width/2))
            print("3 dist_from_center %f" % dist_from_center)
          elif bestll is not None:
            dist_from_center = abs((ll.closestX(bestll) + laneWidth) - (width/2))
            print("4 dist_from_center %f" % dist_from_center)
          elif bestrl is not None:
            dist_from_center = abs((ll.closestX(bestrl) - laneWidth) - (width/2))
            print("5 dist_from_center %f" % dist_from_center)
          else:
            dist_from_center = width
            print("6 dist_from_center %f" % dist_from_center)
        else:
          # Major Penalty, resulting in 0 steering, 0 throttle
          # ARD emergency_stop: make sure we have POV before continuing?
          print("reward = EMERG_STOP")
          # min reward
          return VP, VP2, steering, throttle, cfg.EMERGENCY_STOP
        if dist_from_center < cfg.MIN_DIST_FROM_CENTER:
          dist_from_center = 0
          # print("7 dist_from_center %f" % dist_from_center)
        # steering = min(max((ll.closestX(cl) - midpix) / denom,-1),1)
        # reward_center is based upon post-action ROI (n+1)
        # print("reward_center %f laneWidth %f" % (reward_center, laneWidth))
        # reward_throttle is based upon pre-action ROI (n) at ControlPI:
        #   reward_throttle = throttle - minthrottle - battery_adjustment + 1

        # normalize reward_center & reward_throttle betweeen (0,1]
        reward_center = min(max(((3*laneWidth/4 - dist_from_center)/(3*laneWidth/4)),cfg.EMERGENCY_STOP) , 1) * 0.8
        reward_throttle = throttle
        # normalize total reward between (0,1]
        reward = (reward_center) * (1 + (cfg.THROTTLE_BOOST*throttle))
        print("reward %f reward_center %f reward_throttle %f laneWidth %f" % (reward, reward_center, reward_throttle, laneWidth))
        reward = reward*1000

        # print("STEER %f THROT %f" % (steering, throttle))
        return VP, VP2, steering, throttle, reward

    def setFPSHz(hz):
      # needs to be done as a reurn parameter
      pass

    def run_nn(self, img, RL_ready):
      global KRLC, TB, MsgClnt2, MsgSvr2
      # print("RUN_NN")
      if TB.throttleCheckInProgress():
        TB.setMinMaxThrottle(img)
      minthrottle, maxthrottle, battery_adjustment = TB.getThrottleInfo()
      # ARD debugging
      # hardcoding angle, throttle works
      # angle = 7
      # throttle = 0
      # calling keras will lose connection with Unity
      # angle, throttle = KRLC.predict(img)
      if cfg.PORT_CONTROLPI2 is None:
        # print("local trial_run")
        predicted_angle, predicted_throttle, trial_angle, trial_throttle = KRLC.trial_run(img, RL_ready)
        cur_model_state = cfg.STATE_TRIAL_NN
      else:
        # print("MsgClnt2.sendmsg_roi")
        MsgClnt2.sendmsg_roi(img, RL_ready)
        predicted_angle, predicted_throttle, trial_angle, trial_throttle, trial_reward = MsgSvr2.recvmsg_angle_throttle_reward()
      # angle, throttle, reward is returned in float to control pi
      print("run_nn trial_run angle %f throttle %f ; minth %f batadj %f" % (trial_angle, trial_throttle, minthrottle, battery_adjustment))
      return predicted_angle, predicted_throttle, trial_angle, trial_throttle

    def run(self, img):
      return self.run_by_state(img)

    def __init__(self):
      global model_state, KRLC, cfg, TB, last_model_update
      global imgid, MsgSvr, MsgClnt, MsgClnt2, MsgSvr2, ll
      global RLPiState, RLPi_msgcnt, RLPi_weightcnt

      RLPi_weightcnt = 0
      last_model_update =  0
      cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))
      TB = ThrottleBase()
      ll = LaneLines(TB)
      KRLC = KerasRLContinuous(LaneLine = ll)
      imgid = 0
      RLPi_msgcnt = imgid
      # self.setFPSHz(10)
      # cfg.STATE_EMERGENCY_STOP = 0
      model_state = cfg.EMERGENCY_STOP
      MsgClnt = MessageClient(portid=cfg.PORT_RLPI, nonblocking = True)
      MsgSvr = MessageServer(portid=cfg.PORT_CONTROLPI, nonblocking = True)
      if cfg.PORT_CONTROLPI2 is not None:
        print("MsgClnt2")
        MsgClnt2 = MessageClient(portid=cfg.PORT_CONTROLPI2RL, nonblocking = False)
        MsgSvr2 = MessageServer(portid=cfg.PORT_CONTROLPI2, nonblocking = False)
      RLPiState = cfg.RLPI_OPENCV

