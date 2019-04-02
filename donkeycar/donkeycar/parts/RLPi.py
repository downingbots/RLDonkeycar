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
import zmq
import time
import numpy as np
from donkeycar.parts.RLMsg import MessageServer, MessageClient
from donkeycar.parts.RLKeras import KerasRLContinuous
from donkeycar.parts.RLOpenCV import ThrottleBase, HoughBundler, LaneLines
import donkeycar as dk

class RLServer():

    def cache_state(self, state):
      global cfg
      
      if state[self.STATE] == cfg.STATE_TRIAL_NN or state[self.OPENCV_REWARD] is None:
        # compute predicted reward from actor NN @ Control Pi 
        # with real OpenCV-based rwd
        # Note that this reward is actual the result of previous action!
        opencv_reward = self.Model.rl_compute_reward(state[self.ROI], state[self.ACTUAL_THROTTLE], state[self.ACTUAL_ANGLE])
        if opencv_reward == cfg.EMERGENCY_STOP:
          self.MsgClnt.sendmsg_emergency_stop(state[self.IMGID])
        # Critic is only run at RLPI
        # predicted_total_reward = state[self.PREDICTED_TOTAL_REWARD]
        predicted_total_reward = self.Model.predicted_reward(state[self.ROI])
        # print("predicted_total_reward")
        # print(predicted_total_reward)
        predicted_angle = state[self.PREDICTED_ANGLE]
        # print("predicted_angle")
        # print(predicted_angle)
        predicted_throttle = state[self.PREDICTED_THROTTLE]
        # print("predicted_throttle")
        # print(predicted_throttle)
        # could also compute critic total reward at this point 
        # so deq would only fit actor/critic in rl_fit()
      else:
        opencv_reward = state[self.OPENCV_REWARD]
        predicted_total_reward = self.Model.predicted_reward(state[self.ROI])
        predicted_angle, predicted_throttle = self.Model.predicted_actions(state[self.ROI])
        if predicted_angle < -1:
          print("Predicted angle too low: %f" % predicted_angle)
        if predicted_angle > 1:
          print("Predicted angle too high: %f" % predicted_angle)
      # self.imgid = imgid
      # while queue isn't too long
      # don't do if in "discard" mode
      if self.cachelen < cfg.Q_LEN_THRESH or (self.cachelen < cfg.Q_LEN_MAX and self.discarded_reward_num == 0) or self.cachelen < cfg.Q_LEN_THRESH:
        newentry = self.cachestart + self.cachelen 
        if newentry >= cfg.Q_LEN_MAX:
          newentry -= cfg.Q_LEN_MAX
        elif newentry < 0:
          # adding to empty queue 
          newentry = 0
          self.cachestart = 0
        self.clear_cache_entry(newentry)
        total_reward = None
        # self.cache[newentry] = [ imgid, state, angle, throttle, reward, roi]
        self.cache[newentry] = state
        print("new entry = %d" % newentry)
        self.opencvreward[newentry] = opencv_reward
        self.predictedtotalreward[newentry] = predicted_total_reward 
        self.predictedangle[newentry] = predicted_angle
        self.predictedthrottle[newentry] = predicted_throttle
        # print("self.predictedthrottle[newentry]")
        # print(self.predictedthrottle[newentry].shape)
        self.cachelen += 1
        self.batchlen += 1
        if opencv_reward < cfg.RWD_LOW or self.batchlen >= cfg.Q_FIT_BATCH_LEN_THRESH:
          if opencv_reward < cfg.RWD_LOW:
            print("completed sequence: low reward %d batchlen %d" % (opencv_reward, self.batchlen))
          # end of batch. compute total rewards
          # actually computing total reward for previous entry
          i = newentry - 1
          if i < 0:
            i = cfg.Q_LEN_MAX - 1
          k = 0
          self.actualbatchlen[i] = self.batchlen
          while self.batchlen - k - 1 > 0:
            rwd = self.opencvreward[i]
            if k > 0:
              total_reward += cfg.DISCOUNT_FACTOR * rwd
            else:
              total_reward = rwd
            k += 1
            if  opencv_reward < cfg.RWD_LOW or self.batchlen - k - 1 <= 0:
              self.actualtotalreward[i] = total_reward
              self.ready_cnt += 1
              print("1 r %d tr %d batchlen %d i %d k %d cs %d cl %d rc %d" % (rwd, total_reward, self.batchlen, i, k, self.cachestart, self.cachelen, self.ready_cnt))
            # print("2 r %d tr %d batchlen %d i %d k %d cs %d cl %d" % (rwd, total_reward, self.batchlen, i, k, self.cachestart, self.cachelen))
            i -= 1
            if i < 0:
              i = cfg.Q_LEN_MAX - 1
          if opencv_reward < cfg.RWD_LOW:
            # self.batchlen = 0
            # ARD: reward computed for prev entry, so is batchlen = 1 ?
            self.batchlen = 1
          else:
            self.batchlen = cfg.Q_FIT_BATCH_LEN_THRESH - 1
          # print("total reward %d" % total_reward)
          # print("cachestart total reward %d" % self.actualtotalreward[self.cachestart])
      elif self.batchlen > 0 and self.discarded_reward_num > 0 and self.cachelen >= cfg.Q_LEN_THRESH:
        print("discarding")
        # compute total reward of batch and drop entry
        if opencv_reward >= cfg.RWD_LOW and self.batchlen + self.discarded_reward_num < Q_FIT_BATCH_LEN_THRESH:
          self.discarded_reward[self.discarded_reward_num] = opencv_reward
          self.discarded_reward_num += 1
        elif self.discarded_reward_num > 0:
          # end of batch reached
          self.discarded_reward[self.discarded_reward_num] = opencv_reward
          self.discarded_reward_num += 1
          total_reward = opencv_reward
          # find beginning/oldest entry of batch 
          lastentry = self.cachestart + self.cachelen
          if lastentry >= cfg.Q_LEN_MAX:
            lastentry -= cfg.Q_LEN_MAX 
          total_reward = 0
          j = 0
          # compute total reward of reward-only cache
          while self.discarded_reward_num - j > 0:
            if j > 0:
              total_reward += cfg.DISCOUNT_FACTOR * self.discarded_reward[self.discarded_reward_num - j]
            else:
              total_reward = self.discarded_reward[self.discarded_reward_num - j]
            j += 1
          k = 0
          # actually computing total reward for previous entry
          i = newentry - 1
          if i < 0:
            i = cfg.Q_LEN_MAX - 1
            # i = cfg.Q_LEN_MAX
          k = 0
          while self.batchlen - k - 1 > 0:
            rwd = self.opencvreward[i]
            if k > 0:
              total_reward += cfg.DISCOUNT_FACTOR * self.discarded_reward[self.discarded_reward_num - k]
            else:
              total_reward += cfg.DISCOUNT_FACTOR * rwd
            k += 1
            # self.discarded_reward[self.discarded_reward_num] = reward
            if self.batchlen + self.discarded_reward_num - k >= cfg.Q_FIT_BATCH_LEN_THRESH:
              self.actualtotalreward[i] = total_reward
              self.ready_cnt += 1
              print("3 r %d tr %d batchlen %d i %d " % (opencv_reward, total_reward, self.batchlen, i))
            i -= 1
            if i < 0:
              i = cfg.Q_LEN_MAX - 1
          if opencv_reward < cfg.RWD_LOW:
            # self.batchlen = 0
            # ARD: reward computed for prev entry, so is batchlen = 1 ?
            self.batchlen = 1
          else:
            self.batchlen -= 1 # oldest batchlen now computed
      elif self.cachelen < cfg.Q_LEN_MAX:
        # just drop the message until dequeue goes below THRESHOLD
        print("discarding")
        pass 

    def batch_ready(self):
      if self.cachelen <= 0 or self.cachestart == -1:
        print("1 batch not ready")
        return False
      if self.actualtotalreward[self.cachestart] is None:
        print("2 batch not ready")
        # total_reward is computed when a batch is ready
        return False
      if self.ready_cnt < cfg.Q_FIT_BATCH_LEN_THRESH:
        print("3 batch not ready %d" % self.ready_cnt)
        return False
      # if self.batchlen < cfg.Q_FIT_BATCH_LEN_THRESH:
        # print("3 batch not ready %d" % self.ready_cnt)
        # return False
      print("4 batch ready %d" % self.ready_cnt)
      return True

    def dequeue_batch(self):
        global cfg

        if not self.batch_ready():
          return None

        tmp_batch = [[], [], [], [], [], [], []]
        PPO_SUPERVISED_TRAINING = -1000      # out of range value
        # print("start bl %d" % self.batchlen)
        batch_len = self.ready_cnt
        throttle_batch = []
        # for bl in range(1):
        for bl in range(cfg.Q_FIT_BATCH_LEN_THRESH):
           imgid, state, predicted_angle, predicted_throttle, trial_angle, trial_throttle, opencv_reward, actual_total_reward, predicted_total_reward, actual_batch_len, trial_roi = self.cache_dequeue()
           if imgid is None:
             return None
           if actual_batch_len is None or actual_batch_len < cfg.Q_MIN_ACTUAL_BATCH_LEN:
             continue
           tmp_batch[0].append(trial_roi)
           advantage = actual_total_reward - predicted_total_reward
           print("bl %d adv %d" % (self.batchlen, advantage))
           if state == cfg.STATE_NN:
             tmp_batch[1].append(advantage)
             tmp_batch[2].append(predicted_angle)
             tmp_batch[3].append(predicted_throttle)
           else:
             tmp_batch[1].append(PPO_SUPERVISED_TRAINING)
             tmp_batch[2].append(PPO_SUPERVISED_TRAINING)
             tmp_batch[3].append(PPO_SUPERVISED_TRAINING)
           tmp_batch[4].append(trial_angle)
           tmp_batch[5].append(trial_throttle)
           tmp_batch[6].append(actual_total_reward)
        # batch = [[], [], [], [], [], [], [], []]
        batch = [[], [], [], [], [], [], []]
        batch[0] = np.array(tmp_batch[0])
        batch[1] =  np.reshape(np.array(tmp_batch[1]),(len(tmp_batch[1]),))
        batch[2] =  np.reshape(np.array(tmp_batch[2]),(len(tmp_batch[2]),))
        batch[3] =  np.reshape(np.array(tmp_batch[3]), (len(tmp_batch[3]),1))
        batch[4] =  np.reshape(np.array(tmp_batch[4]),(len(tmp_batch[4]),1))
        batch[5] =  np.reshape(np.array(tmp_batch[5]),(len(tmp_batch[5]),1))
        batch[6] =  np.reshape(np.array(tmp_batch[6]),(len(tmp_batch[6]),1))
        # print("predicted_angle")
        # print(batch[2])
        # print("predicted_throttle")
        # print(batch[3])
        # print("actual_angle")
        # print(batch[4])
        # print("actual_throttle")
        # print(batch[5])
        # batch[7] =  np.zeros(len(tmp_batch[6]))
        # for i in range(8):
        #   print("batch %d shape:"%i)
        #   print(batch[i].shape)
        # for i in range(8):
        #   if i == 0:
        #     continue
        #   print("batch %d:"%i)
        #   print(batch[i])
        # pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        # return np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4]), np.array(batch[5]), np.array(batch[6])
        # return np.array(batch[0]), np.reshape(np.array(batch[1]),(len(batch[1]),1)), np.reshape(np.array(batch[2]),(len(batch[2]),1)), np.reshape(np.array(batch[3]), (len(batch[3]),1)), np.reshape(np.array(batch[4]),(len(batch[4]),1)), np.reshape(np.array(batch[5]),(len(batch[5]),1)), np.reshape(np.array(batch[6]),(len(batch[6]),1))
        return batch

    def cache_dequeue(self):
      # if self.batchlen <= 0:
        # print("empty queue")
        # return None, None, None, None, None, None, None, None, None, None 
      if self.cachelen <= 0 or self.cachestart < 0:
        print("bad dequeue 1: empty queue")
        return None, None, None, None, None, None, None, None, None, None 
        # return None
      if self.actualtotalreward[self.cachestart] is None:
        # total_reward is computed when a batch is ready
        print("no batch ready: bad dequeue 2")
        print("cache start %d c len %d b len %d rdy %d" % (self.cachestart, self.cachelen, self.batchlen, self.ready_cnt))
        print(self.actualtotalreward[self.cachestart])
        print(self.actualtotalreward[self.cachestart+1])
        # return None
        return None, None, None, None, None, None, None, None, None, None 
      self.ready_cnt -= 1
      i = self.cachestart 
      print("dequeue entry = %d" % i)
      self.cachelen -= 1
      self.cachestart += 1
      if self.cachestart >= cfg.Q_LEN_MAX:
        self.cachestart -= cfg.Q_LEN_MAX
      if self.cachelen <= 0:
        self.cachestart = -1
        self.batchlen = 0
      # imgid, state, trial_angle, trial_throttle, trial_reward, trial_roi, totl_reward 
      # print("cache")
      # print( self.cache[i][0]) 
      # print(self.cache[i][1])
      # print( self.cache[i][2])
      # print( self.cache[i][3])
      # print( self.cache[i][4]) 
      # print(self.cache[i][5]) 
      # print(self.cache[i][6]) 
      # if self.cache[i][7] is None: 
      #   print("ROI is None")
      # print("opencvreward")
      # print(self.opencvreward[i]) 
      # print(self.actualtotalreward[i])
      # print(self.predictedtotalreward[i])
      # print(self.predictedangle[i])
      # print("self.predictedthrottle[i]")
      # print(self.predictedthrottle[i])

      return self.cache[i][self.IMGID], self.cache[i][self.STATE], self.predictedangle[i], self.predictedthrottle[i], self.cache[i][self.ACTUAL_ANGLE], self.cache[i][self.ACTUAL_THROTTLE], self.opencvreward[i], self.actualtotalreward[i], self.predictedtotalreward[i], self.actualbatchlen[i], self.cache[i][self.ROI] 

    def clear_cache_entry(self, i):
      self.cache[i] = None
      self.predictedangle[i] = None
      self.predictedthrottle[i] = None
      self.opencvreward[i] = None
      self.actualtotalreward[i] = None
      self.actualbatchlen[i] = None
      self.predictedtotalreward[i] = None

    def __init__(self):
      global cfg

      self.IMGID                  = 0
      self.STATE                  = 1
      self.PREDICTED_ANGLE        = 2
      self.PREDICTED_THROTTLE     = 3
      self.ACTUAL_ANGLE           = 4
      self.ACTUAL_THROTTLE        = 5
      self.OPENCV_REWARD          = 6
      self.ROI                    = 7
      self.ACTUAL_TOTAL_REWARD    = 8
      self.PREDICTED_TOTAL_REWARD = 9

      cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))
      self.DBG = False
      context = zmq.Context()
      blocking = False
      self.cache = [[]]
      self.predictedangle = []
      self.predictedthrottle = []
      self.opencvreward = []
      self.predictedtotalreward = []
      self.actualtotalreward = []
      self.actualbatchlen = []
      self.discarded_reward = []
      self.cachelen   = 0
      self.cachestart = -1
      self.batchlen = 0
      self.discarded_reward_num = 0
      self.ready_cnt = 0

      self.TB = ThrottleBase()
      self.ll = LaneLines(self.TB)
      self.MsgSvr = MessageServer(portid = cfg.PORT_RLPI, nonblocking = True)
      self.MsgClnt = MessageClient(portid = cfg.PORT_CONTROLPI)
      self.Model = KerasRLContinuous(LaneLine=self.ll)

      for i in range(cfg.Q_LEN_MAX):
        self.cache.append(None)
        self.predictedtotalreward.append(None)
        self.opencvreward.append(None)
        self.predictedangle.append(None)
        self.predictedthrottle.append(None)
        self.actualtotalreward.append(None)
        self.actualbatchlen.append(None)
      for i in range(cfg.Q_FIT_BATCH_LEN_THRESH):
        self.discarded_reward.append(None)
      roi_cnt = 0
      trial_angle = 0 
      trial_throttle = 0 
      trial_roi = None
      previmgid = None

      self.imgid = -1
      previmgid = -1
      while True:
        # print("loop")
        # msgtype = self.MsgSvr.recv_msgtype()
        
          # no new message to process, process a ready entry from the cache
          # cache_dequeue returns: imgid, state, angle, throttle, reward, img , total reward
        self.imgid = self.MsgSvr.get_imgid()
        if self.imgid == previmgid:
          if self.batch_ready():
            # print("batch ready")
            self.Model.rl_fit( self.dequeue_batch())
          else:
            time.sleep(0.1)
            # time.sleep(0)
        else:
          # print("cache MSG_STATE_ANGLE_THROTTLE_REWARD_ROI")
          self.cache_state( self.MsgSvr.get_state_angle_throttle_reward_roi())
          # os.path.expanduser(cfg.RL_MODEL_PATH)
          previmgid = self.imgid
        if self.MsgSvr.send_weights():
          print("MSG_SEND_WEIGHTS")
          if not self.MsgSvr.same_donkey_state():
            wcnt, wvar, wmean, ycnt, yvar, ymean, scnt, svar, smean, lw = self.MsgSvr.get_donkey_state()
            self.ll.setDonkeyState(wcnt, wvar, wmean, ycnt, yvar, ymean, scnt, svar, smean, lw )
          self.MsgClnt.sendmsg_weights(roi_cnt, self.Model.get_weights("actor"))
          self.MsgSvr.weights_sent()

RLS = RLServer()
