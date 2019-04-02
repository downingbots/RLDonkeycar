'''
This variation is based upon:
https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/3-reinforce/cartpole_reinforce.py

We combine angle/throttle into a single 7*3 catagorical output
and use the loss function to define the reward.

---
Methods to create, use, save and load pilots. Pilots 
contain the highlevel logic used to determine the angle

models to help direct the vehicles motion. 

'''
# STATE_OPENCV:
#           supervised learning based upon OpenCV
#           OpenCv results incl reward @ control rpi & sent via msg
#           cache_state @ RLPi
#           rlfit: predict critic, train actor & critic @ deq @ RLPI
#              - note: OpenCv action not like actor results
#           Computation:
#              - Control: Open CV run once
#              - RLPi:    critic NN run @ enq batch 
#                         actor  NN run @ enq batch 
#                         actor and critic runs fit in deq batch
#           Proposed Computation:
#                         run actor predict, DUMMY, DUMMY during enqueue 
#                         run critic NN during enqueue 
#                         parallize actor and critic fits in deq batch
#                         3 Pi for Control, critic @ enq, actor @ enq
#                                  actor fit @ deq, critic fit @ deq
# STATE_TRIAL_NN:
#           RL training randomized trial_run(){actor} @ control pi
#           cache_state @ enq @ RLPI
#           compute_reward: OpenCv @ enq @ RLPI 
#           rlfit: predict critic, train actor & critic @ deq @ RLPI
#              - note: randomization of actor actions possible
#              - note: control pi actor will be slightly out-of-sync (OK)
#              - pass both predicted & real values to RLPI
#           Computation:
#              - Control: actor NN run once
#              - RLPi:    
#                         opencv run during enq
#                         critic NN run during enq
#                         batch actor and critic runs fit in deq batch
#           Proposed Computation:
#                         5 Pi for Control; critic @ enq; opencv@enq;
#                                  actor fit @ deq; critic fit @ deq
#                         3 Pi for Control; critic @ enq & opencv@enq;
#                            batch actor fit @ deq & critic fit @ deq
import os
import numpy as np
import keras
import donkeycar as dk
from keras.optimizers import Adam
from donkeycar.parts.RLOpenCV import ThrottleBase, HoughBundler, LaneLines
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, merge
from keras.layers import Input, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense

def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
      def loss(y_true, y_pred):

        import tensorflow as tf
        # return K.mean(K.square(y_pred - y_true), axis=-1)

        # adv = K.squeeze(advantage, axis = 1)
        # pred = K.squeeze(old_prediction, axis = 1)
        # adv_sum = K.sum(adv, axis=-1)
        # adv_mean = K.mean(adv, axis=-1)
        # pred_sum = K.sum(pred, axis=-1)
        # pred_mean = K.mean(pred, axis=-1)
        # if (pred_mean == PPO_OUT_OF_RANGE):
        # if (K.sum(adv_sum, -pred_sum) == K.sum(adv_mean, -pred_mean) and adv_sum != adv_mean):
        # if (K.equal(adv_sum, pred_sum) and K.equal(adv_mean, pred_mean) and K.not_equal(adv_sum, adv_mean)):
        # out_of_range = tf.constant(PPO_OUT_OF_RANGE, tf.shape(advantage))
        # out_of_range = K.constant(PPO_OUT_OF_RANGE, dtype=old_prediction.dtype, shape=old_prediction.shape)
        # pred_out_of_range = K.equal(old_prediction, out_of_range)
        # pred_out_of_range = K.equal(old_prediction, PPO_OUT_OF_RANGE)

        mean_sq_err = K.mean(K.square(y_pred - y_true), axis=-1)

        try:
          PPO_OUT_OF_RANGE = 1000    # negative of -1000
          checkifzero = K.sum(old_prediction, PPO_OUT_OF_RANGE)
          divbyzero = old_prediction / checkifzero
        except:
          return mean_sq_err
          
          
        # pred_out_of_range = K.mean((old_prediction / PPO_OUT_OF_RANGE), axis=-1)
        # pred_out_of_range = K.mean(K.equal(old_prediction , PPO_OUT_OF_RANGE), axis=-1)
        pred_out_of_range = K.mean(old_prediction, axis=-1)

        PPO_NOISE = 1.0
        var = keras.backend.square(PPO_NOISE)
        denom = K.sqrt(2 * np.pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        PPO_LOSS_CLIPPING = 0.2
        PPO_ENTROPY_LOSS = 5 * 1e-3 # Does not converge without entropy penalty
        # return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - PPO_LOSS_CLIPPING, max_value=1 + PPO_LOSS_CLIPPING) * advantage))
        # ppo_loss = -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - PPO_LOSS_CLIPPING, max_value=1 + PPO_LOSS_CLIPPING) * advantage)) + PPO_ENTROPY_LOSS * (prob * K.log(prob + 1e-10))
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - PPO_LOSS_CLIPPING, max_value=1 + PPO_LOSS_CLIPPING) * advantage)) + PPO_ENTROPY_LOSS * (prob * K.log(prob + 1e-10))

        # out = tf.where(tf.equal(pred_out_of_range, PPO_OUT_OF_RANGE), mean_sq_err,  ppo_loss)
        # out = K.switch(K.equal(-1000, PPO_OUT_OF_RANGE), mean_sq_err,  ppo_loss)
        # out = K.switch(K.equal(pred_out_of_range, PPO_OUT_OF_RANGE), mean_sq_err,  ppo_loss)
        # out = K.switch( pred_out_of_range, K.zeros_like(pred_out_of_range),  K.zeros_like(pred_out_of_range))
        # out = K.switch( K.mean(old_prediction/PPO_OUT_OF_RANGE), mean_sq_err,  ppo_loss)
        # return out
      return loss


class KerasPilot():
 
    def load(self, model, model_path=None):
      global cfg

      # ARD: TODO: try using save_weights and load_weights instead of the whole
      # model. use the model definition from this file.
      try:
        #load keras model
        if model_path is None:
          model_path = cfg.RL_MODEL_PATH
        print("LOADING %s_%s" % (model_path, model))
        if model == "critic":
          Model = keras.models.load_model(os.path.expanduser(model_path + "_" + model))
        else:
          advantage_in = Input(shape=(1,))
          old_prediction_in = Input(shape=(1,))
          Model = keras.models.load_model(os.path.expanduser(model_path + "_" + model), custom_objects={'loss': proximal_policy_optimization_loss_continuous( advantage=advantage_in, old_prediction=old_prediction_in)})
        print("loaded")
        return Model
      except:
        print("No model to load")
        return None

    def load_weights(self, model, model_path=None):
      global cfg

      # ARD: TODO: try using save_weights and load_weights instead of the whole
      # model. use the model definition from this file.
      try:
        #load keras model
        if model_path is None:
          model_path = cfg.RL_MODEL_PATH
        print("LOADING %s_%s" % (model_path, model))
        if model == "critic":
          self.critic.load_weights(os.path.expanduser(model_path + "_" + model))
        else:
          self.actor.load_weights(os.path.expanduser(model_path + "_" + model))
        print("loaded")
        return True
      except:
        print("No weights to load")
        # raise
        return False

    def set_weights(self, w, model):
        if model == "actor":
          return self.actor.set_weights(w)
        else:
          return self.critic.set_weights(w)

    def get_weights(self, model):
        if model == "actor":
          return self.actor.get_weights()
        else:
          return self.critic.get_weights()

    def clear_session(self, model):
        # otherwise get_weights keeps consuming memory
        if model == "actor":
          return self.actor.backend.clear_session()
        else:
          return self.critic.backend.clear_session()

    def predicted_actions(self, img):
        # print("predicted_actions shape %d" % len(img.shape))
        if len(img.shape) != 4:
          img_arr = img.reshape((1,) + img.shape)
        else:
          img_arr = img
        # PPO_DUMMY_ADVANTAGE, PPO_DUMMY_ANGLE, PPO_DUMMY_THROTTLE, PPO_DUMMY_ZEROS = np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
        # outputs = self.actor.predict([img_arr, PPO_DUMMY_ADVANTAGE, PPO_DUMMY_ANGLE, PPO_DUMMY_THROTTLE, PPO_DUMMY_ZEROS], batch_size=1)
        PPO_DUMMY_ADVANTAGE, PPO_DUMMY_ANGLE, PPO_DUMMY_THROTTLE = np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
        outputs = self.actor.predict([img_arr, PPO_DUMMY_ADVANTAGE, PPO_DUMMY_ANGLE, PPO_DUMMY_THROTTLE], batch_size=1)
        predicted_angle = outputs[0]
        if predicted_angle < -1:
          print("Predicted angle too low: %f" % predicted_angle)
        if predicted_angle > 1:
          print("Predicted angle too high: %f" % predicted_angle)
        # print("predicted actions")
        # print(outputs[0].shape)
        # print(outputs[1].shape)
        predicted_throttle = outputs[1]
        predicted_throttle /= 1000
        predicted_throttle = outputs[1].reshape(1,1)
        predicted_angle = min(max(predicted_angle, -1),1)
        predicted_throttle = max(predicted_throttle, 0)
        # print(predicted_throttle.shape)
        # print("predicted_actions")
        # print(predicted_angle)
        # print(predicted_throttle)
        return predicted_angle, predicted_throttle

    def predicted_reward(self, img):
        img_arr = img.reshape((1,) + img.shape)
        output = self.critic.predict([img_arr], batch_size=1)
        reward = output[0][0]
        return reward

class KerasRLContinuous(KerasPilot):

    def __init__(self, model=None, LaneLine=None, *args, **kwargs):
        global cfg
        self.trial_cnt = 0
        self.ll = LaneLine
        self.critic = None
        self.actor = None
        cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))
        super(KerasRLContinuous, self).__init__(*args, **kwargs)
        # following initializes self.actor / self.critic
        self.default_continuous(model)

        
    ################
    # CONTROL PI
    ################
    # trial_* are run at the control pi

    def trial_run(self, img_arr, trial_gps):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        # angle,throttle = self.model.predict(img_arr, batch_size=1)
        # PPO_DUMMY_ADVANTAGE, PPO_DUMMY_ANGLE, PPO_DUMMY_THROTTLE, PPO_DUMMY_ZEROS = np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
        # outputs = self.actor.predict([img_arr, PPO_DUMMY_ADVANTAGE, PPO_DUMMY_ANGLE, PPO_DUMMY_THROTTLE, PPO_DUMMY_ZEROS], batch_size=1)
        PPO_DUMMY_ADVANTAGE, PPO_DUMMY_ANGLE, PPO_DUMMY_THROTTLE = np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
        outputs = self.actor.predict([img_arr, PPO_DUMMY_ADVANTAGE, PPO_DUMMY_ANGLE, PPO_DUMMY_THROTTLE], batch_size=1)
        predicted_angle = outputs[0]
        predicted_throttle = outputs[1]
        predicted_throttle /= 1000
        # reward = outputs[2]
        rdm = np.random.random() * 10
        print("t_unb %f a_unb %f " % (predicted_throttle, predicted_angle))
        if (trial_gps):
          rdm = np.random.random() * 10
          # Training is done with throttle = LOW, so higher prob of MED/HIGH
          # random throttle: 0  / 1 /  2
          #                 10% /60%/30%
          if rdm < 1:
            delta_throttle = -1
          elif rdm < 2:
            delta_throttle = 0
          elif rdm < 8:
            delta_throttle = 1
          else:
            delta_throttle = 2
          trial_throttle = predicted_throttle + (delta_throttle * cfg.THROTTLE_INCREMENT)
          rdm = np.random.random() * 5
          if rdm <= 1:
            # fully random
            rdm = np.random.random() 
            delta_steering = 2*(rdm - .5)
            trial_angle = delta_steering
            print("random steering: %f" % trial_angle)
          else:
            # bounded random
            # scale = int(np.random.random() * 4) + 1
            scale = 1
            # random delta steering: -2 / -1/  0/ +1/ +2
            #   with distribution:   10% 20% 40% 20% 10%
            # rdm = np.random.random() * 10
            rdm = 6    # no change
            if rdm < 1:
              delta_steering = -2
            elif rdm < 3:
              delta_steering = -1 
            elif rdm < 7:
              delta_steering = 0 
            elif rdm < 9:
              delta_steering = 1
            else:
              delta_steering = 2
            scale = 4   # scale of randomization
            # maximize entropy
            if predicted_angle + scale * abs(delta_steering) / 15 < -1:
              trial_angle = -1 - scale * abs(delta_steering) / 15
            if predicted_angle + scale * abs(delta_steering) / 15 > 1:
              trial_angle = 1 - scale * abs(delta_steering) / 15
            else:
              # return unbinned values back to the Control Pi
              trial_angle = predicted_angle + scale * delta_steering / 15
            print("tpred %f apred %f ttrl %f atrl %f t_delta %d a_delta %d scale %f" % (predicted_throttle, predicted_angle, trial_throttle, trial_angle, delta_throttle, delta_steering, scale))
        else:
          trial_angle = predicted_angle
          trial_throttle = predicted_throttle
        # angle is passed as unbinned between processes to maintain precision
        # throttle is just low/med/high
        trial_angle = min(max(trial_angle, -1),1)
        trial_throttle = max(trial_throttle, 0)
        return predicted_angle, predicted_throttle, trial_angle, trial_throttle

    ################
    # Reinforcement Learning PI
    ################
    # rl_* are run at the RL pi
    def rl_compute_reward(self, img, actual_throttle, actual_steering):
    
        simplecl = None
        lines    = None
        ylines   = None
        wlines   = None
        curcline = []
        currline = []
        curlline = []
        if cfg.USE_COLOR:
          line_color_simple, line_color_yellow, line_color_white = self.ll.get_line_color_info()

        if cfg.USE_COLOR and line_color_yellow is not None and line_color_white is not None and line_color_simple is not None:
          simplecl, wlines, ylines, roi = self.ll.process_img_color(img)
          if wlines is not None:
            currline, curlline, dummycurcline = self.ll.lrcsort(wlines)
          if ylines is not None:
            dummycurrline, dummycurlline, curcline = self.ll.lrcsort(ylines)
          if roi is not None:
            steering, throttle = self.ll.lrclines(currline, curlline, curcline, roi)
        else:
          simplecl, lines, roi = self.ll.process_img(img)
          if lines is not None and roi is not None:
            currline, curlline, curcline = self.ll.lrcsort(lines)
            if cfg.USE_COLOR:
              # img used to compute line colors
              steering, throttle = self.ll.lrclines(currline, curlline, curcline, img)
            else:
              # roi only used for debugging
              steering, throttle = self.ll.lrclines(currline, curlline, curcline, roi)

        laneWidth, pixPerFrame, donkeyvpangle, bestll, bestcl, bestrl, width, curpos = self.ll.get_map_data()

        if not cfg.DISABLE_EMERGENCY_STOP and laneWidth >0 and (curpos == 0 or curpos == 6): 
          if simplecl is not None:
            curpos = 4
          else:
            # out of bounds
            print("reward = EMERG_STOP")
            # min reward
            return cfg.EMERGENCY_STOP
        '''
        '''
        if simplecl is not None:
          dist_from_center = abs(self.ll.closestX(simplecl) - (width/2))
          print("1 dist_from_center %f" % dist_from_center)
        elif (not cfg.USE_COLOR and lines is not None) or (cfg.USE_COLOR and (ylines is not None or wlines is not None)):
          if bestcl is not None:
            dist_from_center = abs(self.ll.closestX(bestcl) - (width/2))
            print("2 dist_from_center %f" % dist_from_center)
          elif bestll is not None and bestrl is not None:
            dist_from_center = abs((self.ll.closestX(bestll) + self.ll.closestX(bestrl))/2 - (width/2))
            print("3 dist_from_center %f" % dist_from_center)
          elif bestll is not None:
            dist_from_center = abs((self.ll.closestX(bestll) + laneWidth) - (width/2))
            print("4 dist_from_center %f" % dist_from_center)
          elif bestrl is not None:
            dist_from_center = abs((self.ll.closestX(bestrl) - laneWidth) - (width/2))
            print("5 dist_from_center %f" % dist_from_center)
          else:
            dist_from_center = width
            print("6 dist_from_center %f" % dist_from_center)
        else:
          # Major Penalty, resulting in 0 steering, 0 throttle
          # ARD emergency_stop: make sure we have POV before continuing?
          print("reward = EMERG_STOP")
          # min reward
          return cfg.EMERGENCY_STOP
        if dist_from_center < cfg.MIN_DIST_FROM_CENTER:
          dist_from_center = 0
          # print("7 dist_from_center %f" % dist_from_center)
        # steering = min(max((self.ll.closestX(cl) - midpix) / denom,-1),1)
        # reward_center is based upon post-action ROI (n+1)
        # print("reward_center %f laneWidth %f" % (reward_center, laneWidth))
        # reward_throttle is based upon pre-action ROI (n) at ControlPI:
        #   reward_throttle = throttle - minthrottle - battery_adjustment + 1

        # normalize reward_center & reward_throttle betweeen (0,1]
        reward_center = min(max(((3*laneWidth/4 - dist_from_center)/(3*laneWidth/4)),cfg.EMERGENCY_STOP) , 1) * 0.8
        reward_throttle = actual_throttle 
        # normalize total reward between (0,1]
        reward = (reward_center) * (1 + (cfg.THROTTLE_BOOST*actual_throttle))
        print("reward %f reward_center %f reward_throttle %f laneWidth %f" % (reward, reward_center, reward_throttle, laneWidth))
        return reward*1000
       
    # called when batch is ready at RLPI
    # run at RL_pi
    # def rl_fit(self, trial_roi_batch, advantage_batch, angle_batch, predicted_angle_batch, predicted_throttle_batch, trial_angle_batch, trial_throttle_batch, actual_total_reward_batch):
    def rl_fit(self, batch):
      global cfg

      if batch is None:
        return   # batch not ready
      self.trial_cnt += 1
      # trial_roi = trial_roi.reshape((1,) + trial_roi.shape)
      print("actor train_on_batch")
      # print(batch[0])
      # print(batch[0].shape)
      # print(batch[1].shape)
      # print(batch[1])
      # print(batch[2].shape)
      # # print(batch[2])
      # print(batch[3].shape) # (20,1)
      # # print(batch[3])
      # print(batch[4].shape)
      # print(batch[4])
      # print(batch[5].shape)
      # print(batch[5])
      # print(batch[7].shape)
      # print(batch[7])
      # self.actor.train_on_batch([batch[0],batch[1],batch[2],batch[3],batch[7]], [batch[4],batch[5]])
      self.actor.train_on_batch([batch[0],batch[1],batch[2],batch[3]], [batch[4],batch[5]])
      print("critic train_on_batch")
      # print(batch[0].shape)
      # print(batch[0])
      # print(batch[6].shape)
      # print(batch[6])
      self.critic.train_on_batch([batch[0]], [batch[6]])
      if (self.trial_cnt % cfg.SAVE_NN == 0):
        self.actor.save_weights(os.path.expanduser(cfg.RL_MODEL_PATH + "_actor"))
        self.critic.save_weights(os.path.expanduser(cfg.RL_MODEL_PATH + "_critic"))
        # self.actor.save(os.path.expanduser(cfg.RL_MODEL_PATH + "_actor"))
        # self.critic.save(os.path.expanduser(cfg.RL_MODEL_PATH + "_critic"))

##########################################
# PPO
##########################################
# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
# This variation derived from https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py

# move to RLConfig
# Only implemented clipping for the surrogate loss, paper said it was best
# PPO_LOSS_CLIPPING = 0.2 
# PPO_NOISE = 1.0
# PPO_GAMMA = 0.99
# PPO_DUMMY_ACTION, PPO_DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))
# PPO_ENTROPY_LOSS = 5 * 1e-3 # Does not converge without entropy penalty
# PPO_LR = 1e-4 # Lower lr stabilises training greatly


    '''
    def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
      def loss(y_true, y_pred):
        PPO_NOISE = 1.0
        var = keras.backend.square(PPO_NOISE)
        denom = K.sqrt(2 * np.pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        PPO_LOSS_CLIPPING = 0.2 
        PPO_ENTROPY_LOSS = 5 * 1e-3 # Does not converge without entropy penalty
        # return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - PPO_LOSS_CLIPPING, max_value=1 + PPO_LOSS_CLIPPING) * advantage))
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - PPO_LOSS_CLIPPING, max_value=1 + PPO_LOSS_CLIPPING) * advantage)) + PPO_ENTROPY_LOSS * (prob * K.log(prob + 1e-10))
      return loss
    '''

    ##########################################
    # MODEL USED BY CONTROLPI, RLPI (and controlpi2 if run)
    ##########################################
    def default_model(self, actor_critic = "actor"):
      from keras.models import Model
      from keras.layers import Input, Lambda
      from keras.layers import multiply, add
      
      PPO_LR = 1e-4 # Lower lr stabilises training greatly
      img_in = Input(shape=(120, 160, 3), name='img_in')
  
      if actor_critic == "actor":
        advantage_in = Input(shape=(1,))
        old_angle_prediction = Input(shape=(1,))
        old_throttle_prediction = Input(shape=(1,))
        # dummy_zeros = Input(shape=(1,))
        # continuous output of the angle, throttle and reward
        # lambda function uses the loss-only inputs: requirement for load
        # but lambda function is not trainable...
        # x = Lambda(lambda x: x[0])([img_in, advantage_in, old_angle_prediction, old_throttle_prediction])
        x = img_in
        # y = multiply([advantage_in, old_angle_prediction, old_throttle_prediction, dummy_zeros])
        # Convolution2D class name is an alias for Conv2D
        x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = Flatten(name='flattened')(x)
        angle_out = Dense(units=1, activation='linear', name='angle_out')(x)
        # angle_out = add([angle_out,y])
        throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

        # model = Model(inputs=[l_input], outputs=[angle_out, throttle_out])
        # model = Model(inputs=[img_in, advantage_in, old_angle_prediction, old_throttle_prediction, dummy_zeros], outputs=[angle_out, throttle_out])
        model = Model(inputs=[img_in, advantage_in, old_angle_prediction, old_throttle_prediction], outputs=[angle_out, throttle_out])
        # action, action_matrix, prediction from trial_run
        # reward is a function( angle, throttle)
        model.compile(optimizer=Adam(lr=PPO_LR), 
                      loss=[proximal_policy_optimization_loss_continuous( advantage=advantage_in, old_prediction=old_angle_prediction),
                            proximal_policy_optimization_loss_continuous( advantage=advantage_in, old_prediction=old_throttle_prediction)])
      elif actor_critic == "critic":
        y = img_in
        # Convolution2D class name is an alias for Conv2D
        y = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(y)
        y = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(y)
        y = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(y)
        y = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(y)
        y = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(y)
        y = Flatten(name='flattened')(y)
        total_reward = Dense(units=1, activation='linear', name='total_reward')(y)
        model = Model(inputs=[img_in], outputs=[total_reward])
        model.compile(optimizer=Adam(lr=PPO_LR), 
                      loss={'total_reward': 'mean_squared_error'})
      # model.summary()
  
      return model

    def default_continuous(self, mymodel=None):
      if (mymodel is None):
        # self.critic = self.load("critic")
        # print(self.critic)
        # print(self.actor)
        if self.critic is None:
          self.critic = self.default_model("critic")
          self.load_weights("critic")
        # self.actor = self.load("actor")
        if self.actor is None:
          self.actor = self.default_model("actor")
          self.load_weights("actor")
        print(" ")
        print("ACTOR MODEL SUMMARY:")
        self.actor.summary()
        print(" ")
        print("CRITIC MODEL SUMMARY:")
        self.critic.summary()
      else:
        if mymodel == "actor" or mymodel == "control":
          # self.actor = self.load(mymodel)
          if self.actor is None:
            self.actor = self.default_model(mymodel)
            self.load_weights(mymodel)
          print(" ")
          print("ACTOR MODEL SUMMARY:")
          self.actor.summary()
        else:
          # self.critic = self.load("critic")
          if self.critic is None:
            self.critic = self.default_model("critic")
            self.load_weights("critic")
          print(" ")
          print("CRITIC MODEL SUMMARY:")
          self.critic.summary()
