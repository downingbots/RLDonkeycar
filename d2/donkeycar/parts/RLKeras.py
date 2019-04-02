'''
pilots.py

Methods to create, use, save and load pilots. Pilots 
contain the highlevel logic used to determine the angle
and throttle of a vehicle. Pilots can include one or more 
models to help direct the vehicles motion. 
'''

import os
import numpy as np
import keras
from keras import backend as K
import donkeycar as dk
from donkeycar.parts.RLOpenCV import ThrottleBase, HoughBundler, LaneLines
# import tensorflow as tf

class KerasPilot():
 
    def load(self, model_path):
        global graph
        print("LOADING %s" % model_path)
        self.model = keras.models.load_model(model_path)
        # self.model._make_predict_function()
        # graph = tf.get_default_graph()
    
    def shutdown(self):
        pass
    
    def clone(self, model):
        model_copy = keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        return model_copy

    def set_weights(self, w):
        global graph
        # K.clear_session()  # https://github.com/tensorflow/tensorflow/issues/14356
        # https://github.com/keras-team/keras/issues/2397
        # with graph.as_default():
        self.model.set_weights(w)
        # # self.model = self.model.load_model(model_path)
        # # self.model._make_predict_function()
        # # graph = tf.get_default_graph()
        # self.model.save('/home/ros/d2/models/rlpilot.control')
        '''
          self.model.save('/home/ros/d2/models/rlpilot.control')
          save_best = keras.callbacks.ModelCheckpoint('/home/ros/d2/models/rlpilot.control',
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      mode='min')
        '''

    def get_weights(self):
        return self.model.get_weights()

    # ARD debugging
    def predict(self, img):
        # https://github.com/keras-team/keras/issues/2397
        # global graph
        # with graph.as_default():
        # K.clear_session()  # https://github.com/tensorflow/tensorflow/issues/14356
          # self.model = keras.models.load_model('/home/ros/d2/models/rlpilot.control')
          # img_arr = img.reshape((1,) + img.shape)
          predicted_angle, predicted_throttle, predicted_reward = self.model.predict(img,batch_size=1)
          # self.model = keras.models.load_model('/home/ros/d2/models/rlpilot.control')
          # K.clear_session()  # https://github.com/tensorflow/tensorflow/issues/14356
          predicted_throttle /= 1000
          predicted_angle_bin = np.argmax(predicted_angle)
          print("predict %d a %d t %f r %f" % (trial_cnt, predicted_angle_bin, predicted_throttle, predicted_reward))
          return predicted_angle_bin, predicted_throttle, predicted_reward

    def clear_session(self):
        # otherwise get_weights keeps consuming memory
        return K.clear_session()

    def train(self, train_gen, val_gen, 
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of
        """

        #checkpoint to save model after each epoch
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path, 
                                                    monitor='val_loss', 
                                                    verbose=verbose, 
                                                    save_best_only=True, 
                                                    mode='min')
        
        #stop training if the validation error stops improving.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   min_delta=min_delta, 
                                                   patience=patience, 
                                                   verbose=verbose, 
                                                   mode='auto')
        
        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)
        
        hist = self.model.fit_generator(
                        train_gen, 
                        steps_per_epoch=steps, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_data=val_gen,
                        callbacks=callbacks_list, 
                        validation_steps=steps*(1.0 - train_split)/train_split)
        return hist


class KerasRLCategorical(KerasPilot):
    def __init__(self, model=None, LaneLine=None, *args, **kwargs):
        global ll, cfg, trial_cnt
        trial_cnt = 0
        ll = LaneLine
        cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))
        super(KerasRLCategorical, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = default_categorical()

        
    ################
    # CONTROL PI
    ################
    # trial_* are run at the control pi

    '''
    # now a parameter to trial_run
    def trial_gps(self):
        global gps_server_ready 
        if gps_server_ready == True:
          return True
        else:
          return False
    '''

    def trial_run(self, img_arr, trial_gps):
        global cfg, trial_cnt

        img_arr = img_arr.reshape((1,) + img_arr.shape)
        predicted_angle_bin, predicted_throttle, predicted_reward = super(KerasRLCategorical, self). predict(img_arr)
        rdm = np.random.random() * 10
        MAXUNBINNED = 14 * cfg.THROTTLE_INCREMENT
        if (trial_gps):
          rdm = np.random.random() * 10
          # random delta throttle: +2 / +1/  0/ -1
          #                        10%/50%/30%/10%
          # If steering angle > 20:
          # random delta throttle: +1 / +1/  0/ -1
          if rdm < 1:
            if (6 <= predicted_angle_bin <= 8):
              delta_throttle = 2
            else:
              delta_throttle = 1
          elif rdm < 6:
            delta_throttle = 1
          elif rdm < 9:
            delta_throttle = 0
          else:
            delta_throttle = -1
          rdm = np.random.random() * 10
          # random delta steering: -2 / -1/  0/ +1/ +2
          #   with distribution:   10% 20% 40% 20% 10%
          if rdm < 1:
            delta_steering = -2
          elif rdm < 3:
            delta_steering = -1 
          elif rdm < 5:
            delta_steering = 0 
          elif rdm < 9:
            delta_steering = 1
          else:
            delta_steering = 2
          # print("predicted a %d t %f r %f ; delta_a %d delta_t %d" % (predicted_angle_bin, predicted_throttle, predicted_reward, delta_steering, delta_throttle))
          print("predicted a %d t %f r %f ; delta_a %d delta_t %d" % (predicted_angle_bin, predicted_throttle, predicted_reward, delta_steering, delta_throttle))
          # return unbinned values back to the Control Pi
          angle_unbinned = (predicted_angle_bin + delta_steering - 7) / 7
          angle_unbinned = min(max(angle_unbinned, -1),1)
          throttle_unbinned = min(max(predicted_throttle + delta_throttle*cfg.THROTTLE_INCREMENT, 0),MAXUNBINNED)
        else:
          angle_unbinned = (predicted_angle_bin - 7) / 7
          throttle_unbinned = predicted_throttle
        return angle_unbinned, throttle_unbinned

    ################
    # Reinforcement Learning PI
    ################
    # rl_* are run at the RL pi
    def rl_compute_reward(self, img, actual_throttle, actual_steering):
        global width, laneWidth, ll
        global pixPerFrame, bestll, bestcl, bestrl, donkeyvpangle
    
        simplecl, lines, roi = ll.process_img(img)
        if lines is not None and roi is not None:
          steering, throttle = ll.lrclines(lines,roi)
        laneWidth, pixPerFrame, donkeyvpangle, bestll, bestcl, bestrl, width, curpos = ll.get_map_data()

        '''
        if laneWidth >0 and (curpos == 0 or curpos == 6): 
          # out of bounds
          print("reward = EMERG_STOP")
          # min reward
          return cfg.EMERGENCY_STOP
        '''
        if simplecl is not None:
          dist_from_center = abs(ll.closestX(simplecl) - (width/2))
          print("1 dist_from_center %f" % dist_from_center)
        elif lines is not None:
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
          return cfg.EMERGENCY_STOP
        if dist_from_center < cfg.MIN_DIST_FROM_CENTER:
          dist_from_center = 0
          # print("7 dist_from_center %f" % dist_from_center)
        # steering = min(max((ll.closestX(cl) - midpix) / denom,-1),1)
        # reward_center is based upon post-action ROI (n+1)
        # print("reward_center %f laneWidth %f" % (reward_center, laneWidth))
        # reward_throttle is based upon pre-action ROI (n) at ControlPI:
        #   reward_throttle = throttle - minthrottle - battery_adjustment + 1

        # normalize reward_center & reward_throttle betweeen (0,1]
        # reward_center = min((3*laneWidth/4 - dist_from_center),cfg.EMERGENCY_STOP) 
        reward_center = min(max(((3*laneWidth/4 - dist_from_center)/(3*laneWidth
/4)),cfg.EMERGENCY_STOP) , 0.99999)

        reward_throttle = (actual_throttle / cfg.THROTTLE_INCREMENT) / 15
        # normalize total reward between (0,1]
        reward = (reward_center*2 + reward_throttle) / 3
        print("reward %f reward_center %f reward_throttle %f laneWidth %f" % (reward, reward_center, reward_throttle, laneWidth))
        # return reward 
        return reward * 1000
       
    # params sent in msg from control_pi
    # run at RL_pi
    def rl_fit(self, roi_cnt, trial_roi, results_roi, trial_angle, trial_throttle, saved_model_path):
      global cfg, trial_cnt
      # note: trial_angle is the raw 0-14 bin , 
      # throttle factors out minthrottle and battery adjustment:
      #   trial_bin = throttle - minthrottle - battery_adjustment + 1

      # flattened number of potential angle outputs = 15 ; make 1-hot
      reward = self.rl_compute_reward(results_roi, trial_throttle, trial_angle)

      trial_cnt += 1
      trial_roi = trial_roi.reshape((1,) + trial_roi.shape)
      trial_angle_bin = int((trial_angle * 7) + 7.5)

      good_fit = True
      trial_cnt = roi_cnt
      # trial_cnt += 1
      if (trial_cnt > cfg.SWITCH_TO_NN):
        print("trial   %d a %d t %f r %f" %(trial_cnt, trial_angle_bin, trial_throttle, reward))
        predicted_angle, predicted_throttle, predicted_reward = self.model.predict(trial_roi,batch_size=1)
        predicted_throttle /= 1000
        predicted_angle_bin = np.argmax(predicted_angle)
        print("predict %d a %d t %f r %f" % (trial_cnt, predicted_angle_bin, predicted_throttle, predicted_reward))

        if predicted_angle_bin == trial_angle_bin and predicted_throttle == trial_throttle:
          # valid training data
          good_fit = True
        elif reward < predicted_reward:
          good_fit = False
          print("BAD FIT %d" % trial_cnt)
      elif (trial_cnt % cfg.UPDATE_NN == 0 and trial_cnt > 0):
        print("trial   %d a %d t %f r %f" %(trial_cnt, trial_angle_bin, trial_throttle, reward))
        predicted_angle, predicted_throttle, predicted_reward = self.model.predict(trial_roi,batch_size=1)
        predicted_throttle /= 1000
        predicted_angle_bin = np.argmax(predicted_angle)
        print("predict %d a %d t %f r %f" % (trial_cnt, predicted_angle_bin, predicted_throttle, predicted_reward))

        print(predicted_angle)
      else:
        print("trial   %d a %d t %f r %f" %(trial_cnt, trial_angle_bin, trial_throttle, reward))
      if good_fit:
        # make 1-hot angle category
        angle_out = np.zeros(15)
        angle_out[trial_angle_bin] = 1
        angle_out = angle_out.reshape(-1,15)
        print(angle_out)
        # print("angle_bin %d trial_angle %f" % (trial_angle_bin, trial_angle))
        tt = np.zeros(1)
        tt[0] = trial_throttle * 1000
        tt = tt.reshape(-1,1)
        r = np.zeros(1)
        r[0] = reward
        r = r.reshape(-1,1)
        self.model.fit({'img_in': trial_roi}, {'angle_out':angle_out, 'throttle_out':tt, 'reward_out':r}, batch_size=1, verbose=0)
        # self.model.fit({'img_in': trial_roi}, {'angle_out':angle_out, 'throttle_out':tt, 'reward_out':r}, batch_size=1, epochs=1, verbose=0)

      if (trial_cnt % cfg.UPDATE_NN == 0):
        self.model.save(saved_model_path)
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path,
                                                    monitor='val_loss',
                                                    verbose=0,
                                                    save_best_only=True,
                                                    mode='min')
      return reward
        

def default_categorical():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense
    global graph
    
    img_in = Input(shape=(120, 160, 3), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    # Flatten to 1D (Fully connected)
    x = Flatten(name='flattened')(x)                                        

    '''  Original Model. Dropout not required for RL.
    # Classify the data into 100 features, make all negatives 0
    # x = Dense(100, activation='relu')(x)                                    
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    # x = Dropout(.1)(x)                                                      
    # Classify the data into 50 features, make all negatives 0
    # x = Dense(50, activation='relu')(x)                                     
    # Randomly drop out 10% of the neurons (Prevent overfitting)
    # x = Dropout(.1)(x)   
    # Connect every input with every output and output 15 hidden units. 
    # Use Softmax to give percentage. 15 categories 
    # angle_out = Dense(15, activation='softmax', name='angle_out')(x)        
    # continous output of throttle:  Reduce to 1 number, Positive number only
    # model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    # model.compile(optimizer='adam',
    #               loss={'angle_out': 'categorical_crossentropy',
    #                     'throttle_out': 'mean_absolute_error'},
    #               loss_weights={'angle_out': 0.9, 'throttle_out': .001})
    '''

    # Classify the data into 100 features, make all negatives 0
    # x = Dense(100, activation='relu')(x)                                    
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    # x = Dropout(.1)(x)                                                      
    # Classify the data into 50 features, make all negatives 0
    # x = Dense(50, activation='relu')(x)                                     
    # Randomly drop out 10% of the neurons (Prevent overfitting)
    # Changes for RL
    x = Dense(15, activation='relu')(x)                                    
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)
    reward_out = Dense(1, activation='relu', name='reward_out')(x)      
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out, reward_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy', 
                        'throttle_out': 'mean_squared_error',
                        'reward_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 1, 'throttle_out': .1, 'reward_out':.1})
                  # loss_weights={'angle_out': 0.9, 'throttle_out': .03, 'reward_out':.07})
    # https://github.com/keras-team/keras/issues/3181
    # model._make_predict_function()	# have to initialize before threading
    # graph = tf.get_default_graph()
    return model
