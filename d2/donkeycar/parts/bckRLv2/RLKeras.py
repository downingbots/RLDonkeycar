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
import donkeycar as dk
from donkeycar.parts.RLOpenCV import ThrottleBase, HoughBundler, LaneLines

class KerasPilot():
 
    def load(self, model_path):
        print("LOADING %s" % model_path)
        self.model = keras.models.load_model(model_path)

    
    def shutdown(self):
        pass
    
    
    def clone(self, model):
        model_copy = keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        return model_copy

    def set_weights(self, w):
        return self.model.set_weights(w)

    def get_weights(self):
        return self.model.get_weights()

    def clear_session(self):
        # otherwise get_weights keeps consuming memory
        return self.model.backend.clear_session()

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
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_throttle = self.model.predict(img_arr, batch_size=1)
        # angle_throttle = self.model.predict_proba(img_arr)
        # angle_binned, throttle_binned = self.model.predict(img_arr)
        rdm = np.random.random() * 10
        if rdm <= 1:
          print("angle_throttle")
          print(angle_throttle)
        '''
        print("angle_throttle")
        print(angle_throttle)
        '''

        # max_angle_throttle = max(angle_throttle)
        index_angle_throttle = np.argmax(angle_throttle)
        # print("max angle_throt iter = %d reward = %f" % (index_angle_throttle, angle_throttle[[index_angle_throttle]]))  
        print("max angle_throt iter = %d " % (index_angle_throttle))
        # keep same throttles batched together
        # throttle_binned = int(index_angle_throttle / 15)
        angle_binned = (index_angle_throttle % 15)
        # print('throttle', throttle)
        angle_unbinned = (angle_binned - 6) / 7
        # throttle_unbinned = throttle_binned * cfg.THROTTLE_INCREMENT
        throttle_unbinned = 0
        print("t_b %d t_unb %f a_b %d a_unb %f" % (throttle_binned, throttle_unbinned, angle_binned, angle_unbinned))
        MAXUNBINNED = 14 * cfg.THROTTLE_INCREMENT
        if (trial_gps):
          rdm = np.random.random() * 10
          # random delta throttle: +2 / +1/  0/ -1
          #                        10%/50%/30%/10%
          # If steering angle > 20:
          # random delta throttle: +1 / +1/  0/ -1
          if rdm < 1:
            # always: angle_unbinned <= 15
            # if (angle_unbinned <= 20):
            if (abs(angle_unbinned) <= 1/7):
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
          # return unbinned values back to the Control Pi
          # angle_unbinned = max(min(angle_unbinned + delta_steering, 0),15)
          # angle_unbinned = max(min(angle_unbinned + delta_steering, -7),7)
          # angle_unbinned = (angle_binned + delta_steering - 7) / 7
          angle_unbinned = (angle_binned + delta_steering - 6) / 7
          angle_unbinned = min(max(angle_unbinned, -1),1)
          # throttle_unbinned = min(max(throttle_unbinned + delta_throttle*cfg.THROTTLE_INCREMENT, 0),MAXUNBINNED)
          print("t_unb %f t_delta %f a_unb %f a_delta %d" % (throttle_unbinned, delta_throttle, angle_unbinned, delta_steering))
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
        reward_center = min(max(((3*laneWidth/4 - dist_from_center)/(3*laneWidth/4)),cfg.EMERGENCY_STOP) , 1)
        reward_throttle = (actual_throttle / cfg.THROTTLE_INCREMENT) / 15
        # normalize total reward between (0,1]
        # reward = (reward_center*2 + reward_throttle) / 3
        reward = reward_center
        print("reward %f reward_center %f reward_throttle %f laneWidth %f" % (reward, reward_center, reward_throttle, laneWidth))
        return reward
       
    # params sent in msg from control_pi
    # run at RL_pi
    def rl_fit(self, trial_roi, results_roi, trial_angle, trial_throttle, saved_model_path):
      global cfg, trial_cnt
      # note: trial_throttle is the raw 0-14 bin , factoring out minthrottle
      # and battery adjustment:
      #   trial_bin = throttle - minthrottle - battery_adjustment + 1

      # too expensive to compute NN, hard-code shape instead
      # angle_throttle = model.predict(trial_roi)[0]
      # flattened number of potential outputs = 15 * 15 = 225
      # angle_throttle = [0] * 225
      angle_throttle = np.zeros(225)
      reward = self.rl_compute_reward(results_roi, trial_throttle, trial_angle)
      # 15 bins each. Start with a max increase of throttle of 6
      throttle_bin = int(trial_throttle / cfg.THROTTLE_INCREMENT)
      # angle_bin = int(trial_angle / cfg.ANGLE_INCREMENT) - 1
      # angle_bin = int((trial_angle + 1)*15/2)
      angle_bin = int((trial_angle * 7) + 7.5)

      # keep same throttles batched together
      trial_cnt += 1
      print("cnt %d reward %f tt %f tb %d ta %f ab %d" % (trial_cnt, reward, trial_throttle, throttle_bin, trial_angle, angle_bin))
      # index = (throttle_bin * 15) + angle_bin
      index = angle_bin
      if index > 224:
        index = 224
        print("ERROR: throttle bin %d ; angle_bin %d" %(throttle_bin, angle_bin))
      angle_throttle[index] = reward * 1.0
      # self.model.fit(trial_roi.reshape((1, 120, 160, 3)), angle_throttle.reshape(-1, 225), epochs=1, verbose=0)
      # self.model.fit(trial_roi.reshape((1, 120, 160, 3)), angle_throttle.reshape(-1, 225), batch_size=1, epochs=1, verbose=0)
      trial_roi = trial_roi.reshape((1,) + trial_roi.shape)

      # ValueError: Error when checking target: expected angle_throttle to have shape (None, 225) but got array with shape (225, 1)

      angle_throttle = angle_throttle.reshape(-1,225)
      self.model.fit(trial_roi, angle_throttle, batch_size=1, epochs=1, verbose=0)
      if (trial_cnt % 100 == 0):
        self.model.save(saved_model_path)
        '''
        # ARD: following not appropriate:
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path, 
                                                    monitor='val_loss', 
                                                    verbose=0, 
                                                    save_best_only=True, 
                                                    mode='min')
        '''
      # if (trial_cnt % 1000 == 0 and trial_cnt > 0):
      if (trial_cnt > cfg.SWITCH_TO_NN/3):
        angle_throttle = self.model.predict(trial_roi,batch_size=1)
        print("trial   angle_throt iter = %d t %d a %d" % (index, throttle_bin, angle_bin))
        index_angle_throttle = np.argmax(angle_throttle)
        # throttle_binned = int(index_angle_throttle / 15)
        throttle_binned = 0
        angle_binned = (index_angle_throttle % 15)
        print("predict angle_throt iter = %d t %d a %d" % (index_angle_throttle, throttle_binned, angle_binned))
      elif (trial_cnt % 100 == 0 and trial_cnt > 0):
      # if (True):
        # roi = trial_roi
        # roi = roi.reshape((1,) + roi.shape)
        # angle_throttle = self.model.predict_proba(roi)
        angle_throttle = self.model.predict(trial_roi,batch_size=1)
        print("angle_throttle")
        print(angle_throttle)
        print("trial   angle_throt iter = %d t %d a %d" % (index, throttle_bin, angle_bin))
        index_angle_throttle = np.argmax(angle_throttle)
        # throttle_binned = int(index_angle_throttle / 15)
        throttle_binned = 0
        angle_binned = (index_angle_throttle % 15)
        print("predict angle_throt iter = %d t %d a %d" % (index_angle_throttle, throttle_binned, angle_binned))


      return reward
        

def default_categorical():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense
    
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
    # Classify the data into 100 features, make all negatives 0
    x = Dense(100, activation='relu')(x)                                    
    # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
  # x = Dropout(.1)(x)                                                      
  # ARD: may not be required for RL
    # Classify the data into 50 features, make all negatives 0
    x = Dense(50, activation='relu')(x)                                     
    # Randomly drop out 10% of the neurons (Prevent overfitting)
  # x = Dropout(.1)(x)   
  # ARD: may not be required for RL
    #categorical output of the angle
    # Connect every input with every output and output 15 hidden units. 
    # Use Softmax to give percentage. 
    # 15 categories 
    # angle_out = Dense(15, activation='softmax', name='angle_out')(x)        
  # ARD: Change from continuous throttle to categorical
    # continous output of throttle
    # Reduce to 1 number, Positive number only
    # throttle_out = Dense(15, activation='softmax', name='angle_throttle(x)        
    angle_throttle = Dense(15*15, activation='softmax', name='angle_throttle')(x)        
    
    # model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model = Model(inputs=[img_in], outputs=[angle_throttle])
    model.compile(optimizer='adam',
                  loss={'angle_throttle': 'mean_squared_error'})
    return model
