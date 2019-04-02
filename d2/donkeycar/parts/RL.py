'''

RL.py

ReinforcementLearning: use OpenCV for lane following initially to gather
training data. Start slowly, determin circuit, and incrementally improve 
speed and paths. Use RI to improve from there.

'''




import os
import numpy as np

import donkeycar as dk
# from donkeycar.parts.RLOpenCV import HoughBundler, LaneLines
from donkeycar.parts.RLControlPi import ControlPi


class RLPilot():

    def load(self, model_path):
        pass

    def shutdown(self):
        pass

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        pass


class RL(RLPilot):
    def __init__(self, model=None, *args, **kwargs):
        super(RL, self).__init__(*args, **kwargs)
        global steering, throttle, steering_hist, throttle_hist, speed, angle
        global CPi

        self.speed = 0.0
        self.angle = 0.0
        self.steering = 0.0
        self.throttle = 0.0
        # self.top_speed=4.0
        self.top_speed=1.0
        self.top_speed=0.1
        self.steering_hist = []
        self.throttle_hist = []
        print("RL init")
        CPi = ControlPi()

    def run(self, img):
        global CPi
        # print("CPI")
        return CPi.run(img)

