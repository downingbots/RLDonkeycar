#!/usr/bin/env python
'''
Evaluate
Create a server to accept image inputs and run them against a trained neural network.
This then sends the steering output back to the client.
Author: Tawn Kramer
'''
from __future__ import print_function
import os
import argparse
import sys
import numpy as np
import json
import time
import asyncore
import json
import socket
from PIL import Image
from io import BytesIO
import base64
import datetime

from donkeycar.parts.simulation import FPSTimer
from donkeycar.management.base import load_config
from donkeycar.management.tcp_server import IMesgHandler, SimServer
from donkeycar.utils import linear_unbin

class DonkeySimMsgHandler(IMesgHandler):

    STEERING = 0
    THROTTLE = 1

    def __init__(self, constant_throttle, kpart):
        self.constant_throttle = constant_throttle
        self.kpart = kpart
        self.sock = None
        self.timer = FPSTimer()
        self.image_folder = None
        self.verbose = True
        self.init_steer = 0
        self.resetting = False
        self.steering_scale = 1.0
        self.iSceneToLoad = 3  # 'generated_track'
        # self.fns = {'telemetry' : self.on_telemetry}
        self.fns = {'telemetry' : self.on_telemetry,
                    "scene_selection_ready" : self.on_scene_selection_ready,
                    "scene_names": self.on_recv_scene_names,
                    "car_loaded" : self.on_car_loaded }
        # env = gym.make("eeyore")

    #######
    # derived from donkey_sim.py
    def on_scene_selection_ready(self, data):
        print("SceneSelectionReady ")
        msg = { 'msg_type' : 'get_scene_names' }
        self.sock.queue_message(msg)

    def on_recv_scene_names(self, data):
        if data:
            names = data['scene_names']
            if self.verbose:
                print("SceneNames:", names)
            self.send_load_scene(names[self.iSceneToLoad])

    def send_load_scene(self, scene_name):
        msg = { 'msg_type' : 'load_scene', 'scene_name' : scene_name }
        self.sock.queue_message(msg)

    def on_car_loaded(self, data):
        if self.verbose:
            print("car loaded")
        self.loaded = True

    #######
    def on_connect(self, socketHandler):
        self.sock = socketHandler
        self.init_steer = 0
        self.timer.reset()

    def on_recv_message(self, message):
        self.timer.on_frame()
        if not 'msg_type' in message:
            print('expected msg_type field')
            return

        msg_type = message['msg_type']
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print('unknown message type', msg_type)

    def on_telemetry(self, data):
        global cfg

        if self.resetting:
          return
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        hit = data["hit"]

        #reset scene to start if we hit anything.
        if hit != "none":
            print("HIT SOMETHING")
            # self.send_exit_scene()
            self.send_reset_car()
            return

        #Cross track error not always present.
        #Will be missing if path is not setup in the given scene.
        #It should be setup in the 3 scenes available now.
        try:
          cte = data["cte"]
          if cte >= 5.0:    # max error
            self.send_reset_car()
            return
        except:
          pass
        print("cte",cte,"hit",hit)

        # self.predict(image_array)

        if self.init_steer < cfg.INIT_STEER_FRAMES:
          # override first few frames to start near center line
          self.init_steer += 1
          steering = -1
          throttle = .05
          print("init_steer %d" % self.init_steer)
        else:
          # optional change to pre-preocess image before NN sees it
          self.image_part = None
          if self.image_part is not None:
              image_array = self.image_part.run(image_array)
          steering, throttle = self.kpart.run(image_array)
          # ARD: TODO. Match changes in manage.py
          # pilot/modelstate (NN,OpenCV), minthrottle, maxthrottle, bat_adj
          # Use state to change the loop_hz

          # filter throttle here, as our NN doesn't always do a greate job
          # ARD: unfilter throttle to test autopilot throttle
          # throttle = self.throttle_control(last_steering, last_throttle, speed, throttle)

        if throttle == -1000 and steering == -1000:
          print("2 EMERGENCY STOP - RESET CAR")
          self.send_reset_car()
          # self.send_exit_scene()
          return

        # simulator will scale our steering based on it's angle based input.
        # but we have an opportunity for more adjustment here.
        steering *= self.steering_scale

        # send command back to Unity simulator
        self.send_control(steering, throttle)

    def predict(self, image_array):
        outputs = self.model.predict(image_array[None, :, :, :])
        print("ARD DBG: PREDICT")
        self.parse_outputs(outputs)
    
    def parse_outputs(self, outputs):
        print("ARD DBG: PARSE_OUTPUTS")
        res = []
        for iO, output in enumerate(outputs):            
            if len(output.shape) == 2:
                if iO == self.STEERING:
                    steering_angle = linear_unbin(output)
                    res.append(steering_angle)
                elif iO == self.THROTTLE:
                    throttle = linear_unbin(output, N=output.shape[1], offset=0.0, R=0.5)
                    res.append(throttle)
                else:
                    res.append( np.argmax(output) )
            else:
                for i in range(output.shape[0]):
                    res.append(output[i])

        self.on_parsed_outputs(res)
        
    def on_parsed_outputs(self, outputs):
        self.outputs = outputs
        steering_angle = 0.0
        throttle = 0.2

        if len(outputs) > 0:        
            steering_angle = outputs[self.STEERING]

        if self.constant_throttle != 0.0:
            throttle = self.constant_throttle
        elif len(outputs) > 1:
            throttle = outputs[self.THROTTLE] * conf.throttle_out_scale

        if throttle == -1000 and steering_angle == -1000:
          self.send_reset_car()
          # self.send_exit_scene()
        else:
          self.send_control(steering_angle, throttle)

    def send_control(self, steer, throttle):
        msg = { 'msg_type' : 'control', 'steering': steer.__str__(), 'throttle':throttle.__str__(), 'brake': '0.0' }
        #print(steer, throttle)
        print("2 STEER %f" % steer)
        self.sock.queue_message(msg)
         
    def send_reset_car(self):
        self.resetting = True
        msg = { 'msg_type' : 'reset_car' }
        self.sock.queue_message(msg)
        print("RESET CAR")
        self.timer.reset()
        self.resetting = False
        self.init_steer = 0
        time.sleep(4)
        print("RESET CAR TIMER DONE")

    def send_exit_scene(self):
        msg = { 'msg_type' : 'exit_scene' }
        print("EXIT SCENE")
        self.queue_message(msg)

    def on_disconnect(self):
        pass

class Sim():

    def parse_args(self, args):
        parser = argparse.ArgumentParser(description='sim')
        parser.add_argument('--model', help='the model to use for predictions')
        parser.add_argument('--config', default='~/donkeycar/donkeycar/parts/RLConfig.py', help='location of config file to use. default: ./config.py')
        parser.add_argument('--type', default='categorical', help='model type to use when loading. categorical|linear')
        parser.add_argument('--top_speed', default='3', help='what is top speed to drive')
        parsed_args = parser.parse_args(args)
        return parsed_args, parser

    def run(self, args):
        '''
        Start a websocket SocketIO server to talk to a donkey simulator
        '''
        global cfg
        import socketio
        from donkeycar.parts.simulation import SteeringServer
        from donkeycar.parts.Keras import KerasCategorical, KerasLinear
        from donkeycar.parts.RLKeras import KerasRLContinuous
        from donkeycar.parts.RL import RL

        args, parser = self.parse_args(args)

        cfg = load_config(args.config)

        if cfg is None:
            return

        #TODO: this logic should be in a pilot or modle handler part.
        if args.type == "categorical":
            kl = KerasCategorical()
        elif args.type == "linear":
            kl = KerasLinear(num_outputs=2)
        elif args.type == "rl":
            kl = RL()
        else:
            print("didn't recognice type:", args.type)
            return
        #can provide an optional image filter part
        img_stack = None

        #load keras model
        kl.load(args.model)

        address = ('0.0.0.0', 9091)
        #setup the server
        handler = DonkeySimMsgHandler(cfg.THROTTLE_CONSTANT, kl)
        server = SimServer(address, handler)

        try:
            #asyncore.loop() will keep looping as long as any asyncore dispatchers are alive
            asyncore.loop()
        except KeyboardInterrupt:
            #unless some hits Ctrl+C and then we get this interrupt
            print('stopping')

  
    
