'''

camera has 160 degrees
 => 1 degree per pixel?
 -> 80 degree angle to each corner?

When have vanishing point:
 - Add intersection of Horiz line to get pixels and width of track
     - Get line segment from (80,<60>) to VP?
     - (when almost on center line and facing POV angle)
 - car line is straight to VP. Make L/C/R parallel.
     - vert line to compute angle to L/C/R/VP
     - based on car angle, make 90 deg ange to find intercept with L/R/C lines
       - use car angle from L/R/C intercept point to make new L/R/C lines
     - angle 
 - throttle, angles , pose, line equation
 - Morph to parallel lines from top in pixels
 - Angle to append to prev line segent? 
    - based on steering angle?
 - car location L/R and angle to Lines
    
 - complete the loop?

 - Estimate # pixels between images at current speed

 - parameters: steering angle to degrees, speed to pixels
   series of intercepting lines.
     - tweak parameters until loop closed?
     - throttle: assuming linear ( T - C) * p 
         - Center line Dot -> moves frame to frame closer, determines speed
           - when Y moves up, get pixel X/Y
           - follow as Y moves closer, get pixel X/Y
             - average out over speed
             - ignore while line at bottom
             - Note, this is for ROI, not full frame!
         - 30 = 2-4 Pixels/Frame (depends on closeness, need transform
           when near center line! (VP)
         L/R more reliable when near CL due to width of CL.
 
         - VP moves to off center as turn approaches. This is the angle.

         - need better frame to frame tracking to detet bad L/C/R estimates
     - Steering to angle (estimate less than 2 degrees) -> max 16
         (not straight line)
     - FPS = DRIVE_LOOP_HZ = 20

 - Do line segments of current car location?
 - know current location/orientation? Know Throttle (pixels)
   keep track of overlapping line segments?
      - local coords known
   
 - transform L/R/C / orientation to top perspective
 - move up a few transformed pixels 
     -- join L/R/C line endpoints, add new segment
 - add new L/R/C / orientation


15-75-90 triangle
A = hypot   cos 75

  B
  15
a |\
  | \ c
  |__\
90 c  75
C      A
A = Camera
B = Pixel
a - full length not known (wheels off camera)

a = c cos A
b = c cos B
c is known in inches

a1 = c cos A 
a2 = c2 cos A

------------

x = v * cos(A)
y = v * sin(A)
A = v * tan(a)
a = angular steering angle
A = orientation wrt x axis
'''

import os, sys, time
import json
import numpy as np
from PIL import Image
import cv2
import math
import mergelines
import matplotlib.pyplot as plt
from stat import S_ISREG, ST_MTIME, ST_MODE, ST_CTIME, ST_ATIME
from pylab import array, plot, show, axis, arange, figure, uint8
from numpy import random
from donkeycar.parts.mergelines import HoughBundler, LaneLines
from donkeycar.parts.datastore import Tub


class Map(object):

    def __init__(self, model=None, *args, **kwargs):
        # super(RL, self).__init__(*args, **kwargs)

        global steering_hist, throttle_hist, lw_hist, ppf_hist, vpangle_hist
        global simplecl_hist, ll_hist, cl_hist, rl_hist
        global mapnum

        mapnum = 0
        steering_hist = []
        throttle_hist = []
        lw_hist = []
        ppf_hist = []
        vpangle_hist = []
        simplecl_hist = []
        ll_hist = []
        cl_hist = []
        rl_hist = []

    def image_path(self, tub_path, frame_id):
        return os.path.join(tub_path, str(frame_id) + "_cam-image_array_.jpg")

    def resize_map(self, up, down, left, right):
        global map

        pad_width = (up, down, left, right)
        cval= ((0,0)(0,0))
        map = np.pad(map, pad_width, 'constant', constant_values = cval)
        np.array(map).reshape(2000,2000)
        # offset to start position of car
        offsetX = offsetX + left  
        offsetY = offsetY + down


    def map(self,nm):
        global map
        global steering_hist, throttle_hist, lw_hist, ppf_hist, vpangle_hist
        global simplecl_hist, ll_hist, cl_hist, rl_hist
        global _mapnum

        # map show X (1036,2346) Y (375,1761)
        _INITMAPXSZ = 13500
        _INITMAPYSZ = 13000
        _XINIT = 4800
        _YINIT = 4700
        _WHITE = 255
        _WHITEBINARY = 1
        _BLACK = 0
        _BLACKBINARY = 0
        _STEERING_ANGLE = 15
        offsetX = _XINIT  # first point, changes after every resize
        offsetY = _YINIT
        npmap = np.zeros((_INITMAPXSZ,_INITMAPYSZ, 4), dtype=np.uint8)
        # track = plt.figure()
        # car = plt.subplot(_INITMAPXSZ, _INITMAPYSZ, _XINIT*_YINIT)
        # car.set_ylim(0, _INITMAPYSZ)
        # car.set_xlim(0, _INITMAPXSZ)
 
        # j is the angle scale factor to multiply with steering_angle
        for j in range(25,35):
         # for k in range(250,270):
          k = 266  # (26.6 ppf)
          # The following is interesting.
          # /tmp/map2/care32-265_cam-image_array_.jpg 
          # k = 26
          npmap = np.zeros((_INITMAPXSZ,_INITMAPYSZ, 4), dtype=np.uint8)
          minX = _INITMAPXSZ
          minY = _INITMAPXSZ
          maxX = 0
          maxY = 0
          minPPF = 1000
          maxPPF = 0
          minT = 1000
          maxT = 0
          X = offsetX
          Y = offsetY
          angle = math.radians(90)
          for i in range(len(steering_hist)):
            if ppf_hist[i] < 0:
              continue
            # find the end point for car
            # endY = Y + ppf_hist[i] * math.sin(math.radians(angle))
            # endX = X + ppf_hist[i] * math.cos(math.radians(angle))
            ppf_hist[i] = k/10
            endY = Y + ppf_hist[i] * math.sin(angle)
            endX = X + ppf_hist[i] * math.cos(angle)
            angle = angle + ppf_hist[i] * math.tan(math.radians(90 + steering_hist[i] * j))

            # print("ppf %d angle %d [%d,%d][%d,%d]"%(ppf_hist[i], angle, X,Y,endX,endY))
            # car.plot([X, Y],[endX, endY])
            # in 129, followed white from clips 379 to 953
            if 379 <= i <= 953:
              cv2.line(npmap,(int(X), int(Y)),(int(endX), int(endY)),(0,_WHITE,0),5)
            Y = endY
            X = endX
            minX = min(minX, X)
            maxX = max(maxX, X)
            minY = min(minY, Y)
            maxY = max(maxY, Y)
            minPPF = min(minPPF, ppf_hist[i])
            maxPPF = max(maxPPF, ppf_hist[i])
            minT = min(minT, throttle_hist[i])
            maxT = max(maxT, throttle_hist[i])

            '''
            if i % 10 == 0:
              cv2.imshow('car',npmap)
              cv2.waitKey(0)
              cv2.destroyAllWindows()
            '''
          print("map %d show X (%d,%d) Y (%d,%d) ppf (%d, %d) T (%f, %f)"%(j, minX, maxX, minY, maxY, minPPF, maxPPF, minT, maxT))
          # plt.show()
          # img = Image.fromarray(npmap, 'RGB')
          str = "car%s%d-%d" % (nm, j, k)
          '''
          cv2.imshow('car',npmap)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          '''
          ll = LaneLines()
          out = ll.image_path("/tmp/map2", str)
          cv2.imwrite(out, npmap)


    def update_hist(self, steering, throttle, lw, ppf, vpangle, simplecl, ll, cl, rl):
        global steering_hist, throttle_hist, lw_hist, ppf_hist, vpangle_hist
        global simplecl_hist, ll_hist, cl_hist, rl_hist

        steering_hist.append(steering)
        throttle_hist.append(throttle)
        lw_hist.append(lw_hist)
        ppf_hist.append(ppf)
        vpangle_hist.append(vpangle)
        simplecl_hist.append(simplecl)
        ll_hist.append(ll)
        cl_hist.append(cl)
        rl_hist.append(rl)

    def process_tub(self, img, steering, throttle):
        global tmpsteering, tmpthrottle, steering_hist, throttle_hist, speed, angle

        ll = LaneLines()
        simplecl, lines, roi = ll.process_img(img)
        roi = ll.getROI(img)
        if lines is not None:
          tmpsteering, tmpthrottle = ll.lrclines(lines,roi)

        lw, ppf, vpangle, ll, cl, rl = ll.get_map_data()
        self.update_hist(steering, throttle, lw, ppf, vpangle, simplecl, ll, cl, rl)

    def test_tub(self, tub_path):
        inputs = ['pilot/angle', 'pilot/throttle', 'cam/image']
        types = ['float', 'float', 'image']
        ll = LaneLines()
        self.tub = Tub(path=tub_path, inputs=inputs, types=types)

        seqs = [ int(f.split("_")[0]) for f in os.listdir(tub_path) if f.endswith('.jpg') ]
        seqs.sort()
        entries = ((os.stat(ll.image_path(tub_path, seq))[ST_ATIME], seq) for seq in seqs)

        (last_ts, seq) = next(entries)
        clips = [[seq]]
        for next_ts, next_seq in entries:
            if next_ts - last_ts > 100:  #greater than 1s apart
                clips.append([next_seq])
            else:
                clips[-1].append(next_seq)
            last_ts = next_ts

        for clip in clips:
          for imgseq in clip:
            # if imgseq < 122:
            #   continue
            # imgname = ll.image_path(tub_path, rec['cam/image_array'])
            # img = cv2.imread(imgname)
      
            rec = self.tub.get_record(imgseq)
            img = rec['cam/image_array']
            # print(rec)
            # self.process_tub(imgname, rec['pilot/angle'], rec['pilot/throttle'])
            self.process_tub(img, rec['pilot/angle'], rec['pilot/throttle'])
            # self.map()


mp = Map()
'''
tub_path = '/home/ros/d2/data/tub_126_18-07-15'
mp.test_tub(tub_path)
mp.map("a")
tub_path = '/home/ros/d2/data/tub_127_18-07-15'
mp.test_tub(tub_path)
mp.map("b")
'''
tub_path = '/home/ros/d2/data/tub_129_18-07-15'
mp.test_tub(tub_path)
mp.map("e")
'''
tub_path = '/home/ros/d2/data/tub_140_18-07-15'
mp.test_tub(tub_path)
mp.map("d")
'''
