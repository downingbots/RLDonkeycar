# https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
import numpy as np
import cv2
import math
import os
import sys
import copy
import donkeycar as dk
from stat import S_ISREG, ST_MTIME, ST_MODE, ST_CTIME, ST_ATIME
from donkeycar.parts.datastore import Tub
import json

# todo:
# smooth turns
# handle sharp angles
# 3rd line?
# add minimal throttle control

#############################

class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''
    def __init__(self, width, height, bestll, bestcl, bestrl):
      self.width = width
      self.height = height
      self.bestll = bestll
      self.bestcl = bestcl
      self.bestrl = bestrl

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(((line[3]) - (line[1])), (line[2] - (line[0])))
        deg = math.degrees(orientation)
        #  deg:    0  45  90 135 180 -135 -90 -45
        #  ret:   45  90 135 180  45   90 135

        if deg < 0:
          deg += 180
        elif deg < 180:
          pass
        elif deg > 180:
          deg = deg - 180
#     # print("DEG %d" % deg)
        return deg

    def checker(self, line_new, line_old):
        '''Check if line have enough distance and angle to be count as similar
        '''
        # for debugging
        # mindist = 400
        # minangle = 400
        # Parameters to play with
        # min_distance_to_merge = 40   # near
        # min_distance_to_merge2 = 5  # far 
        # max_y = 70
        # max_y = max(line_old[0][1], line_old[1][1], line_new[0][1], line_new[1][1])
        # max_y = 70 - max(line_old[1], line_old[3], line_new[1], line_new[3])
        max_y = max(line_old[1], line_old[3], line_new[1], line_new[3])
        # max_y = max(line_old[0][1], line_old[1][1], line_new[0][1], line_new[1][1])
        min_distance_to_merge = int(max_y / 2) + 5
        # min_distance_to_merge = int(max_y) + 5
        # print("max_y %d dist %d" % (max_y,min_distance_to_merge))
     
        min_angle_to_merge = 20
        # min_angle_to_merge = 30
        # min_angle_to_merge = 40
        if self.get_distance(line_old, line_new) < min_distance_to_merge:
          # check the angle between lines
          orientation_new = self.get_orientation(line_new)
          orientation_old = self.get_orientation(line_old)
          angle = abs(orientation_new - orientation_old)
          # if self.get_distance(line_old, line_new) < mindist:
          #   mindist = self.get_distance(line_old, line_new) 
          #   if minangle > angle:
          #     minangle = angle
          # if all is ok -- line is similar to others in group
          if abs(orientation_new - orientation_old) < min_angle_to_merge:
            # print("angle %d %d %d dist %d" % (orientation_new, orientation_old, min_angle_to_merge, self.get_distance(line_old, line_new)))
            return True
          # print("angle %d %d %d dist %d - NO MERGE" % (orientation_new, orientation_old, min_angle_to_merge, self.get_distance(line_old, line_new)))
        return False

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        merge_groups = []

        # print("line len %d" % len(lines))
        for line_new in lines[1:]:
          groupnum = -1
          mergegroup = []
          for group in groups:
            groupnum += 1
            for line_old in group:
              # print("----")
              # print(line_new)
              # print(line_old)
              if self.checker(line_new, line_old):
                mergegroup.append(groupnum)
                break
          mergegrouplen = len(mergegroup)
          # print("mg len %d" % mergegrouplen)
          if mergegrouplen == 0 or len(group) == 0:
            # add group
            groups.append([line_new])
          else:
            # merge all groups that line is in into mergegroup[0]
            for i in range(mergegrouplen-2):
              groups[mergegroup[0]].extend(groups[mergegroup[mergegrouplen-i-1]])
              del(groups[mergegroup[mergegrouplen-i-1]])
            # print("merged line into %d groups" % mergegrouplen)
            # add line to merged group
            groups[mergegroup[0]].append(line_new)
          # print("groups_len %d" % len(groups))
        # print("groups len %d" % len(groups))
        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        # if 45 < orientation < 135:
        # if 15 < orientation < 165:
        if 15 < orientation < 165:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        ll = False
        cl = False
        rl = False
        # print(self.bestll)
        # print(lines)
        # print("------")
        if lines is not None and self.bestrl is not None and self.bestcl is not None and self.bestll is not None:
            for line_i in [l[0] for l in lines]:
              # print(line_i)
              linell = (self.bestll[0][0],self.bestll[0][1], self.bestll[1][0], self.bestll[1][1])
              linecl = (self.bestcl[0][0],self.bestcl[0][1], self.bestcl[1][0], self.bestcl[1][1])
              linerl = (self.bestrl[0][0],self.bestrl[0][1], self.bestrl[1][0], self.bestrl[1][1])
              if self.checker(line_i, linell):
                ll = True
                lines_y.append(line_i)
              elif self.checker(line_i, linerl):
                rl = True
                lines_y.append(line_i)
              elif self.checker(line_i, linecl):
                cl = True
                lines_y.append(line_i)
              '''
              if rl and cl and ll:
                print("checker ll cl rl")
              elif cl and ll:
                print("checker ll cl   ")
              elif rl and ll:
                print("checker ll    rl")
              elif rl and cl:
                print("checker    cl rl")
              else:
                print("checker         ")
              '''
        if lines is not None and not (rl and cl and ll):
            lines_y = []
            for line_i in [l[0] for l in lines]:
                orientation = self.get_orientation(line_i)
                # print("post checker: orientation %d"% orientation)
                # if vertical
                if 15 < orientation < 165:
                    lines_y.append(line_i)
                else:
                    lines_x.append(line_i)
        elif lines is None: 
          return None

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        # for i in [lines_x, lines_y]:
        for i in [lines_y]:
                if len(i) > 0:
                    groups = self.merge_lines_pipeline_2(i)
                    merged_lines = []
                    # print(groups)
                    for group in groups:
                        merged_lines.append(self.merge_lines_segments1(group))

                    merged_lines_all.extend(merged_lines)
                    # print(merged_lines)

      # print("num merged lines %d " % len(merged_lines_all))
        return merged_lines_all

    def line_follower(self, lines):
        midpix = self.width / 2
        mindist = 10000
        minangle = 90
        cl = None
        if lines is not None:
          for line_i in [l[0] for l in lines]:
            angle = self.get_orientation(line_i)
            dist = math.hypot(line_i[0] - midpix, line_i[1] - self.height)
            dist2 = math.hypot(line_i[2] - midpix, line_i[3] - self.height)
            llen = math.hypot(line_i[0] - line_i[2], line_i[1] - line_i[3])
            # dist = math.hypot(line_i[0][0] - midpix, line_i[0][1] - self.height)
            # dist2 = math.hypot(line_i[1][0] - midpix, line_i[1][1] - self.height)
            # llen = math.hypot(line_i[1][0] - line_i[0][1], line_i[1][1] - line_i[0][1])
            mind = min(dist, dist2)

            ANGLE_THRESH = 60
            DIST_THRESH = 100
            # DIST_THRESH = 30
            # print("mind %d mindist %d angle %d minangle %d llen %d" % (mind, mindist, angle, minangle, llen))
            if ((mind < mindist or abs(angle-90) < minangle) and 
                llen > 10 and
                mind < DIST_THRESH and abs(angle-90) < ANGLE_THRESH):
              if ((mind < mindist and abs(angle-90) < minangle) or
               (abs(mindist - mind) > 2*abs((abs(angle-90) - minangle))) or
               (2*abs(mindist - mind) < abs((abs(angle-90) - minangle)))):
                mindist = mind
                minangle = abs(angle-90)
                cl = copy.deepcopy(line_i)
                cl = [[cl[0],cl[1]],[cl[2],cl[3]]]
        # else:
          # print("simplecl lines = None")
        return cl

class ThrottleBase:
  def __init__(self):
    global cfg

    cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))
    self.emergency_stop = False
    self.minthrottle = 0
    self.ch_th_seq = 0
    self.resetThrottleInfo()
    self.frame = None

  def optflow(self, old_frame, new_frame):
    # cap = cv.VideoCapture('slow.flv')
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    # ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    try:
      p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    except:
      print("OPT FLOW FAILS")
      return False
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    dist = 0
    numpts = 0
    frame1 = new_frame
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        dist += math.hypot(a-c,b-d)
        numpts += 1
        # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        # frame1 = cv2.circle(frame1,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(new_frame,mask)
    # cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    # Now update the previous frame and previous points
    # old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    # cv2.destroyAllWindows()
    if numpts != 0:
      dist /= numpts
    else:
      dist = 0
    print("optflow dist %f minthrot %d maxthrot %d" % (dist,self.minthrottle,self.maxthrottle))
    # note: PPF also used to ensure that moving
    # tried 0.75, 0.9, 1
    # OPTFLOWTHRESH = 0.8
    if dist > cfg.OPTFLOWTHRESH:
      return True
    else:
      return False

  def setEmergencyStop(self, TF):
      self.emergency_stop = TF
      # turn off Emergency Stop for now
      if cfg.DISABLE_EMERGENCY_STOP:
        self.emergency_stop = False
      if self.emergency_stop:
        self.resetThrottleInfo()

  def emergencyStop(self):
      return self.emergency_stop

  def setMinMaxThrottle(self, new_frame):
    if self.frame is None:
      self.frame = new_frame
      print("frame is None")
      return False
    print("minthrottle %d" % self.minthrottle)
    if self.optflow(self.frame, new_frame):
      # car is moving, set maxthrottle if we need to
      if self.maxthrottle < 0:
        # if previouly not moving.
        # self.minthrottle += 2
        self.maxthrottle = self.minthrottle + 10
      self.check_throttle = False
      # self.frame = new_frame
      self.frame = None
      return True
    # we're not moving. Reset maxthrottle. Reset throttle via optflow.
    self.frame = new_frame
    self.minthrottle += cfg.OPTFLOWINCR
    self.maxthrottle = -1
    return False

  def getThrottleInfo(self):
    return self.minthrottle, self.maxthrottle, self.battery_adjustment

  def resetThrottleInfo(self):
    global cfg

    # take about a second to find new minthrottle
    self.minthrottle = max(cfg.MINMINTHROT, (self.minthrottle-1))
    self.maxthrottle = -1
    self.battery_adjustment = 0
    self.battery_count = 0
    self.ch_th_seq += 1
    self.check_throttle = True

  def throttleCheckInProgress(self):
    if self.maxthrottle < 0 or self.check_throttle:
      return True
    return False

  def adjustForBattery(self,pixPerFrame = -1):
    # DESIREDPPF = 35
    # MAXBATADJ  = .10
    # MAXBATCNT  = 1000
    self.ch_th_seq = 0  # postpone opt flow throttle check
    if pixPerFrame > 0 and pixPerFrame < cfg.DESIREDPPF and self.battery_adjustment < cfg.MAXBATADJ and self.maxthrottle > 0:
      self.battery_adjustment += cfg.BATADJ
      print("BATTERY ADJUSTMENT %f" % self.battery_adjustment)
      self.battery_count = 0
    else:
      self.battery_count += 1
      if self.battery_count > cfg.MAXBATCNT:
        print("BATTERY ADJUSTMENT %f" % self.battery_adjustment)
        self.battery_count = 0
        self.battery_adjustment += .01

  def checkThrottle(self,expected_throttle):
      self.ch_th_seq += 1
      # CHECK_THROTTLE_THRESH = 20
      # if self.ch_th_seq % cfg.CHECK_THROTTLE_THRESH == 0 and expected_throttle >= self.minthrottle / 100:
      if self.ch_th_seq % cfg.CHECK_THROTTLE_THRESH == 0:
        self.check_throttle = True
        print("CHECK THROTTLE")
        return False
      else:
        # print("seq %d th %f minth %f" % (ch_th_seq, expected_throttle, self.minthrottle))
        return True

class LaneLines:

  #############################
  # derived from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

  def ccw(self, A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
  # return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

  # Return true if line segments AB and CD intersect
  def intersect(self, line1, line2):
    A = line1[0]
    B = line1[1]
    C = line2[0]
    D = line2[1]
    return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

  ###############
  # where infinite lines would intersect
  def intersection (self, line1, line2):
    D = (line1[0][0] - line1[1][0]) * (line2[0][1] - line2[1][1]) - (line1[0][1] - line1[1][1]) * (line2[0][0] - line2[1][0])
    A = (line1[0][0] *line1[1][1]) - (line1[0][1] *line1[1][0])
    B = (line2[0][0] *line2[1][1]) - (line2[0][1] *line2[1][0])
    # print(line1)
    # print(line2)
    if D != 0:
      x = int (A * (line2[0][0] - line2[1][0]) - B*(line1[0][0] - line1[1][0]) ) / D
      y = int (A * (line2[0][1] - line2[1][1]) - B*(line1[0][1] - line1[1][1]) ) / D
      # print("intersect %d %d" % (x,y))
      return x, y
    else:
      # print("no intersect")
      return -1000, -1000

  #############################
  # from https://github.com/dmytronasyrov/CarND-LaneLines-P1/blob/master/P1.ipynb
  def getROI(self,img):
    # w 160 h 120
    # remove top 40 - 50 lines
    # croptop = 45
    croptop = 44
    
    self.width = img.shape[1]
    self.height = img.shape[0]
    self.roiheight = self.height - croptop
    # print("w %d h %d" % (self.width, self.height))
    roi_img = img[croptop:self.height, 0:self.width]
    return roi_img

  # find potential vanishing point and lane width
  def vanishing_point2(self, line1, line2):

    midpix = int(self.width / 2)
    x,y = self.intersection(line1,line2)
    donkeyvpl = ((midpix, self.roiheight),(x,y))
    self.donkeyvpangle = self.orientation(donkeyvpl)

    # find width
    if self.donkeyvpangle < 90:
      slope = self.donkeyvpangle + 90
    else:
      slope = self.donkeyvpangle - 90
    slope = math.tan(math.radians(slope))

    # if donkeyvpangle != 90:
    dkvpy =  midpix - self.roiheight * slope
    donkeyhorizline =  ((midpix, self.roiheight),(0, dkvpy))
    # else:
      # donkeyhorizline =  ((line3[0][0], self.roiheight),(0, self.roiheight))
    l1x, l1y = self.intersection(donkeyhorizline,line1)
    l2x, l2y = self.intersection(donkeyhorizline,line2)

    lanewidth1 = math.hypot(l1x - l2x, l1y - l2y)
  # print("vp2: x %d y %d lane width %f LW %f" % (x,y,lanewidth1, laneWidth))
    return x,y, self.donkeyvpangle, lanewidth1


  # if 3 lines have same vanishing point, then we are confident of result
  # we can also compute width of track and pixels per frame speed
  def vanishing_point(self, line1, line2, line3):
    maxdiffallowed = 10
    x12,y12 = self.intersection(line1,line2)
    x13,y13 = self.intersection(line1,line3)
    x23,y23 = self.intersection(line2,line3)
    if x12 == -1000 or y12 == -1000 or x13 == -1000 or y13 == -1000 or x23 == -1000 or y23 == -1000:
      print("VP False ")
      return False, -1000, -1000
    xmaxdif = max(abs(x12-x13), abs(x12-x23), abs(x13-x23) )
    ymaxdif = max(abs(y12-y13), abs(y12-y23), abs(y13-y23)) 
    ###############
    # find lane width, and transformed parallel L/C/R lines
    ###############
    midpix = int(self.width / 2)
    vpx = int((x12+x13+x23)/3)
    vpy = int((y12+y13+y23)/3)
    donkeyvpl = ((midpix, self.roiheight),(vpx,vpy))
    self.donkeyvpangle = self.orientation(donkeyvpl)
    if (self.VPx is None):
      self.VPx = vpx
      self.VPy = vpy
    else:
      self.VPx = .8*self.VPx + .2*vpx
      self.VPy = .8*self.VPy + .2*vpy
    if self.donkeyvpangle < 90:
      slope = self.donkeyvpangle + 90
    else:
      slope = self.donkeyvpangle - 90
    slope = math.tan(math.radians(slope))

    ###############
    # find transfomed pixels per frame 
    ###############

    if self.PPFx is None:
      self.maxppfy = 0
      if line2[1][1] >= self.roiheight - 1:
      # print("point1y > point0y: PASS")
        pass
      elif (line2[0][1] < line2[1][1]):
        # most likely TRUE
      # print("point1y > point0y")
        self.PPFx = line2[1][0]
        self.PPFy = line2[1][1]
      else:
        # most likely FALSE
        # should add identical logic here too
        self.PPFx = line2[0][0]
        self.PPFy = line2[0][1]
      # print("point0y %d > point1y %d !!!" % (self.PPFy, line2[1][1]))
    else:
      if (line2[0][1] < line2[1][1]):
        if line2[1][1] < self.roiheight - 1:
          # note 15, 75, 90 right triangle
          ppfx = line2[1][0] - self.PPFx
          ppfy = line2[1][1] - self.PPFy
          if ppfy > 0:
            if ppfy > self.maxppfy:
              self.maxppfy = ppfy
            # self.pixPerFrame = math.hypot(ppfx, ppfy)
          # print("point1y %d > point0y: PPF %f" %(line2[1][1], self.maxppfy))
        else:
          # the closer we get to the next dash, the higher the num pix.
          # use weighted average of max to estimate birds-eye view num pix
          if self.pixPerFrame < 0:
            self.pixPerFrame = self.maxppfy
          elif self.maxppfy > 0:
            self.pixPerFrame = .8*self.pixPerFrame + .2*self.maxppfy
          # print("reset PPF: PPF = %d" % self.pixPerFrame)

          # adjust for battery depletion
          self.TB.adjustForBattery(self.pixPerFrame)

          #reset
          self.PPFx = None
          self.PPFy = None
          self.maxppfy = 0
        self.PPFx = line2[1][0]
        self.PPFy = line2[1][1]
      else:
        # Most likely False, but not always
        # should add identical logic here too
        self.PPFx = line2[0][0]
        self.PPFy = line2[0][1]
      # print("point0y %d > point1y %d !!!" % (self.PPFy, line2[1][1]))

    # get y intersect
    # if donkeyvpangle != 90:
    y =  midpix - self.roiheight * slope
    donkeyhorizline =  ((midpix, self.roiheight),(0, y))

    l1x, l1y = self.intersection(donkeyhorizline,line1)
    l2x, l2y = self.intersection(donkeyhorizline,line2)
    l3x, l3y = self.intersection(donkeyhorizline,line3)
    # donkeyvpangle
    lanewidth1 = math.hypot(l1x - l2x, l1y - l2y)
    lanewidth2 = math.hypot(l3x - l2x, l3y - l2y)
    lanewidth3 = math.hypot(l3x - l1x, l3y - l1y)
    self.dklinept = ((l1x,l1y),(l2x,l2y),(l3x,l2y),(midpix,self.roiheight),(vpx,vpy))
    if abs(lanewidth1 + lanewidth2 - lanewidth3) < 5: 
      # print("lw : %f, %f = %f + %f, %f angle %f" % (lanewidth3, lanewidth2 + lanewidth1, lanewidth2, lanewidth1, lanewidth3/2, self.donkeyvpangle))
      lw = lanewidth3
    elif abs(lanewidth1 - lanewidth2 - lanewidth3) < 5: 
      # print("lw : %f, %f = %f + %f, %f angle %f" % (lanewidth1, lanewidth2 + lanewidth3, lanewidth2, lanewidth3, lanewidth1/2, self.donkeyvpangle))
      lw = lanewidth1
    elif abs(lanewidth2 - lanewidth1 - lanewidth3) < 5: 
      # print("lw : %f, %f = %f + %f, %f angle %f" % (lanewidth2, lanewidth1 + lanewidth3, lanewidth1, lanewidth3, lanewidth2/2, self.donkeyvpangle))
      lw = lanewidth2
    else:
      lw = -1
    
    if xmaxdif < maxdiffallowed and ymaxdif < maxdiffallowed:
      MAXLANEWIDTH = 400  # should be much smaller
      if lw >=0 and lw < MAXLANEWIDTH and 60 < self.donkeyvpangle < 120:
        if self.laneWidth < 0:
          self.laneWidth = lw/2
        else:
          self.laneWidth = .8 * self.laneWidth + .2*lw/2
    # print("lane width %f" % self.laneWidth)
    
      print("VP True %d %d" % ( int((x12+x13+x23)/3), int((y12+y13+y23)/3)))
      # compute color
      return True, vpx, vpy
    else:
      print("VP False %d %d" % (xmaxdif, ymaxdif))
      return False, xmaxdif, ymaxdif

  def is_vanishing_point(self):
    VP = False
    if self.bestll is not None and self.bestcl is not None and self.bestrl is not None:
      VP, x, y = self.vanishing_point(self.bestll, self.bestcl, self.bestrl)
    return VP

  def is_vanishing_point2(self):
    # print("laneWidth %d" % self.laneWidth)
    if self.laneWidth < 0:
      return True
    if self.bestcl is not None and self.bestrl is not None:
      x,y, dkvp2angle, lanewidth1 = self.vanishing_point2(self.bestcl,self.bestrl)
      # print("vp2a: x %d y %d lane width %f LW %f" % (x,y,lanewidth1, self.laneWidth))
      return True
    if self.bestcl is not None and self.bestll is not None:
      x,y, dkvp2angle, lanewidth1 = self.vanishing_point2(self.bestll,self.bestcl)
      # print("vp2b: x %d y %d lane width %f LW %f" % (x,y,lanewidth1, self.laneWidth))
      return True
    return False

  def linelen(self,line):
     if line is None:
       return -1
     return math.hypot(line[0][0] - line[1][0], line[0][1] - line[1][0])

  def orientation(self, line):
     hb = HoughBundler(self.width, self.height, self.bestll, self.bestcl, self.bestrl)
     line_a = (line[0][0], line[0][1], line[1][0], line[1][1])
     return hb.get_orientation(line_a)

  def check(self, line1, line2):
        hb = HoughBundler(self.width, self.height, self.bestll, self.bestcl, self.bestrl)
        lineA = (line1[0][0],line1[0][1], line1[1][0], line1[1][1])
        lineB = (line2[0][0],line2[0][1], line2[1][0], line2[1][1])
        return hb.checker(lineA, lineB)

  def closestX(self, line):
      if (line[0][1] < line[0][1]):
        x = line[0][0]
      else:
        x = line[1][0]
      return x

  def steerByLine(self, line):
      angle = self.orientation(line)
      denom = 40
      midpix = int(self.width / 2)
      if angle < 60:
        steering = 1
      elif angle > 120:
        steering = -1
      elif angle <= 90:
        steering = min(max( ((self.closestX(line) + self.laneWidth)/2 - midpix) / denom, -1),1)
      elif angle > 90:
        steering = min(max( ((self.closestX(line) - self.laneWidth)/2 - midpix) / denom, -1),1)
      return steering


  def setSteerThrottle(self, pos, ll, cl, rl, conf):
    steering = 0.0
    minthrottle, maxthrottle, battery_adjustment = self.TB.getThrottleInfo()
    if self.TB.throttleCheckInProgress():
      # still initiallizing min/max throttle
      throttle = minthrottle
    else:
      throttle = minthrottle + conf
      if throttle > maxthrottle:
        throttle = maxthrottle
      # print("throt %f conf %f minth %f maxthrottle %d" % (throttle, conf, minthrottle, maxthrottle))
    # for self,sim
    throttle /= 100
    midpix = int(self.width / 2)

    str = "lw %d " % self.laneWidth
    if ll is not None:
      str += "; l %d - %d " % (ll[0][0], ll[1][0])
    if cl is not None:
      str += "; c %d - %d " % (cl[0][0], cl[1][0])
    if rl is not None:
      str += "; r %d - %d " % (rl[0][0], rl[1][0])
    # rl and ll can be same!
    # cl and rl can be reversed!
    # lw varies too much
    # print(str)
    if cl is not None:
      angle = self.orientation(cl)
      denom = 40
      steering = min(max((self.closestX(cl) - midpix) / denom,-1),1)
      llen = self.linelen(cl)
      if angle < 60 and llen > 20:
        steering = 1
      elif angle > 120 and llen > 20:
        steering = -1
      # steering = self.steerByLine(cl, 0)
    elif ll is not None and rl is not None:
      llsteering = self.steerByLine(ll)
      rlsteering = self.steerByLine(rl)
      steering = (llsteering + rlsteering) / 2
      # steering = ( ((self.closestX(ll) + self.closestX(rl))/2 - midpix) / 80)
    elif ll is not None :
      steering = self.steerByLine(ll)
    elif rl is not None :
      steering = self.steerByLine(rl)
    '''
    Mapping: 
      do projection for the Lines only.
      Factor in speed, steering
      Get total speed/steering

    Remove pictures/movies
    '''
    # if on a turn, need to add throttle
    if abs(steering ) > .9:
      throttle += cfg.TURNADJ
      # print("TurnAdj %f BatAdj %f" % (cfg.TURNADJ, battery_adjustment))
    elif abs(steering ) > .8:
      throttle += cfg.TURNADJ / 2
      # print("TurnAdj %f BatAdj %f" % (cfg.TURNADJ/2, battery_adjustment))
    # else:
      # print("TurnAdj 0.00 BatAdj %f" % battery_adjustment)
    throttle += battery_adjustment

    if pos == 0 or pos == 6:
      throttle = 0.0

    return conf, steering, throttle 

  def setCurPos(self, bestll, bestcl, bestrl, pos):
    # ARD TODO: not right, depends on angle of lines

    midpix = self.width / 2
    curpos = -1
    #  -|---|---|
    #  0|123|345|6
    if bestcl is not None and bestrl is not None and bestll is not None:
      # difx = (bestcl[0][0] - bestll[0][0]) / 3
      # difx2 = (bestrl[0][0] - bestcl[0][0]) / 3
      if self.laneWidth > 0:
        difx = self.laneWidth / 3
        difx2 = self.laneWidth / 3
      else:
        difx = (bestcl[0][0] - bestll[0][0]) / 3
        difx2 = (bestrl[0][0] - bestcl[0][0]) / 3
      if midpix <= (bestll[0][0]):
        curpos = 0
        print("0a")
        # print("bestll")
        # print(bestll)
        # print("bestcl")
        # print(bestcl)
        # print("bestrl")
        # print(bestrl)
      elif midpix <= (bestll[0][0] + difx) :
        curpos = 1
      elif midpix <= (bestll[0][0] + 2*difx) :
        curpos = 2
      elif midpix <= (bestcl[0][0] + difx2) :
        curpos = 3
      elif midpix <= (bestcl[0][0] + 2*difx2) :
        curpos = 4
      elif midpix <= (bestrl[0][0]) :
        curpos = 5
      else:
        print("6a")
        curpos = 6
        # print("bestll")
        # print(bestll)
        # print("bestcl")
        # print(bestcl)
        # print("bestrl")
        # print(bestrl)

    elif bestll is not None and bestcl is not None and bestrl is None:
      # difx = (bestcl[0][0] - bestll[0][0]) / 3
      if self.laneWidth > 0:
        difx = self.laneWidth / 3
      else:
        difx = (bestcl[0][0] - bestll[0][0]) / 3
      if midpix <= (bestll[0][0]):
        curpos = 1
        print("1b")
        # print("bestll")
        # print(bestll)
        # print("bestcl")
        # print(bestcl)
      elif midpix <= (bestll[0][0] + difx) :
        curpos = 1
      elif midpix <= (bestll[0][0] + 2*difx) :
        curpos = 2
      elif midpix < (bestcl[0][0] + difx ) :
        curpos = 3
      elif midpix <= (bestcl[0][0] + 2*difx) :
        curpos = 4
      elif midpix <= (bestcl[0][0] + 3*difx) :
        curpos = 5
      else:
        print("6b")
        curpos = 6
        # print("bestll")
        # print(bestll)
        # print("bestcl")
        # print(bestcl)

    elif bestll is None and bestcl is not None and bestrl is not None:
      if self.laneWidth > 0:
        difx = self.laneWidth / 3
      else:
        difx = (bestrl[0][0] - bestcl[0][0]) / 3
      # difx = (bestrl[0][0] - bestcl[0][0]) / 3
      difx = self.laneWidth / 3
      if midpix < (bestcl[0][0] - 3*difx) :
        curpos = 0
        print("0c %d %d " % (bestrl[0][0], bestcl[0][0]))
        # print("bestcl")
        # print(bestcl)
        # print("bestrl")
        # print(bestrl)
      elif midpix <= (bestcl[0][0] - 2*difx) :
        curpos = 1
      elif midpix <= (bestcl[0][0] - difx) :
        curpos = 3
      elif midpix <= (bestcl[0][0] + 2*difx) :
        curpos = 4
      elif midpix <= (bestrl[0][0] + 6*difx) :
        curpos = 5
      else:
        curpos = 6
        print("6c")
        # print("bestcl")
        # print(bestcl)
        # print("bestrl")
        # print(bestrl)

    elif bestll is not None and bestcl is not None and bestrl is None:
      print("setCurPos: huh?")
      if self.laneWidth > 0:
        difx = self.laneWidth / 3
      else:
        difx = (bestrl[0][0] - bestcl[0][0]) / 3
      if midpix < (bestll[0][0] ) :
        print("0d")
        # print("bestll")
        # print(bestll)
        # print("bestcl")
        # print(bestcl)
        curpos = 0
      elif midpix <= (bestll[0][0] + difx) :
        curpos = 1
      elif midpix <= (bestll[0][0] + 2*difx) :
        curpos = 2
      elif midpix <= (bestll[0][0] + 4*difx) :
        curpos = 3
      elif midpix <= (bestcl[0][0] + 2*difx) :
        curpos = 4
      elif midpix <= (bestrl[0][0] + 6*difx) :
        curpos = 5
      else:
        print("6d")
        # print("bestll")
        # print(bestll)
        # print("bestcl")
        # print(bestcl)
        curpos = 6

    elif bestll is not None and bestcl is None and bestrl is not None:
      if self.laneWidth > 0:
        difx = self.laneWidth / 3
      else:
        difx = (bestrl[0][0] - bestll[0][0]) / 6
      if midpix < (bestll[0][0]):
        curpos = 0
        print("0e")
        # print("bestll")
        # print(bestll)
        # print("bestrl")
        # print(bestrl)
      elif midpix <= (bestll[0][0] + difx) :
        curpos = 1
      elif midpix <= (bestll[0][0] + 2*difx) :
        curpos = 2
      elif midpix <= (bestrl[0][0] - 2*difx) :
        curpos = 3
      elif midpix <= (bestrl[0][0] - difx) :
        curpos = 4
      elif midpix <= (bestrl[0][0] ):
        curpos = 5
      else:
        curpos = 6
        print("6e")
        # print("bestll")
        # print(bestll)
        # print("bestrl")
        # print(bestrl)

    elif bestll is not None and bestcl is None and bestrl is None:
      difx = self.laneWidth / 3
      angle = self.orientation(bestll)
      print("angle %d" % angle)
      if angle < 60:
        curpos = 1
        print("1f angle %d" % angle)
      elif midpix > (bestll[0][0] ) and midpix < (bestll[1][0] ) :
        # line nearly horiz
        curpos = 1
        angle = self.orientation(bestll)
        print("1f angle %d" % angle)
      elif midpix < (bestll[0][0] ) :
        curpos = 0
        print("0f")
        # print("bestll")
        # print(bestll)
      elif midpix <= (bestll[0][0] + difx) :
        curpos = 1
      elif midpix <= (bestll[0][0] + 2*difx) :
        curpos = 2
      elif midpix <= (bestll[0][0] + 4*difx) :
        curpos = 3
      elif midpix <= (bestll[0][0] + 5*difx) :
        curpos = 4
      elif midpix >= (bestll[0][0] + 6*difx) :
        curpos = 5
      else:
        curpos = 6
        angle = self.orientation(bestll)
        print("6f angle %d" % angle)
        # print("bestll")
        # print(bestll)

    elif bestll is None and bestcl is not None and bestrl is None:
      difx = self.laneWidth / 3
      if midpix < (bestcl[0][0] - 3*difx) :
        curpos = 0
        print("0g")
        # print("bestcl")
        # print(bestcl)
      elif midpix <= (bestcl[0][0] - 2*difx) :
        curpos = 1
      elif midpix <= (bestcl[0][0] - 1*difx) :
        curpos = 2
      elif midpix <= (bestcl[0][0] + 1*difx) :
        curpos = 3
      elif midpix <= (bestcl[0][0] + 2*difx) :
        curpos = 4
      elif midpix <= (bestcl[0][0] + 3*difx) :
        curpos = 5
      else:
        curpos = 6
        print("6g")
        # print("bestcl")
        # print(bestcl)

    elif bestll is None and bestcl is None and bestrl is not None:
      difx = self.laneWidth / 3
      angle = self.orientation(bestrl)
      print("angle %d" % angle )
      if angle > 120:
        # right lane nearly horiz
        curpos = 1
        print("5h")
      elif midpix <= (bestrl[0][0] - 6*difx) :
        curpos = 0
        print("0h")
        # print("bestrl")
        # print(bestrl)
      elif midpix <= (bestrl[0][0] - 5*difx) :
        curpos = 1
      elif midpix <= (bestrl[0][0] - 4*difx) :
        curpos = 2
      elif midpix <= (bestrl[0][0] - 2*difx) :
        curpos = 3
      elif midpix <= (bestrl[0][0] - difx) :
        curpos = 4
      elif midpix < (bestrl[0][0] ) :
        curpos = 5
      else:
        curpos = 6
        print("6h")
        # print("bestrl")
        # print(bestrl)
    elif bestll is None and bestcl is None and bestrl is None:
      if self.curpos < 3:
        curpose = 0
      else:
        curpose = 6

    if curpos == 6 and pos <= 1:
      curpos = 0
    elif curpos == 0 and pos >= 5:
      curpos = 6
    elif (curpos - pos) > 1:
      curpos = pos + 1
    elif (curpos - pos) < -1:
      curpos = pos - 1

  # print("curpos %d pos %d" %(curpos, pos))
    return curpos

  def strpos(self, pos):
    if pos == 0:
      return "off course left"
    elif pos == 1:
      return "near left line"
    elif pos == 2:
      return "left of center"
    elif pos == 3:
      return "center"
    elif pos == 4:
      return "right of center"
    elif pos == 5:
      return "near right line"
    elif pos == 6:
      return "off course right"

  def lrcsort(self, lines):
    # curpos ('l/r/c of l/r/c line': 0-8
    curlline = []
    currline = []
    curcline = []
    lassigned = False
    rassigned = False
    cassigned = False
    # print("lines:")
    # print(lines)
    unassigned = []
    if lines is not None:
      for line in lines:
        lassigned = False
        rassigned = False
        cassigned = False
        # print("lines:")
        if line is not None:
            for i in range(3):
              # check the easy way first
              # are we close to one of the last 3 readings
              if self.lline[i] is not None and self.check(self.lline[i], line):
              # print("lcheck")
                curlline.append(line )
                lassigned = True
                break
              elif self.rline[i] is not None and self.check(self.rline[i], line):
              # print("rcheck1")
                currline.append(line )
                rassigned = True
                break
              elif self.cline[i] is not None and self.check(self.cline[i], line):
                # ARD TODO: problem if cl was really rl or ll
                curcline.append(line )
                cassigned = True
                break
            if not (cassigned or rassigned or lassigned):
              # print(line)
            # print("unassigned")
              unassigned.append(line )

      for line in unassigned:
      # for line_i in unassigned:
       # for line in line_i:
        # print("line: %d %d" % (((line[1][1]) - (line[0][1])), (line[1][0] - (line[0][0]))))
        # print(line)
        deg = self.orientation(line)
      # print("lrc: deg %d" % deg)
        if deg <= 90:
          if not lassigned:
            # print("lrc: append ll")
            curlline.append(line )
        elif deg > 90 and deg < 180:
          if not rassigned:
            # print("lrc: append rl")
            currline.append(line )
        if not cassigned:
          # print("lrc: append cl")
          curcline.append(line )
    return currline, curlline, curcline

  def saveDonkeyState(self):
        name = "donkeystate.json" 
        checksum = self.line_color_white_count + self.line_color_white_count + int(self.laneWidth)
        if self.checksumDonkeyState >= checksum:
          return
        self.checksumDonkeyState = checksum
        file_path = config_path=os.path.expanduser(os.path.join(cfg.RL_STATE_PATH, name))
        # out = { 'extra': {} }
        out = {}

        out['line_color_white_count'] = self.line_color_white_count
        out['line_color_white_var'] = self.line_color_white_var
        out['line_color_white_mean'] = self.line_color_white_mean
        out['line_color_yellow_count'] = self.line_color_yellow_count
        out['line_color_yellow_var'] = self.line_color_yellow_var
        out['line_color_yellow_mean'] = self.line_color_yellow_mean
        out['line_color_simple_count'] = self.line_color_simple_count
        out['line_color_simple_var'] = self.line_color_simple_var
        out['line_color_simple_mean'] = self.line_color_simple_mean
        out['laneWidth'] = self.laneWidth
        out['minThrottle'] = self.TB.minthrottle
        f = open(file_path, 'w')
        json.dump(out, f)
        f.close()

  def loadDonkeyState(self):
        print("loadDonkeyState")
        name = "donkeystate.json" 
        file_path = config_path=os.path.expanduser(os.path.join(cfg.RL_STATE_PATH, name))
        try:
            with open(file_path, 'r') as fp:
                json_data = json.load(fp)
            self.line_color_white_count = json_data['line_color_white_count'] 
            self.line_color_white_var = json_data['line_color_white_var'] 
            self.line_color_white_mean = json_data['line_color_white_mean'] 
            self.line_color_yellow_count = json_data['line_color_yellow_count'] 
            self.line_color_yellow_var = json_data['line_color_yellow_var'] 
            self.line_color_yellow_mean = json_data['line_color_yellow_mean'] 
            self.line_color_simple_count = json_data['line_color_simple_count'] 
            self.line_color_simple_var = json_data['line_color_simple_var'] 
            self.line_color_simple_mean = json_data['line_color_simple_mean'] 
            self.laneWidth = json_data['laneWidth'] 
            self.TB.minthrottle = json_data['minThrottle'] 
            # self.TB.set_minthrottle(json_data['minThrottle'])
            print(json_data)
            print("lw %d" % self.laneWidth)
        except UnicodeDecodeError:
            raise Exception('bad record: %d. You may want to run `python manage.py check --fix`' % ix)
        except FileNotFoundError:
            self.line_color_simple_mean    = [0,0,0]
            self.line_color_yellow_mean    = [0,0,0]
            self.line_color_white_mean     = [0,0,0]
            self.line_color_simple_var     = [0,0,0]
            self.line_color_yellow_var     = [0,0,0]
            self.line_color_white_var      = [0,0,0]
            self.line_color_simple_count   = 0
            self.line_color_yellow_count   = 0
            self.line_color_white_count    = 0
            self.laneWidth = -1
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

  def getDonkeyState(self):
    return self.line_color_white_count, self.line_color_white_var, self.line_color_white_mean, self.line_color_yellow_count, self.line_color_yellow_var, self.line_color_yellow_mean,  self.line_color_simple_count, self.line_color_simple_var, self.line_color_simple_mean, self.laneWidth, self.TB.minthrottle

  def setDonkeyState(self, wcnt, wvar, wmean, ycnt, yvar, ymean, scnt, svar, smean, lw):
    self.line_color_white_count = wcnt
    self.line_color_white_var = wvar
    self.line_color_white_mean = wmean
    self.line_color_yellow_count = ycnt
    self.line_color_yellow_var = yvar
    self.line_color_yellow_mean = ymean
    self.line_color_simple_count = scnt
    self.line_color_simple_var = svar
    self.line_color_simple_mean = smean
    self.laneWidth = lw

  def createLineIterator(self, line, img, prev_n, prev_var, prev_mean):
    """
    https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator
    Produces and array that consists of the coordinates and intensities of 
    each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point 
             (x,y)
        -P2: a numpy array that consists of the coordinate of the second point
             (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of 
              each pixel in the radii (shape: [numPixels, 3], 
              row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    if line is None:
      return prev_n, prev_var, prev_mean
    P1X = line[0][0]
    P1Y = line[0][1]
    P2X = line[1][0]
    P2Y = line[1][1]
 
    #difference and absolute difference between points
    #used to calculate 
 
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)
 
    #predefine numpy array for output based on distance between points
    xybuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    xybuffer.fill(np.nan)
 
    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        xybuffer[:,0] = P1X
        if negY:
            xybuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            xybuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        xybuffer[:,1] = P1Y
        if negX:
            xybuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            xybuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                xybuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                xybuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            xybuffer[:,0] = (slope*(xybuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                xybuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                xybuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            xybuffer[:,1] = (slope*(xybuffer[:,0]-P1X)).astype(np.int) + P1Y
 
    #Remove points outside of image
    colX = xybuffer[:,0]
    colY = xybuffer[:,1]
    xybuffer= xybuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]
 
    #Get intensities from img ndarray
    it = []
    # V = []
    # m = []
    it[:] = img[xybuffer[:,1].astype(np.uint),xybuffer[:,0].astype(np.uint)]
    # print("it ", it)

    # compute new variance and mean
    n = prev_n
    V = prev_var
    m = prev_mean
    for x_i in it:
      n = n + 1
      m[:] = m[:] + (x_i[:] - m[:]) / n
      V[:] = V[:] + (x_i[:] - m[:]) * (x_i[:] - prev_mean[:])
      # for i in range(3):
      #   m[i] = m[i] + (x_i[i] - m[i]) / n
      #   V[i] = V[i] + (x_i[i] - m[i]) * (x_i[i] - prev_mean[i])
    # math.sqrt(variance)
    # return m, V, n
    # print("n ", n," V ", V, " m ", m)
    return n, V, m

  def lrclines(self, currline, curlline, curcline, roi):
    maxpix = self.width 
    midpix = maxpix / 2
    rllen = len(currline)
    lllen = len(curlline)
    cllen = len(curcline)
    done = False
    self.bestrl = None
    self.bestll = None
    self.bestcl = None
    self.bestvx = 10000
    self.bestvy = 10000
    self.bestvprl = None
    self.bestvpll = None
    self.bestvpcl = None
    self.curconf = 0
    self.curpos = 3
    # print("len ll %d cl %d rl %d"%(lllen, cllen, rllen))
    if rllen > 0 and lllen > 0 and cllen > 0:
      for rl in currline:
        rldeg = self.orientation(rl)
        for ll in curlline:
          lldeg = self.orientation(ll)
          '''
          Check for the perfect lrc
          '''
          # if rl[0][0] <= ll[0][0] or rl[1][0] <= ll[1][0]:
          if ((rl[0][0] <= ll[0][0] and rl[1][0] >= ll[1][0]) or
              (rl[0][0] >= ll[0][0] and rl[1][0] <= ll[1][0])):
          # print("F1")
            continue
          if rl[0][0] <= midpix and (ll[0][0] >= midpix or ll[1][0] >= midpix):
          # print("F2")
            continue
          for cl in curcline:
            cldeg = self.orientation(cl)
            '''
             it is possible that previous cl was really ll or rl.
             Allow for this possibility.
            '''
            if ((rl[0][0] <= cl[0][0] and rl[1][0] >= cl[1][0]) or
                (rl[0][0] >= cl[0][0] and rl[1][0] <= cl[1][0])):
              # print("F3")
              continue
            if ((ll[0][0] >= cl[0][0] and ll[1][0] <= cl[1][0]) or
                (ll[0][0] <= cl[0][0] and ll[1][0] >= cl[1][0])):
              # print("F4")
            # print(cl)
            # print(ll)
              continue
            if ll[0][0] >= midpix and (cl[0][0] >= midpix or cl[1][0] >= midpix):
              # print("F5")
              continue
            # print("rl, cl, ll:")
            # print(rl)
            # print(cl)
            # print(ll)
            vb,vx,vy = self.vanishing_point(rl, cl, ll)
            if vb:
              # found vanishing point, clear emergency start
              self.TB.setEmergencyStop(False) 
              self.curpos = 3
              self.bestrl = rl
              self.bestll = ll
              self.bestcl = cl
              # print("VP FOUND")
              if ((abs(self.bestrl[0][0] - self.bestcl[0][0]) > self.laneWidth / 2) and
                  (abs(self.bestcl[0][0] - self.bestll[0][0]) > self.laneWidth / 2)):
                # print("VP LL")
                # print(bestll)
                # print("VP CL")
                # print(bestcl)
                # print("VP RL")
                # print(bestrl)

                if cfg.USE_COLOR:
                  # roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                  self.line_color_white_count, self.line_color_white_var, self.line_color_white_mean = self.createLineIterator(self.bestrl, roi, self.line_color_white_count, self.line_color_white_var, self.line_color_white_mean)
                  self.line_color_white_count, self.line_color_white_var, self.line_color_white_mean = self.createLineIterator(self.bestll, roi, self.line_color_white_count, self.line_color_white_var, self.line_color_white_mean)
                  self.line_color_yellow_count, self.line_color_yellow_var, self.line_color_yellow_mean = self.createLineIterator(self.bestcl, roi, self.line_color_yellow_count, self.line_color_yellow_var, self.line_color_yellow_mean)

                self.curconf = cfg.MAX_ACCEL
                self.curpos = 3
                done = True
            elif (vx+vy) < (self.bestvx+self.bestvy):
              self.bestvprl = rl
              self.bestvpll = ll
              self.bestvpcl = cl
              self.bestvx = vx
              self.bestvy = vy
              self.donkeyvpangle = None
              self.dklinept = None
              self.curpos = 3
              self.curconf = min(cfg.MAX_ACCEL,int(100/(self.bestvx + self.bestvy)))
            if done:
              break
          if done:
            break
        if done:
          break
    if done:
      pass
    elif rllen == 1 and lllen == 1 and cllen == 0:
      self.bestrl = currline[0]
      self.bestll = curlline[0]
      rldeg = self.orientation(self.bestrl)
      lldeg = self.orientation(self.bestll)
      x,y, dkangle, lanewidth1 = self.vanishing_point2(self.bestrl,self.bestll)
    # print("Possible vp2 %d" % y)
      if self.bestrl[0][0] <= self.bestll[0][0] or self.bestrl[1][0] <= self.bestll[1][0]:
        if abs((rldeg - 90) - (90 - lldeg)) < 10 and self.bestrl[0][0] > 120 and self.bestll[0][0] < 40:
          self.bestcl = [[0,0],[0,0]]
          self.bestcl[0][0] = int((self.bestrl[0][0] + self.bestll[0][0])/2)
          self.bestcl[0][1] = int((self.bestrl[0][1] + self.bestll[0][1])/2)
          self.bestcl[1][0] = int((self.bestrl[1][0] + self.bestll[1][0])/2)
          self.bestcl[1][1] = int((self.bestrl[1][1] + self.bestll[1][1])/2)
        # print("estimated cl")
          for i in range(3):
            if self.cline is not None and self.cline[i] is not None and self.check(self.cline[i], self.bestcl):
              done = True
              self.curconf = min(cfg.MAX_ACCEL,int(100/(self.bestvx + self.bestvy)))
              self.curpos = 3
        if not done and self.pos[2] < 2:
          self.bestcl = self.bestrl
          self.bestrl = None
          self.curpos = 1
        elif not done and self.pos[2] > 3:
          self.bestcl = self.bestll
          self.bestll = None
          self.curpos = 5

    elif (rllen >= 1 or lllen >= 1) and cllen >= 1:
    # print("Try again: len ll %d cl %d rl %d"%(lllen, cllen, rllen))
      self.bestval = 10000
      bestx = 10000
      besty = 10000
      bestw = 10000
      lanewidth1 = None
      lwmaxdif = 10000
      for cl in curcline:
        cldeg = self.orientation(cl)
        if (lllen >= 1):
          for ll in curlline:
            lldeg = self.orientation(ll)
            '''
            if lldeg > cldeg:
              continue
            if ll[0][0] > cl[0][0]:
              continue
            '''
            if self.bestcl is None or 120 > cldeg > 60:
              x,y, self.donkeyvpangle, lanewidth1 = self.vanishing_point2(ll,cl)
              if (self.VPy is None or besty > abs(self.VPy - y)) and y <= 0:
              # print("possibe vp2")
                if self.laneWidth is None or abs(lanewidth1 - self.laneWidth) < lwmaxdif or abs(lanewidth1/2 - self.laneWidth) < lwmaxdif:
                # print("found possible best cl, ll")
                  lwmaxdif = min(abs(lanewidth1 - self.laneWidth), abs(lanewidth1/2 - self.laneWidth))
                  if lwmaxdif == abs(lanewidth1/2 - self.laneWidth):
                    bestcl = None
                    bestrl = cl
                  else:
                    bestcl = cl
                    bestrl = None
                  bestll = ll
                  bestx = x
                  besty = y
                  bestlw = lanewidth1
          '''
          if (self.VPx is None):
            self.VPy = besty
            self.VPx = bestx
            # laneWidth = lanewidth1
          else:
            self.VPy = .8*self.VPy + .2*besty
            self.VPx = .8*VPx + .2*bestx
            if self.laneWidth is None or lanewidth1 is None:
              pass
            else:
              if abs(lanewidth1 - self.laneWidth) < lwmaxdif:
              # self.laneWidth = .8 * self.laneWidth + .2*lanewidth1
              elif abs(lanewidth1/2 - self.laneWidth) < lwmaxdif:
              # self.laneWidth = .8 * self.laneWidth + .2*lanewidth1/2
          '''
        if (rllen >= 1):
          for rl in currline:
            rldeg = self.orientation(rl)
            '''
            if rldeg < cldeg:
              continue
            if rl[0][0] < cl[0][0]:
              continue
            '''
            if self.bestcl is None or 120 > cldeg > 60:
              x,y, self.donkeyvpangle, lanewidth1 = self.vanishing_point2(rl,cl)
              if (self.VPy is None or besty > abs(self.VPy - y)) and y <= 0:
              # print("Possibe vp2")
                if self.laneWidth < 0 or abs(lanewidth1 - self.laneWidth) < lwmaxdif or abs(lanewidth1/2 - self.laneWidth) < lwmaxdif:
                  lwmaxdif = min(abs(lanewidth1 - self.laneWidth), abs(lanewidth1/2 - self.laneWidth))
                # print("Found possible best cl, rl")
                  if self.laneWidth > 0 and lwmaxdif == abs(lanewidth1/2 - self.laneWidth):
                    self.bestcl = None
                    self.bestll = cl
                  else:
                    self.bestcl = cl
                    self.bestll = None
                  self.bestrl = rl
                  bestx = x
                  besty = y
                  bestlw = lanewidth1
        # apply to best of rllen >= 1 and llen >= 1
        if (self.VPx is None):
          self.VPy = besty
          self.VPx = bestx
          # laneWidth = lanewidth1
        else:
          self.VPy = .8*self.VPy + .2*besty
          self.VPx = .8*self.VPx + .2*bestx
          # print("lw : vp2 %d" % lanewidth1)
          ''' 
          # ARD: Need to use 3-line VP.
          # 2-line vp2 laneWidth is sometimes way off. 
          # Need to make sure within a range and angle
          # For most tracks, not worthwhile
          if laneWidth is None or lanewidth1 is None:
            pass
          else:
            if abs(lanewidth1 - laneWidth) < lwmaxdif:
              # laneWidth = .8 * laneWidth + .2*lanewidth1
            elif abs(lanewidth1/2 - laneWidth) < lwmaxdif:
              # laneWidth = .8 * laneWidth + .2*lanewidth1/2
          ''' 

    elif not done:
    # print("not done")
      self.throttle = 0
      self.steering = 0
      
      self.bestcl = None
      if curcline is not None:
        mindif = 10000
        maxlen = 0
        for cl in curcline:
          cldeg = self.orientation(cl)
          if self.cline[2] is not None:
            lastcldeg = self.orientation(self.cline[2])
            if abs(cldeg - lastcldeg) < mindif:
              self.bestcl = cl
              mindif = abs(cldeg - lastcldeg) 
            # print("cl mindif %d" % mindif)
          else:
            cllen = math.hypot(cl[0][0] - cl[1][0], cl[1][1] - cl[1][1])
            if cllen > maxlen:
              maxlen = cllen
              self.bestcl = cl
            # print("cl maxlen %d" % mindif)
      self.bestll = None
      if curlline is not None:
        mindif = 10000
        maxlen = 0
        for ll in curlline:
          lldeg = self.orientation(ll)
          if self.lline[2] is not None:
            lastlldeg = self.orientation(self.lline[2])
            if abs(lldeg - lastlldeg) < mindif:
              self.bestll = ll
              mindif = abs(lldeg - lastlldeg) 
            # print("ll mindif %d" % mindif)
          else:
            lllen = math.hypot(ll[0][0] - ll[1][0], ll[1][1] - ll[1][1])
            if lllen > maxlen:
              self.bestll = ll
              maxlen = lllen
            # print("ll maxlen %d" % mindif)
      self.bestrl = None
      if currline is not None:
        mindif = 10000
        maxlen = 0
        for rl in currline:
          rldeg = self.orientation(rl)
          if self.rline[2] is not None:
            lastrldeg = self.orientation(self.rline[2])
            if abs(rldeg - lastrldeg) < mindif:
              self.bestrl = rl
              mindif = abs(rldeg - lastrldeg) 
            # print("rl mindif %d" % mindif)
          else:
            rllen = math.hypot(rl[0][0] - rl[1][0], rl[1][1] - rl[1][1])
            if rllen > maxlen:
              self.bestrl = rl
              maxlen = rllen
            # print("rl maxlen %d" % mindif)
      self.curconf = 0

    if self.bestrl is not None and self.bestcl is not None and self.bestrl[0][0] < self.bestcl[0][0] and self.bestrl[1][0] < self.bestcl[1][0]:
      tmp = self.bestcl
      self.bestcl = self.bestrl
      self.bestrl = tmp
    if self.bestll is not None and self.bestcl is not None and self.bestcl[0][0] < self.bestll[0][0] and self.bestcl[1][0] < self.bestll[1][0]:
      tmp = self.bestll
      self.bestll = self.bestcl
      self.bestcl = tmp
    if self.bestrl is not None and self.bestll is not None and self.bestrl[0][0] < self.bestll[0][0] and self.bestrl[1][0] < self.bestll[1][0]:
      tmp = self.bestll
      self.bestll = self.bestrl
      self.bestrl = tmp
    if self.bestrl is not None and self.bestll is not None and self.bestrl[0][0] == self.bestll[0][0] and self.bestrl[1][0] == self.bestll[1][0]:
      # self.bestrl and self.bestll are the same
      if midpix > self.bestll[0][0]:
        self.bestrl = None
      else:
        self.bestll = None

    if  ((self.bestrl is not None and self.bestll is not None and self.bestcl is None and abs(self.bestrl[0][0] - self.bestll[0][0]) < self.laneWidth / 2) and abs(self.bestrl[0][0] - self.bestll[0][0]) > self.laneWidth / 6):
      # bestrl is too close to bestll. One is center
      for i in (2,1,0):
        foundR = False
        foundL = False
        foundC = False
        if self.bestll is not None and self.cline[i] is not None and self.check(self.bestll, self.cline[i]):
          self.bestcl = self.bestll
          self.bestll = None
          foundC = True
          break
        if self.bestrl is not None and self.cline[i] is not None and self.check(self.bestrl, self.cline[i]):
          self.bestcl = self.bestrl
          self.bestrl = None
          foundC = True
          break
        if self.bestll is not None and self.lline[i] is not None and self.check(self.bestll, self.lline[i]):
          foundL = True
        if self.bestrl is not None and self.rline[i] is not None and self.check(self.bestrl, self.rline[i]):
          foundR = True
      if foundC:
        pass
      elif foundL and not foundR:
        self.bestcl = self.bestrl
        self.bestrl = None
      elif not foundL and foundR:
        self.bestcl = self.bestll
        self.bestll = None
          
    if  ((self.bestrl is not None and self.bestll is not None and abs(self.bestrl[0][0] - self.bestll[0][0]) < self.laneWidth / 6) or
        (self.bestcl is not None and self.bestll is not None and abs(self.bestcl[0][0] - self.bestll[0][0]) < self.laneWidth / 6) or
        (self.bestcl is not None and self.bestrl is not None and abs(self.bestcl[0][0] - self.bestrl[0][0]) < self.laneWidth / 6)):
      # best lines are too close
      foundR = False
      foundL = False
      foundC = False
      for i in (2,1,0):
        if self.bestrl is not None and self.rline[i] is not None and self.check(self.bestrl, self.rline[i]):
          foundR = True
        if self.bestll is not None and self.lline[i] is not None and self.check(self.bestll, self.lline[i]):
          foundL = True
        if self.bestcl is not None and self.cline[i] is not None and self.check(self.bestcl, self.cline[i]):
          foundC = True
      if  (self.bestrl is not None and self.bestll is not None and abs(self.bestrl[0][0] - self.bestll[0][0]) < self.laneWidth / 6):
        if (foundR and foundL) or (not foundR and not foundL):
          if midpix > self.bestll[0][0]:
            self.bestrl = None
          else:
            self.bestll = None
        elif foundR and not foundL:
            self.bestll = None
        elif not foundR and foundL:
            self.bestrl = None

      if (self.bestcl is not None and self.bestll is not None and abs(self.bestcl[0][0] - self.bestll[0][0]) < self.laneWidth / 6):
        if (foundC and foundL) or (not foundC and not foundL):
          if abs(midpix - self.bestcl[0][0]) < self.laneWidth / 6:
            self.bestll = None
          elif (midpix - self.bestll[0][0]) > self.laneWidth / 6:
            self.bestcl = None
          else:
            self.bestll = None
        elif foundC and not foundL:
            self.bestll = None
        elif not foundC and foundL:
            self.bestcl = None

      if (self.bestcl is not None and self.bestrl is not None and abs(self.bestcl[0][0] - self.bestrl[0][0]) < self.laneWidth / 6):
        if (foundC and foundR) or (not foundC and not foundR):
          if abs(midpix - self.bestcl[0][0]) < self.laneWidth / 6:
            self.bestrl = None
          elif (self.bestrl[0][0] - midpix) > self.laneWidth / 6:
            self.bestcl = None
          else:
            self.bestrl = None
        elif foundC and not foundR:
            self.bestrl = None
        elif not foundC and foundR:
            self.bestcl = None


    '''
    # following has been replaced by above
    if  self.bestrl is not None and self.bestll is not None and self.bestrl[0][0] == self.bestll[0][0] and self.bestrl[0][1] == self.bestll[0][1] and self.bestrl[1][0] == self.bestll[1][0] and self.bestrl[1][1] == self.bestll[1][1]:
        if self.rline[i] is not None and self.check(self.bestrl, self.rline[i]):
          self.bestll = None
          break
        elif self.lline[i] is not None and self.check(self.bestll, self.lline[i]):
          self.bestrl = None
          break
        elif self.lline[i] is None and self.bestll is None and self.bestrl is not None:
          self.bestll = None
          break
        elif self.rline[i] is None and self.bestrl is None and self.bestll is not None:
          self.bestrl = None
          break
        else:
          if self.rline[i] is not None and self.bestll is not None:
            rl = self.rline[i]
            if self.bestll[0][0] < rl[0][0] and self.bestll[1][0] < rl[1][0]:
              self.bestrl = None
              break
          if self.lline[i] is not None and self.bestll is not None:
            ll = self.lline[i]
            if self.bestrl[0][0] > ll[0][0] and self.bestrl[1][0] > ll[1][0]:
              self.bestll = None
              break
    '''

    ########################
    if (((self.bestrl is None and self.bestvprl is not None) or
         (self.bestrl is not None  and np.array_equal(self.bestrl,self.bestvprl))) and
        ((self.bestcl is None and self.bestvpcl is not None) or
         (self.bestcl is not None  and np.array_equal(self.bestcl,self.bestvpcl))) and
        ((self.bestll is None and self.bestvpll is not None) or
         (self.bestll is not None  and np.array_equal(self.bestcl,self.bestvpll)))):
      self.bestrl = self.bestvprl
      self.bestcl = self.bestvpcl
      self.bestll = self.bestvpll
    ########################
    # Done, set globals and return vals
    ########################
    tmppos = self.setCurPos(self.bestll, self.bestcl, self.bestrl, self.pos[2])
    if tmppos >= 0:
      self.curpos = tmppos
    del(self.lline[0])
    self.lline.append(self.bestll)
    del(self.cline[0])
    self.cline.append(self.bestcl)
    del(self.rline[0])
    self.rline.append(self.bestrl)
    del(self.pos[0])
    self.pos.append(self.curpos)
    self.conf = self.curconf
    # print("self.lline")
    # print(self.lline)
    # print(self.bestll)
    # print("self.cline")
    # print(self.cline)
    # print(self.bestcl)
    # print(self.cline[2])
    # print("self.rline")
    # print(self.cline)
    # print(self.bestrl)
    # print(self.rline[2])
   
    self.conf, self.steering, self.throttle = self.setSteerThrottle(self.curpos, self.lline[2], self.cline[2], self.rline[2], self.conf)

    print ("steer %f throt %f conf %d pos(%s)" % (self.steering, self.throttle, self.conf, self.strpos(self.pos[2])))
    #######################
    # print to screen
    #######################
    croi = copy.deepcopy(roi)
    if self.lline is not None or self.rline is not None or self.cline is not None:
      lrclines = []
      str1 = "final: "
      if self.lline[2] is not None:
        lrclines.append(self.lline[2])
        str1 += "ll "
      if self.rline[2] is not None:
        lrclines.append(self.rline[2])
        str1 += "rl "
      if self.cline[2] is not None:
        lrclines.append(self.cline[2])
        str1 += "cl "
      print(str1)
      for line in lrclines:
        if line is not None:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[1][0]
            y2 = line[1][1]
            cv2.line(croi,(x1,y1),(x2,y2),(0,255,0),2)
   
    '''
    cv2.imshow(str(self.seq),croi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    if cfg.SAVE_MOVIE:
      out = self.image_path(cfg.MOVIE_LOC, self.seq)
      cv2.imwrite(out, croi)
      print("wrote %s" % (out))
    # ffmpeg -framerate 4 -i /tmp/movie4/1%03d_cam-image_array_.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

    self.TB.checkThrottle(self.throttle)
    self.saveDonkeyState()
    return self.steering, self.throttle

  def binary_hsv_mask(self, img, color_range):
    lower = np.array(color_range[0])
    upper = np.array(color_range[1])
    return cv2.inRange(img, lower, upper)

  def process_img_color(self, img):
    if self.TB.throttleCheckInProgress():
      self.TB.setMinMaxThrottle(img)
    simplecl = None
    ymergedlines = None
    wmergedlines = None
    smergedlines = None
    roi = self.getROI(img)
    roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cmask = self.binary_hsv_mask(roi, self.line_color_simple)
    cimg = cv2.bitwise_and(roi, roi, mask = cmask)
    # edges = cv2.Canny(roi, 100, 200)
    hb = HoughBundler(self.width, self.height, self.bestll, self.bestcl, self.bestrl)
    print("process_img_color ", self.scannymin, self.scannymax)
    for i in range(self.scannymin, self.scannymax):
        edges = cv2.Canny(cimg, i*10, i*20)
        lines = cv2.HoughLinesP(edges, 1, np.pi/90, 13, 20, 20, 20)
        # simple line follower
        simplecl = hb.line_follower(lines)
        print("simplecl: ", simplecl)
        self.line_color_simple_count, self.line_color_simple_var, self.line_color_simple_mean = self.createLineIterator(simplecl, roi, self.line_color_simple_count, self.line_color_simple_var, self.line_color_simple_mean)
        if simplecl is not None:
          self.scannylast = i
          break

    # cmask = self.binary_hsv_mask(roi, cfg.COLOR_YELLOW)
    cmask = self.binary_hsv_mask(roi, self.line_color_yellow)
    cimg = cv2.bitwise_and(roi, roi, mask = cmask)
    # edges = cv2.Canny(roi, 100, 200) 
    # hb = HoughBundler(self.width, self.height, self.bestll, self.bestcl, self.bestrl)
    for i in range(self.ycannymin, self.ycannymax):
        edges = cv2.Canny(cimg, i*10, i*20) 
        lines = cv2.HoughLinesP(edges, 1, np.pi/90, 13, 20, 20, 20)
        # simple line follower
        # simplecl = hb.line_follower(lines)
        ymergedlines = hb.process_lines(lines)
        if ymergedlines is not None:
          self.ycannylast = i
          break

    # cmask = self.binary_hsv_mask(roi, cfg.COLOR_WHITE)
    cmask = self.binary_hsv_mask(roi, self.line_color_white)
    cimg = cv2.bitwise_and(roi, roi, mask = cmask)
    # edges = cv2.Canny(roi, 100, 200) 
    # edges = cv2.Canny(cimg, 100, 200) 
    for i in range(self.wcannymin, self.wcannymax):
        edges = cv2.Canny(cimg, i*10, i*30) 
        # hb = HoughBundler(self.width, self.height, self.bestll, self.bestcl, self.bestrl)
        lines = cv2.HoughLinesP(edges, 1, np.pi/90, 13, 20, 20, 20)
        wmergedlines = hb.process_lines(lines)
        if wmergedlines is not None:
          self.wcannylast = i
          break
    return simplecl, wmergedlines, ymergedlines, roi

  def vp_confirmed(self):
    if self.scannymin < self.scannylast and self.scannymin < self.scannymax - 1:
      self.scannymin += 1
    if self.scannymax > self.scannylast and self.scannymin < self.scannymax - 1:
      self.scannymax -= 1
    if self.ycannymin < self.ycannylast and self.ycannymin < self.ycannymax - 1:
      self.ycannymin += 1
    if self.ycannymax > self.ycannylast and self.ycannymin < self.ycannymax - 1:
      self.ycannymax -= 1
    if self.wcannymin < self.wcannylast and self.wcannymin < self.wcannymax - 1:
      self.wcannymin += 1
    if self.wcannymax > self.wcannylast and self.wcannymin < self.wcannymax - 1:
      self.wcannymax -= 1

  def process_img(self, img):

    # print("process_img ")
    if self.TB.throttleCheckInProgress():
      self.TB.setMinMaxThrottle(img)

    roi = self.getROI(img)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(roi, 100, 200) 

    lines = cv2.HoughLinesP(edges, 1, np.pi/90, 13, 20, 20, 20)
    hb = HoughBundler(self.width, self.height, self.bestll, self.bestcl, self.bestrl)
    # simple line follower
    simplecl = hb.line_follower(lines)
    if cfg.USE_COLOR and simplecl is not None:
      # print("2 simplecl: ", simplecl)
      self.line_color_simple_count, self.line_color_simple_var, self.line_color_simple_mean = self.createLineIterator(simplecl, roi, self.line_color_simple_count, self.line_color_simple_var, self.line_color_simple_mean)
    mergedlines = hb.process_lines(lines)
    return simplecl, mergedlines, roi

  def get_map_data(self):
        return self.laneWidth, self.pixPerFrame, self.donkeyvpangle, self.bestll, self.bestcl, self.bestrl, self.width, self.curpos


  def image_path(self, tub_path, frame_id):
        return os.path.join(tub_path, str(frame_id) + "_cam-image_array_.jpg")

  def test_tub(self, tub_path):
        seqs = [ int(f.split("_")[0]) for f in os.listdir(tub_path) if f.endswith('.jpg') ]
        seqs.sort()
        entries = ((os.stat(self.image_path(tub_path, seq))[ST_ATIME], seq) for seq in seqs)

        (last_ts, seq) = next(entries)
        clips = [[seq]]
        for next_ts, next_seq in entries:
            if next_ts - last_ts > 100:  #greater than 1s apart
                clips.append([next_seq])
            else:
                clips[-1].append(next_seq)
            last_ts = next_ts

        # import donkey.parts.map
        # map = Map()

        # inputs = ['user/angle', 'user/throttle', 'cam/image']
        inputs = ['pilot/angle', 'pilot/throttle', 'cam/image']
        types = ['float', 'float', 'image']
        self.tub = Tub(path=tub_path, inputs=inputs, types=types)
        for clip in clips:
          for imgseq in clip:
            # if imgseq < 200:
              # continue
            # if seq < 19:
            #   seq += 1
            #   continue
            imgname = self.image_path(tub_path, imgseq)
            img = cv2.imread(imgname)
            if img is None:
              continue
            self.seq = imgseq

            # rec = self.tub.get_record(imgseq)
            # Did we record he wrong thing? should be pilot?
            # print("tub speed %f  throttle %f" % (float(rec["user/angle"]),float(rec["user/throttle"])))

            # TUB TUB TUB
            if cfg.USE_COLOR:
              if self.line_color_simple_count > 3000:
                # for some unknown reason, 2 stddev is not enough for simple
                stddev[:] = math.sqrt(self.line_color_simple_var[:] /  (self.line_color_simple_count[:] - 1))
                mean[:] = self.line_color_simple_mean[:]
                self.line_color_simple = [[int(mean[0]-3*stddev[0]), int(mean[1]-3*stddev[1]), int(mean[2]-3*stddev[2])], [int(mean[0]+3*stddev[0]), int(mean[1]+3*stddev[1]), int(mean[2]+3*stddev[2])]]
                print("line_color_s: ",self.line_color_simple_count, self.line_color_simple)
              else:
                print("line_color_s_cnt: ", self.line_color_simple_count)

              if self.line_color_yellow_count > 3000:
                # for some unknown reason, 2 stddev is not enough for yellow
                stddev[:] = math.sqrt(self.line_color_yellow_var[:] /  (self.line_color_yellow_count[:] - 1))
                mean[:] = self.line_color_yellow_mean[:]
                self.line_color_yellow = [[int(mean[0]-3*stddev[0]), int(mean[1]-3*stddev[1]), int(mean[2]-3*stddev[2])], [int(mean[0]+3*stddev[0]), int(mean[1]+3*stddev[1]), int(mean[2]+3*stddev[2])]] 
                print("line_color_y: ", self.line_color_yellow_count, self.line_color_yellow)
              else:
                print("line_color_y_cnt: ", self.line_color_yellow_count)
                  
              if self.line_color_white_count > 3000:
                stddev[:] = math.sqrt(self.line_color_white_var[:] / (self.line_color_white_count[:] - 1))
                mean[:] = self.line_color_white_mean[:]
                self.line_color_white = [[int(mean[0]-3*stddev[0]), int(mean[1]-3*stddev[1]), int(mean[2]-3*stddev[2])], [int(mean[0]+3*stddev[0]), int(mean[1]+3*stddev[1]), int(mean[2]+3*stddev[2])]] 
                print("line_color_w: ", self.line_color_white_count, self.line_color_white)
              else:
                print("line_color_w_cnt: ", self.line_color_white_count)
    
            if cfg.USE_COLOR and self.line_color_yellow is not None and self.line_color_white is not None and self.line_color_simple is not None:
              simplecl, wlines, ylines, roi = self.process_img_color(img)
              if wlines is not None:
                currline, curlline, dummycurcline = self.lrcsort(wlines)
              if ylines is not None:
                dummycurrline, dummycurlline, curcline = self.lrcsort(ylines)
              if roi is not None:
                steering, throttle = self.lrclines(currline, curlline, curcline, roi)
            else:
              simplecl, lines, roi = self.process_img(img)
              if lines is not None and roi is not None:
                currline, curlline, curcline = self.lrcsort(lines)
                if cfg.USE_COLOR:
                  # img used to compute line colors
                  steering, throttle = self.lrclines(currline, curlline, curcline, img)
                else:
                  # roi only used for debugging
                  steering, throttle = self.lrclines(currline, curlline, curcline, roi)

            laneWidth, pixPerFrame, self.donkeyvpangle, bestll, bestcl, bestrl, width, curpos = self.get_map_data()


  def get_line_color_info(self):
    if self.line_color_simple_count > 3000:
      stddev = []
      for n in range(3):
        stddev.append(math.sqrt(self.line_color_simple_var[n]/(self.line_color_simple_count-1)))
      self.line_color_simple = [[max(int(self.line_color_simple_mean[0]-4*stddev[0]), 0), max(0,int(self.line_color_simple_mean[1]-4*stddev[1])), max(int(self.line_color_simple_mean[2]-4*stddev[2]),0)], [min(int(self.line_color_simple_mean[0]+4*stddev[0]),255), min(int(self.line_color_simple_mean[1]+4*stddev[1]),255), min(int(self.line_color_simple_mean[2]+4*stddev[2]),255)]]
      print("line_color_s: ",self.line_color_simple_count, self.line_color_simple)
    else:
      print("line_color_s_cnt: ", self.line_color_simple_count)

    if self.line_color_yellow_count > 1000:
      # v = []
      # for i in range(3):
        # v[i] = int(self.line_color_yellow_var[i]/self.line_color_yellow_count)
      # print("ycnt ", self.line_color_yellow_count, " v ", int(self.line_color_yellow_var/self.line_color_yellow_count))
      # print("ycnt ", self.line_color_yellow_count, " v ", v)
      stddev = [] 
      for n in range(3):
        stddev.append(math.sqrt(self.line_color_yellow_var[n]/(self.line_color_yellow_count-1)))
        # stddev[n] = math.sqrt(self.line_color_yellow_var[n])
      # print("ym ", self.line_color_yellow_mean, " sdev ", stddev)
      self.line_color_yellow = [[max(int(self.line_color_yellow_mean[0]-4*stddev[0]), 0), max(0,int(self.line_color_yellow_mean[1]-4*stddev[1])), max(int(self.line_color_yellow_mean[2]-4*stddev[2]),0)], [min(int(self.line_color_yellow_mean[0]+4*stddev[0]),255), min(int(self.line_color_yellow_mean[1]+4*stddev[1]),255), min(int(self.line_color_yellow_mean[2]+4*stddev[2]),255)]] 
      print("line_color_y: ",self.line_color_yellow_count,self.line_color_yellow)
    else:
      print("line_color_y_cnt: ", self.line_color_yellow_count)

    if self.line_color_white_count > 1000:
      # v = []
      # for i in range(3):
        # v[i] = int(self.line_color_white_var[i]/self.line_color_white_count)
      # print("wcnt ", self.line_color_white_count, " v ", v)
      stddev = [] 
      for n in range(3):
        stddev.append( math.sqrt(self.line_color_white_var[n]/(self.line_color_yellow_count-1)))
        # stddev[n] = math.sqrt(self.line_color_white_var[n])
      # print("wm ", self.line_color_white_mean, " sdev ", stddev)
      self.line_color_white = [[max(int(self.line_color_white_mean[0]-3*stddev[0]), 0), max(0,int(self.line_color_white_mean[1]-3*stddev[1])), max(int(self.line_color_white_mean[2]-3*stddev[2]),0)], [min(int(self.line_color_white_mean[0]+3*stddev[0]),255), min(int(self.line_color_white_mean[1]+3*stddev[1]),255), min(int(self.line_color_white_mean[2]+3*stddev[2]),255)]] 
      print("line_color_w: ", self.line_color_white_count,self.line_color_white)
    else:
      print("line_color_w_cnt: ", self.line_color_white_count)

    return self.line_color_simple, self.line_color_yellow, self.line_color_white

  def __init__(self, tb):
    global cfg

    self.line_color_simple         = None
    self.line_color_yellow         = None
    self.line_color_white          = None
    self.checksumDonkeyState       = 0
    self.TB = tb
    self.loadDonkeyState()

    self.donkeyvpangle = None
    self.bestrl = None 
    self.bestll = None
    self.bestcl = None
    self.bestvx = 1000
    self.bestvy = 1000
    self.curconf = -1
    self.curpos = -1000
    cfg = dk.load_config(config_path=os.path.expanduser('~/donkeycar/donkeycar/parts/RLConfig.py'))
    
    # each array is global state cache of 3
    self.lline = [None,None,None]
    self.rline = [None,None,None]
    self.cline = [None,None,None]
    self.throttle = [0,0,0]
    self.steering = [0,0,0]
    self.pos = [3,3,3]
    self.conf = cfg.MAX_ACCEL / 3
    self.seq = 0
    self.width = -1
    self.height = -1
    self.dklinept = None
    self.pixPerFrame = -1
    self.VPx = None
    self.VPy = None
    self.PPFx = None
    self.PPFy = None
    self.bestvprl = None
    self.bestvpcl = None
    self.bestvpll = None
    self.ycannylast = -1
    # self.ycannymin = 0
    # self.ycannymax = 20
    self.ycannymin = 10
    self.ycannymax = 11
    self.wcannylast = 0
    # self.wcannymin = -1
    # self.wcannymax = 20
    self.wcannymin = 10
    self.wcannymax = 11
    self.scannylast = -1
    self.scannymin = 10
    self.scannymax = 11

    
# each array is global state cache of 3
MTB = ThrottleBase()
mll = LaneLines(MTB)
if cfg.SAVE_MOVIE:
  mll.test_tub(cfg.TEST_TUB)
  print("Test_tub done")
else:
  print("Test_tub not done")
