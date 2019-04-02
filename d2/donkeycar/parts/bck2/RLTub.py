'''
  RLTub.py

  run RL on Tub or set of Tubs to create and save model.
'''
class RLTub():

  def image_path(self, tub_path, frame_id):
        return os.path.join(tub_path, str(frame_id) + "_cam-image_array_.jpg")

  def test_tub(self, tub_path):
        global width, height, pixPerFrame, laneWidth, donkeyvpangle
        global seq

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
            # global seq
            # if seq < 19:
            #   seq += 1
            #   continue
            imgname = self.image_path(tub_path, imgseq)
            img = cv2.imread(imgname)
            if img is None:
              continue
            seq = imgseq

            # rec = self.tub.get_record(imgseq)
            # Did we record he wrong thing? should be pilot?
            # print("tub speed %f  throttle %f" % (float(rec["user/angle"]),float(rec["user/throttle"])))

            simplecl, lines, roi = self.process_img(img)
            # roi = self.getROI(img)
            if lines is not None:
              steering, throttle = self.lrclines(lines,roi)
            if simplecl is not None:
              pos = 4
              conf = 10
              conf, steering, throttle = self.setSteerThrottle(pos, None, simplecl, None, conf)

            # map.update(steering, throttle, lines, dklinept, laneWidth, pixPerFrame)
            
            # return steering, position

  def __init()__:
    self.test_tub("/home/ros/d2/data/tub_153_18-07-29")

