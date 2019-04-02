'''
Example:
  import dk
  cfg = dk.load_config(config_path='~/donkeycar/donkeycar/parts/RLConfig.py')
  print(cfg.CAMERA_RESOLUTION)
'''


MODE_COMPLEX_LANE_FOLLOW = 0 
MODE_SIMPLE_LINE_FOLLOW = 1 
MODE_STEER_THROTTLE = MODE_COMPLEX_LANE_FOLLOW
# MODE_STEER_THROTTLE = MODE_SIMPLE_LINE_FOLLOW

PARTIAL_NN_CNT = 45000

# SWITCH_TO_NN = 45000
# SWITCH_TO_NN = 15000
# SWITCH_TO_NN = 9000
# SWITCH_TO_NN = 6000
# SWITCH_TO_NN = 300
# SWITCH_TO_NN = 10
# SWITCH_TO_NN = 100
# SWITCH_TO_NN = 150
# SWITCH_TO_NN = 7500
# SWITCH_TO_NN = 3000
SWITCH_TO_NN = 1000
UPDATE_NN = 1000
# UPDATE_NN = 100
SAVE_NN = 1000

THROTTLE_CONSTANT = 0
# THROTTLE_CONSTANT = .3

STATE_EMERGENCY_STOP = 0
STATE_NN = 1
STATE_OPENCV = 2
STATE_MODEL_TRANSFER_STARTED = 3
STATE_MODEL_PREPARE_NN = 4
STATE_MODEL_WEIGHTS_SET = 5
STATE_PARTIAL_NN = 6
STATE_TRIAL_NN = 7

EMERGENCY_STOP = 0.000001
SIM_EMERGENCY_STOP = -1000
# DISABLE_EMERGENCY_STOP = True
DISABLE_EMERGENCY_STOP = False

# USE_COLOR = False
USE_COLOR = True
# HSV
# Note: The YELLOW/WHITE parameters are longer used and are now dynamically computed
# from simulation:
# line_color_y:  [[84, 107, 148], [155, 190, 232]]
# line_color_w:  [[32, 70, 101], [240, 240, 248]]

# COLOR_YELLOW = [[20, 80, 100], [35, 255, 255]]
# COLOR_YELLOW  = [[20, 0, 100], [30, 255, 255]]
# COLOR_YELLOW  = [[20, 40, 70], [70, 89, 95]]
COLOR_YELLOW  = [[84, 107, 148], [155, 190, 232]]

# COLOR_YELLOW  = 50, 75, 83
# using saturation 40 for WHITE
# COLOR_WHITE   = [[0,0,255-40],[255,40,255]]
# COLOR_WHITE   = [[50,0,80],[155,10,100]]
COLOR_WHITE   = [[32, 70, 101], [240, 240, 248]]
# COLOR_WHITE   = 72,2, 90]


TURNADJ = .02
# DESIREDPPF = 35
DESIREDPPF = 20
# MAXBATADJ  = .10
# BATADJ  = .001
MAXBATADJ  = .000  # simulation doesn't have battery
BATADJ  = .000     # simulation doesn't have battery
RL_MODEL_PATH = "~/d2/models/rlpilot"
RL_STATE_PATH = "~/d2/models/"
MAXBATCNT = 1000
# MINMINTHROT = 0.035  # for Sim
MINMINTHROT = 0.05  # for Sim
# OPTFLOWTHRESH = 0.75 # for Sim
OPTFLOWTHRESH = 0.14 # for Sim
OPTFLOWINCR = 0.01   # for Sim
# OPTFLOWINCR = 0.01   # for Sim
# MINMINTHROT = 25 # real car
# MINMINTHROT = 29 # real car
# OPTFLOWTHRESH = 0.40   # real
# OPTFLOWINCR = 0.001

# MAX_ACCEL = 10
MAX_ACCEL  = 3

# CHECK_THROTTLE_THRESH = 20
CHECK_THROTTLE_THRESH = 40
MAXLANEWIDTH = 400  # should be much smaller
MIN_DIST_FROM_CENTER = 20

# client to server
MSG_NONE                              = -1
MSG_GET_WEIGHTS                       = 1
MSG_STATE_ANGLE_THROTTLE_REWARD_ROI   = 2

# server to client
MSG_RESULT                            = 4
MSG_WEIGHTS                           = 5
MSG_EMERGENCY_STOP                    = 6

# control1 to control2
MSG_ROI                               = 7

# control2 to control1
MSG_ANGLE_THROTTLE_REWARD             = 8

# RLPi States
RLPI_READY1   = 1
RLPI_READY2   = 2

RLPI_OPENCV   = 1
RLPI_TRIAL_NN = 2
RLPI_NN       = 3
# PORT_RLPI = "10.0.0.5:5557"
# PORT_RLPI = "localhost:5557"
# PORT_CONTROLPI = "localhost:5558"
PORT_RLPI         = 5557
PORT_CONTROLPI    = 5558
PORT_CONTROLPI2   = None
PORT_CONTROLPI2RL = None
# PORT_CONTROLPI2   = 5556
# PORT_CONTROLPI2RL = 5555

# Original reward for throttle was too high. Reduce.
# THROTTLE_INCREMENT = .4
# THROTTLE_BOOST = .1
THROTTLE_INCREMENT = .3
THROTTLE_BOOST = .05
REWARD_BATCH_MIN = 3
REWARD_BATCH_MAX = 10
REWARD_BATCH_END = 50
REWARD_BATCH_BEGIN = 500
# pass angle bin back and forth; based on 15 bins
ANGLE_INCREMENT = (1/15) 

SAVE_MOVIE = False
# SAVE_MOVIE = True
TEST_TUB = "/home/ros/d2/data/tub_18_18-08-18"
MOVIE_LOC = "/tmp/movie4"
# to make movie from jpg in MOVIE_LOC use something like:  
# ffmpeg -framerate 4 -i /tmp/movie4/1%03d_cam-image_array_.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

# INIT_STEER_FRAMES = 25
INIT_STEER_FRAMES = 125

# For PPO 
Q_LEN_THRESH = 200
Q_LEN_MAX = 250
# Almost all initial batches were under 20 
Q_FIT_BATCH_LEN_THRESH = 50
# Very short batches to compute rewards are likely "car resets"
# maybe should be 5
Q_MIN_ACTUAL_BATCH_LEN = 4
RWD_LOW  = 10
RWD_HIGH = 500
RWD_HIGH_THRESH = 3
DISCOUNT_FACTOR = .8

