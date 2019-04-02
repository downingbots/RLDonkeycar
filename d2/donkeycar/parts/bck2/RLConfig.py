'''
Example:
  import dk
  cfg = dk.load_config(config_path='~/donkeycar/donkeycar/parts/RLConfig.py')
  print(cfg.CAMERA_RESOLUTION)
'''

MODE_COMPLEX_LANE_FOLLOW = 0 
MODE_SIMPLE_LINE_FOLLOW = 1 

SWITCH_TO_NN = 1500
UPDATE_NN = 500

STATE_EMERGENCY_STOP = 0
STATE_NN = 1
STATE_OPENCV = 2
STATE_MODEL_TRANSFER_STARTED = 3

EMERGENCY_STOP = -1000

DESIREDPPF = 35
MAXBATADJ  = .10
MAXBATCNT  = 1000

CHECK_THROTTLE_THRESH = 20
MAXLANEWIDTH = 400  # should be much smaller

# client to server
MSG_NONE                 = -1
MSG_GET_WEIGHTS          = 1
MSG_ANGLE_THROTTLE_ROI   = 2
MSG_REWARD_ROI           = 3

# server to client
MSG_RESULT               = 4
MSG_WEIGHTS              = 5

# RLPi States
RLPI_READY1 = 1
RLPI_READY2 = 2
RLPI_WAITING = 3

PORT_RLPI = 5557
PORT_CONTROLPI = 5558

# each of throttle's 15 slots is worth the following (4.5 max throttle)
THROTTLE_INCREMENT = .3
ANGLE_INCREMENT = 1       # pass angle bin back and forth
