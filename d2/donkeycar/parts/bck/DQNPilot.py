# From main.py in https://github.com/kzhang8850/lane_follower 
import cv2
import rospy
# from cmdVelPublisher import CmdVelPublisher
# from imageSubscriber import ImageSubscriber
from network import DQN

class DQNPilot(ImageSubscriber):
    """
    """

    def __init__(self):
        # super calls to parent classes
        # super(RobotController, self).__init__()

        #initializes the work with starting parameters
        self.dqn = DQN(.0003, .1, .25)
        self.dqn.start()

    def robot_control(self, steering, throttle):
        """
        Given action, will exceute a specificed behavior from the robot
        action.
        """
        try:
            if action < 0 or action > 3:
                raise ValueError("Action is invalid")
            self.state[action].__call__()
        except:
            # make robot stop
            print "Invalid action - stopping robot"
            self.state[3].__call__()

        # self.sendMessage()

    def run(self, img):
        # visualizes the img
        # cv2.imshow('video_window', self.img)
        # cv2.waitKey(5)
        # feeds binary image into dqn to receive action with 
        #   corresponding Q-values
        # ARD: initially based on mergelines.py
        Steering, Throttle, Q = self.dqn.feed_forward(self.img)
        # moves based on move probable action
        self.robot_control(Steering, Throttle)
        # updates the dqn parameters based on what happened from 
        # the action step
        self.dqn.update(self.img)

    def stop(self):
        self.dqn.stop()

if __name__ == '__main__':
    #initializes Robot Controller and runs it
    node = DQNPilot()
    node.run()

