# RLDonkeycar: Real-time reinforcement learning of a donkeycar with two raspberry pi's

### Overview

The High-Level Goal is to plop down a donkeycar (with minor hardward enhancements) on any track with a yellow center line and white lane lines and watch the donkeycar improve its track time each time it drives around the track. In its current form, the RLDonkeycar needs a human minder as it drives around the track, and the ability to use reinforcement learning to improve track-times is hardware-limited.

An example early run by the RLDonkecar (filmed by Yana Bezus) can be seen here:
https://drive.google.com/open?id=1TIjjgSAZopL1PxoWHfXPwQQFAPlu5QcM

In a little more detail, the goal is:
* to plop the donkeycar down on any track that it had never seen before.
* slowly self-drive around the track using opencv-based line-following
* Use the opencv line-following to train a Neural Net (NN) like the 5-level NN in the original donkeycar code 
* Use Reinforcement Learning (RL) to increase the thrattle and improve the steering to optimize the path of the donkeycar.
* Push Raspberry Pi's to their limit to see what is possible and minimize the changes from the original donkeycar design.
* Use the same code to run on the Donkey Car simulator. The simulator is especially useful for debugging without requiring access to a real track.

#### Software/Hardware Architecture

The RLDonkeycar has minimal hardward changes from the original donkeycar design: 
* An additional raspberry pi with battery and SD-card
*  a short (6") ethernet cable between the pi's 
* stand-offs to stack the raspberry pi's
![RLDonkeycar](https://github.com/downingbots/RLDonkeycar/RLDonkeycar.jpg)

The Software features:
* on-car real-time control of donkeycar driving based upon OpenCV line-following or execution of Keras Neural Net on other Raspberry Pi 3B+ (aka "Control Pi")
* on-car real-time supervised learning or reinforcement learning (RL) on one Raspberry Pi 3B+ (aka RLPi.) 
* uses Keras RL implementation based on OpenAI's Proximal Policy Optimization (PPO https://arxiv.org/pdf/1707.06347.pdf). A good introduction to PPO is available on YouTube by Arxiv Insights at https://www.youtube.com/watch?v=5P7I-xPq8u8 . Briefly, PPO is a gradient descent algorithm with an actor-critic component and a reward function based upon a history of results. 
* periodic update of the Control Pi's Keras model weights computed at RLPi 
* uses same code to run with the donkey-car simulation

#### Details of the Flow-Control

1. Initially the donkeycar is placed on a center yellow line (optionally dashed)
2. Control Pi uses optical flow to determine the minimal speed that the car can move at.
3. Once moving, the Control Pi uses open-CV and a simple Hough-Transform algorithm to follow the Center Dashed yellow line. The white lines, the white/yellow colors, and land widths are also tracked by identifying "vanishing points".  A human typically follows the car as it makes its way around the track. If the car leaves the track, the human picks up the car and resets it to the center of the track. Mixing things up like driving direction, lighting, rerunning at failure points are recommended.
4. The OpenCV-based throttle and steering values plus the image are sent to the RLPi to perform supervised learning of the Keras model
5. Once sufficient supervised learning has been trained at the RLPi, the weights of the Keras model are sent to the Control Pi and loaded there.
6. The Control Pi now uses the Keras Model to compute the throttle and steering instead of the Open-CV line-following.  Every few messages (configurable), a random bounded change to the computed throttle and steering are made. This is increase the search-space.
7.  The Keras model's throttle and steering values plus the image are sent to the RLPi. The RLPi caches this message and runs OpenCV on the image to compute ther frame's reward as a function of throttle and distance from center. If it is determined that the car has left the track, a mini-batch is completed allowing the total reward to be comuputed by accumulating the frame rewards within the min-batch with a time-decay function. Very small mini-batches are ignored as these are typically caused by manual resetting of the car.  If a Mini-batch reaches a length of 20, then this is acceptable to end the batch because the cause-and-effect of actions 20 images apart is very small in autonomous racing (unlike in video games where there can be sparse large rewards.)
8.  Once a certain number of messages are cached and the mini-batch has completed. the Keras model is trained on these messages and these cached messages are discared.
7. Periodically, the revised weights of the Keras PPO model are sent to the Control Pi and loaded there.
6. The Control Pi now uses the revised Keras Model to compute the throttle and steering instead of the Open-CV line-following.  Goto step 6.

### High-Level Results

#### Original Donkeycar Code
* The immitation/supervised learning in the original donkeycar code could learn how to autonomously drive around the track after training on data obained from as few as 3 manually driven trips around the track and could drive at decent speed. With relatively little training data, the trained car would effectively memorize its way around the course by observing features off the track like posts, cones, chairs, etc. Such over-fitting on features would result in poor performance at events with large numbers of spectators, when such features could be obfuscated. More training in different scenarios and lighting conditions could greatly improve the intelligence, reliability and performance of the donkeycar. The better the human driver, the better the results.

#### OpenCV Donkeycar Line-Following
* The OpenCV line-following is slower than the NN in Frames-per-second (fps) in the original donkeycar code. Even with the simplest line-following agorithm, the donkeycar could only get around the track at the slowest speed the car could drive. 
* The slowest speed was achieved by detecting movement of the car by optical flow. Movement by spectators could fool optical flow into thinking the car was moving. Such false positives can be reduced via manual parameter tuning that may also result in a higher minimum speed.
* Once moving, the Control Pi uses open-CV and a simple Hough-Transform algorithm to follow the Center Dashed yellow line. The white lines, the white/yellow colors, and land widths are also tracked by identifying "vanishing points"

#### Reinforcement Learning

* Unfortunately, early attempts at simple RL algorithms did not do well. Switched to PPO (implemented using Keras), which succeeded in learning line-following instead of overfitting on the training data. The reward function is a function of speed and distance from the middle dashed line of the track as determined by the OpenCV code.
* So, the raspberry pi can store frames in a cache that are then trained in a mini-batch of at most 20 images or until the openCV state computation determines that the car has left the track. [A good introduction to PPO is by Arxiv Insights](https://www.youtube.com/watch?v=5P7I-xPq8u8)
* The Raspberry Pi 3B+ could only run the PPO algorithm at about 1 FPS. At this frame rate, the donkeycar couldn't get around the track much faster than the OpenCV line-following algorithm
* Periodically, update the Control Pi's Keras model weights with those recently computed in real-time at the RLPi.  When the RL weights are updated, the NN would increase the throttle in order to increase the reward. Unfortunately, the battery power would decrease resulting in the throttle decreasing before the next RL weight update. Such real-world issues don't show up in the simulations. The net result in throttle gain was negligble in the current tuning. The throttle gain can be tweaked by changing the reward function, but can easily result in the throttle exceeding what the raspberry pi can handle. Instead of fine-tuning, I intend to follow my "Next Steps."

### Next Steps

* Use one or two NVidia Jetson Nanos and a raspberry pi 3+. The raspberry pi 3+ would provide the wi-fi. Ideally the Jetson Nano will be used to run the Control Pi as parallelism can be exploited during execution the PPO neural net. Improved frame rates should increase the achievable throttle speed.  Using the raspberry pi for training is accepatable as training is done asynchronously and can shed loads by strategically dropping frames if it can't keep up with the Jetson nano.

* To address the battery issues, the next step is to add a rotary encoder and pid based upon the work done by Alan Wells ( https://www.bountysource.com/issues/49439689-rotary-encoder-updates ). This will enable the donkeycar to travel at the desired speed despite a draining battery without making the neural net more complex.

### Acknowledgements

First thanks to Will Roscoe and Adam Conway for the code, donkeycar design, and website (donkeycar.com). Will and Adam built the "hello world" of neural nets and real-world robotics. This repository tries to extend their work to Reinforcement Learning... but the coding isn't nearly as pretty :-) and may never be ready to be integrated into the main donkeycar branch.

Thanks to Carlos Uranga at [DeepRacing](http://deepracing.com/). His efforts have created a space to develop and test autonomous RC cars on a regular basis. As of April 2019, deepracing meets bi-weekly at [TheShop.Build in San Jose]( https://theshop.build/San-Jose )

Thanks to Chris Anderson who hosts [DYIRobocars](https://diyrobocars.com/ ). These quarterly events draw hundreds of spectators and a dozen or more cars. It is fascinating to watch cars that would normally circumnavigate the track fail to handle to the change of scenary due to the number of spectators.

### Using the Reinforcement Learning Code

First, you must create your own donkeycar and follow the instructions on donkeycar.com to play with the unmodified software.

To modify the donkeycar to run the RL version, buy:
* an additional Raspberry Pi B+ with SD card
* an additional battery (fits side-to-side with the other battery under the pi's and 3-D printed plate)
* offsets to stack the raspberry pi's. The Control Pi should be on top and host the camera.
* connect the raspberry pis with a 6" ethernet cable. Tie the cable back so it is ouside the field of view of the camera.

Then download the code from this github repository onto both raspberry pi's. This code is derived from an old fork of the donkeycar github. The code in the d2 directory is derived from the "drive script" created after running the "donkey createcar ~/mycar". The file d2/manage.py has been modified to accept running the following at the ControlPi:
    python manage.py drive --model rl

To run with the Unity-based simulator, use the version of the simulator designed to run with the OpenAI Gym ( https://github.com/tawnkramer/donkey_gym ). Code has been changed to use this version that supports a closed-circuit donkeycar track and the ability to reset the car.  The drive script has been modified to accept running the following at the ControlPi:
    donkey sim --type=rl 

To run the RL code on the RLPi run:
    python RLPi

Then put the car down near the middle of the track and start training in real-time.

#### Based on an Old Donkeycar Github Clone

The donkeycar code was cloned somewhere around March 20, 2018. As my laptop was dealing with incompatible versions of some of the dependencies like keras, some of the code were customized for my laptop. There have been hundreds of commits to the donkeycar github repository since the clone.  Most of the new code to support reinforcement learning was separated into new files beginning with RL in the parts directory. A few integration points like manage.py were also modified. So, it is likely possible to update the code to the latest donkeycar version and integrate the code changes, this has not been done at this point in time. The immediate plans is to work around the main limitations in the current hardware - upgrading to use a Jetson Nano and to support an encoder - at which point, the code might just diverge.  Before such divergence, it makes sense to upload the RL code so it can be run on a donkeycar with minimal changes.
 
#### The RL Code

The RL code is mostly in files starting with RL inside the donkeycar/donkeycar/parts dirctory:

* RLControlPi.py : the logic that changes the mode between the OpenCV code and the RL code. Sends/recieves message to/from the RLPi.

* RL.py: a wrapper around some RL classes to integrate with manage.py  

* RLPi.py: the main code executed at the RLPi. Loops around receiving and caching messages from the ControlPi, processing batches of messages ready for PPO, and sending the updated weights to the ControlPi.

* RLMsg.py: The ZMQ messages. ZMQ must be installed on the pis.

* RLConfig.py: a config file used on both pi's for RL parameters. See below for details.

* RLOpenCV.py: the OpenCV line-following code. HoughLinesP are used to find lines. These lines are bundled together into Left/Center/Right lines. If a vanishing point is found (e.g., during straight-aways on the track), yellow and white colors are computed for the center lines and lane lines respectively.

* RLPPO.py: the PPO algorithm as implemented by others with minimal changes. The code is derived from [the initial framework](https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py) and [the PPO implementation](https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py)
* RLKeras.py: the code create the actor and critic models, to get weights, run fit, randomize output for the Keras models, etc
* RLControlPi2.py : Obsolete. Used for debugging an earlier version of the code. The code not run as a result of a parameter in RLConfig.

Other modified files:
* manage.py:  modified to support the command "python manage.py drive --model rl" and to integrate the RL code

* donkeygymsim.py: supports the newest openGym-compatible Unity siumulator
* tcp_server.py: a tcp socket server (by Tawn Kramer) to talk to the unity donkey simulator

#### RL Model and State Files

The d2/model directory is the repository for state stored between runs.

The PPO models for the actor are stored in: rlpilot_actor  rlpilot_critic

donkeystate.json is a human-readable and editable json file track-specific state and throttle starting points.  Track-specific state includes white/yellow color computations and lane width. The "minThrottle" is the starting throttle to look at when using Optical Flow to determine whether moving. In the current implemenation, the value can change dramatically based upon whether you are running in simulation (set to near-zero), or running with a bad or drained battery.  If you change batteries, you proably need to manually edit the file to reset the minThrottle (to around 30 with my car / battery.)

#### Configuration Parameters in RLConfig.py

Lots of parameters in this file are obsolete, constants, or reasonably tuned. Others may require changes.

When gathering training data for the first time, you want the opencv to gather 10,000 to 15,000 images of good line-following. You don't want to over-fit before moving over to RL. Start with SWITCH_TO_NN set to 5000. Soon after gathering 5000 images, RL will take over and soon go off-track. Restart the processes on both the Pi's and try again until the donkey car begins to somewhat follow lines and some curves. 
* SWITCH_TO_NN = 5000

After the donkey car begins to somewhat follow lines and some curves, change SWITCH_TO_NN to a low value like 10. From then on, RL will kick in right after the car begins to move.
* SWITCH_TO_NN = 10

If possible, start the car near the center yellow line of the track, pointing away from anything the optical flow may consider moving. If two consecutive images from the stationary camera would look the same, set OPTFLOWTHRESH to be low:
* OPTFLOWTHRESH = 0.14
Otherwise, you need to bump up the OPTFLOWTHRESH:
* OPTFLOWTHRESH = 0.9

To run on your laptop with the simulation, you need to change the following params:
* PORT_CONTROLPI = "localhost:5558"
* PORT_RLPI         = 5557

To run on the raspberry pi's, you need to change the following params to use their assigned IP addresses. The values for these variables will look different on the different pi's. For example:
* PORT_CONTROLPI = "10.0.0.4:5558"
* PORT_RLPI         = 5557

To tune PPO processing in batches, tune the values for:
* Q_LEN_THRESH = 200  # Q_LEN_THRESH = Q_LEN_MAX - Q_FIT_BATCH_LEN_THRESH
* Q_LEN_MAX = 250     # size of the cache
* Q_FIT_BATCH_LEN_THRESH = 50  # number of messages used to micro-batch fit()

