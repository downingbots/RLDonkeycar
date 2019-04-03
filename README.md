# RLDonkeycar
## Real-time reinforcement learning of a donkeycar with two raspberry pi's

### Overview

The high-level goal is to plop down a donkeycar (with minor hardware enhancements) on a track and watch the donkeycar improve its lap time each time it drives around the track. In its current form, the RLDonkeycar needs a human minder as it drives around the track, and the ability to use reinforcement learning to improve track-times is hardware-limited.

[![An example early run by the RLDonkeycar (filmed by Yana Bezus)](https://github.com/downingbots/RLDonkeycar/blob/master/RLDonkeycar_vid.png)](https://photos.google.com/share/AF1QipMlxMwhAUZxUmiD4-NLPlO-bo2fk7351VvwHwDEcD6pAKZGtQaQdehy3EjVbIur1A/photo/AF1QipMRmGCoAM7wmMUT726chX_j62kb2KRdNKO9wcFO?key=OElIbW5zQ3B4ZUhFTnpiOHJ5SURxb1RlMnk0cVZ3)

In a little more detail, the goal of the RLDonkeycar software is:
* To plop the donkeycar down on the center of a track, even tracks it had never seen before.  The track should have a dashed or solid yellow center line and white lane lines.
* Slowly self-drive around the track using on-car real-time automated control of donkeycar. The automated driving is based upon OpenCV line-following or execution of a Keras Neural Net on a Raspberry Pi 3B+ (aka the "Control Pi")
* Use the opencv line-following to do on-car real-time imitation learning or reinforcement learning (RL) on the second Raspberry Pi 3B+ (aka RLPi.) The Neural Net (NN) is similar to the small convolutional NN in the original donkeycar code but enhanced to support RL.
* Uses Keras RL implementation based on OpenAI's Proximal Policy Optimization ([PPO paper](https://arxiv.org/pdf/1707.06347.pdf)). A good introduction to PPO is available on [YouTube by Arxiv Insights.](https://www.youtube.com/watch?v=5P7I-xPq8u8) Briefly, PPO is a gradient descent algorithm with an actor-critic component and a reward function based upon a history of results. 
* Periodically update the Control Pi's Keras model weights as computed at RLPi to support real-time learning as the RLDonkeycar drives continuously along the track.
* Push Raspberry Pi's to their limit to see what is possible and minimize the changes from the original donkeycar design. The original donkeycar does remarkably well with its simple convolutional NN on a raspberry pi, but can raspberry pi's do real-time RL?
* Use the same code to run on the enhanced Donkey Car simulator. The simulator is especially useful for debugging on a laptop without requiring access to a real car or track.

#### Software/Hardware Architecture

The RLDonkeycar has minimal hardware changes from the original donkeycar design: 
* an additional Raspberry Pi 3B+ with SD card
* an additional battery (fits side-to-side with the other battery under the pi's and their 3-D printed plate)
* offsets to stack the raspberry pi's. The Control Pi should be on top and host the camera.
* connect the raspberry pis with a 6" Ethernet cable. Tie the cable back so it is outside the field of view of the camera.

When assembled, the RLDonkeycar should look like:
![RLDonkeycar](https://github.com/downingbots/RLDonkeycar/blob/master/RLDonkeycar.jpg)

#### Details of the Flow-Control

1. Initially the donkeycar is placed on or near the center yellow line (optionally dashed)
2. Control Pi uses [optical flow](https://en.wikipedia.org/wiki/Optical_flow) to determine the minimal speed that the car can move at.
3. Once moving, the Control Pi uses open-CV and a simple Hough-Transform algorithm to follow the center dashed yellow line. The white/yellow lines and their colors plus lane widths are dynamically computed by identifying [vanishing points](https://en.wikipedia.org/wiki/Vanishing_point) and then tracking these lines/colors.  A human typically follows the car as it makes its way around the track. If the car leaves the track, the human picks up the car and resets it to the center of the track. Mixing things up like driving direction, lighting, rerunning at failure points are recommended. The slow speed of travel is required due to the frame-per-second (fps) limitations of the raspberry pi despite the simple OpenCV algorithm. To see much more complicated OpenCV algorithms, look at [Udacity student blogs.](https://github.com/anuragreddygv323/CarND-Student-Blogs)
4. The OpenCV-based throttle and steering values plus the image are sent to the RLPi to perform imitation learning of the Keras model.
5. Once sufficient imitation learning has been trained at the RLPi, the weights of the Keras model are sent to the Control Pi and loaded there.
6. The Control Pi now uses the Keras Model to compute the throttle and steering instead of the Open-CV line-following.  Every few images (configurable), a random bounded change to the computed throttle and steering are made. This randomness increases the RL search-space.
7.  The Keras model's throttle and steering values plus the image are sent to the RLPi. The RLPi caches this message and runs OpenCV on the image to compute the frame's reward as a function of throttle and distance from center. If it is determined that the car has left the track, a mini-batch is completed allowing the total reward to be computed by accumulating the frame rewards within the mini-batch with a time-decay function. Very small mini-batches are ignored as these are typically caused by manual resetting of the car.  If a mini-batch reaches a length of 20, then we end the batch because the cause-and-effect of actions 20 images apart is very small in autonomous racing (unlike in video games where there can be sparse large rewards) and because we want to limit the cache size and because we want to facilitate real-time learning.
8.  Once a certain number of message are cached and the mini-batch has completed, the Keras model is trained on the images and data contained in these messages and these cached messages are discarded.
7. Periodically, the revised weights of the Keras PPO model are sent to the Control Pi and loaded there.
6. The Control Pi now uses the revised Keras Model to compute the throttle and steering instead of the Open-CV line-following.  Goto step 6.

### High-Level Results

#### Original Donkeycar Code
The RLDonkeycar code was cloned back in March 2018 and then enhanced with new RL code. The RLDonkeycar can still use the imitation learning provided for the original donkeycar. Just invoke the code as documented on donkeycar.com and use a remote control to drive the car around the track.

The imitation learning in the original donkeycar code could learn how to autonomously drive around the track after training on data obtained from as few as 3 manually driven trips around the track and could drive at decent speed. With relatively little training data, the trained car would effectively memorize its way around the course by observing features off the track like posts, cones, chairs, etc. Such over-fitting on features would result in poor performance at events with large numbers of spectators, when such features could be obfuscated. More training in different scenarios and lighting conditions could greatly improve the intelligence, reliability and performance of the donkeycar. The better the human driver, the better the results.

#### OpenCV Donkeycar Line-Following
Instead of a human using a remote control, the RLDonkeycar is trained by a line-following program using OpenCV. The OpenCV line-following ended up with the following minimal set of features:

* The OpenCV line-following is slower than the NN in Frames-per-second (fps) in the original donkeycar code. Even with the simplest line-following algorithm, the donkeycar could only get around the track at the slowest speed the car could drive. If the speed was increased only slightly, the car would frequently drive off the track or not handle sharp turns. Sharp turns require a higher throttle in order to provide the additional torque required.
* The slowest speed was determined by detecting movement of the car by optical flow. Movement by spectators could fool optical flow into thinking the car was moving. Such false positives can be reduced via manual tuning of the optical flow parameters that require higher thresholds to detect movement but may also result in a higher minimum speed. Periodically, optical flow would be used to ensure that movement was still happening as a constant throttle results in battery drainage and the car slowing down. In its minimalistic design, the donkeycar does not have a wheel encoder to determine speed and only has a camera as sensor input.
* Once moving, the Control Pi uses open-CV and a simple Hough-Transform algorithm to follow the Center Dashed yellow line. The white lines, the white/yellow colors, and lane widths are also tracked by identifying "vanishing points". On some tracks, using gray-scale worked better and other tracks color images were better. Lighting makes a huge difference, and dynamically computing colors and their standard deviation worked best.

#### Reinforcement Learning

By initially doing imitation learning, RL can skip the long awkward phase learning by doing random movements. Random movements are still used during reinforcement learning, but they are bounded so that the RLDonkeycar will have the opportunity to recover from a bad move and still stay on the track, while still allowing the car to from learn better moves.  The RL is done asynchronously on the "RLPi" while the Control Pi drives. This attempt to do incremental real-time learning ended up with the following implementation:

* Unfortunately, early attempts at simple RL algorithms did not do well. After Switching to PPO (implemented using Keras), the RLDonkeycar succeeded in learning line-following instead of overfitting on the training data. The reward function is a function of speed and distance from the middle dashed line of the track as determined by the OpenCV code.
* PPO uses a history of inputs and moves for training.  The RLPi stores frames in a cache.  Rewards are accumulated over a set of consecutive images as long as openCV determines that the car remains on the track (up to a maximum of 20 consecutive images.) Manual resetting a car that goes off the track could take one or two consecutive images so very short runs were ignored for training purposes.
* After the total accumulated reward for an image has been computed, the image is eligible to be dequeued and used to train PPO. Training on a batch of images is more efficient, so the current implementation uses a batch of at least 50 images (configurable to trade off cache size and timeliness).
* The Raspberry Pi 3B+ could only run the PPO algorithm at about 1 FPS. At this frame rate, the donkeycar couldn't get around the track much faster than the OpenCV line-following algorithm. Like the original donkeycar NN, the success of the NN was tied to the speed that it was trained at.
* The PPO seemed to converge to a usable model on a real track in fewer training images than the simulated "generated track" which is relatively featureless.
* Periodically, the Control Pi's Keras model weights would be updated with those recently computed in real-time at the RLPi.  When the RL weights are updated, the NN would typically increase the throttle in order to increase the reward. Unfortunately, the battery power would decrease resulting in the throttle decreasing before the next RL weight update. The net result in throttle gain was negligible using the current tuning (see the THROTTLE_BOOST paramenter descriptions below.) Such real-world issues don't show up in the simulations. The throttle gain can be tweaked by changing the reward function, but can easily result in the throttle exceeding what the raspberry pi can handle. Instead of fine-tuning, I intend to follow my "Next Steps."

### Next Steps

* To address the battery issues, the next step is to add a rotary encoder and pid based upon the work done by Alan Wells. This will enable the donkeycar to travel at the desired speed despite a draining battery without making the neural net more complex.

* Use one or two NVidia Jetson Nanos and a raspberry pi 3b+. The raspberry pi 3b+ would provide the wi-fi. Ideally the Jetson Nano will be used to run the Control Pi as parallelism can be exploited during execution the PPO neural net. Improved frame rates should increase the achievable throttle speed.  Using the raspberry pi for training is acceptable as training is done asynchronously and can shed loads by strategically dropping frames if it can't keep up with the Jetson nano. Alternatively, another level of functional parallelism can be added by using the raspberry pi for low-level control and handling the encoder while using one nano for running the NN for control and the other nano for training. 

* There's more that can be done with the minimalist 2 raspberry pi design. For example, instead of using a NN inspired by the simple convolutional NN used so well by the original donkeycar, a 2-level fully connected NN should be tried with PPO. A simplified NN should result in more FPS and faster throughput. Plenty of fine-tuning remains to improve performance.

### Acknowledgements

First thanks to Will Roscoe and Adam Conway for the initial donkeycar code, donkeycar design, and website (donkeycar.com). Will and Adam built the "hello world" of neural nets and real-world robotics. This repository tries to extend their work to Reinforcement Learning... but the RL coding isn't nearly as pretty :-) and may never be ready to be integrated into the main donkeycar branch.

Thanks to Carlos Uranga at [DeepRacing](http://deepracing.com/). His efforts have created a space to develop and test autonomous RC cars on a regular basis. As of April 2019, DeepRacing meets bi-weekly at [TheShop.Build in San Jose]( https://theshop.build/San-Jose )

Thanks to Chris Anderson who hosts [DIYRobocars](https://diyrobocars.com/ ). These quarterly events draw hundreds of spectators and a dozen or more cars. It is fascinating to watch cars that would normally circumnavigate the track fail to handle to the change of scenery due to the number of spectators.

### Using the Reinforcement Learning Code

First, you must create your own donkeycar and follow the instructions on donkeycar.com to play with the unmodified software. After gaining experience with the the unmodified donkeycar, it's time to enhance the donkey car by buying and assembling the raspberry pi and peripherals as outlined earlier.

Next download the code from this github repository onto both raspberry pi's. This code is derived from an old clone of the donkeycar github. The code in the d2 directory was created by running the "donkey createcar ~/d2" and then modified to support the RLDonkeycar. The file [d2/manage.py](https://github.com/downingbots/RLDonkeycar/blob/master/d2/manage.py) has been modified to accept running the following at the ControlPi:
* python manage.py drive --model rl

To run with the Unity-based simulator, use [the version of the simulator designed to run with the OpenAI Gym.](https://github.com/tawnkramer/donkey_gym) RLDonkeycar code has been changed to use version 18.9 that supports a closed-circuit donkeycar track and the ability to reset the car.  The drive script has been modified to accept running the following at the ControlPi:
* donkey sim --type=rl 

To run the RL code on the RLPi or with the simulator run:
* python RLPi.py

If using a physical car, put the car down near the middle of the track and start training in real-time. 

If running the simulation, start the simulator (preferable version 18.9) and select the screen resolution and "Generated Track" using a command like:
* ./DonkeySimLinux/donkey_sim.x86_64 

The virtual donkeycar should start running automatically. Typically it will restart automatically if it goes off-track. The OpenCV line-follower rarely goes off-track.


#### Based on an Old Donkeycar Github Clone

The donkeycar code was cloned somewhere around March 20, 2018. As my laptop was dealing with incompatible versions of some of the dependencies like keras, some of the code were customized for my laptop. There have been hundreds of commits to the donkeycar github repository since the clone.  Most of the new code in the RLDonkeycar repository to support reinforcement learning was separated into new files with names beginning with RL in the parts directory. A few integration points like manage.py were also modified. So, it is likely possible to update the code to the latest donkeycar version and integrate the code changes, but this has not been done at this point in time. The immediate plans is to work around the main limitations in the current hardware - upgrading to use a Jetson Nano and to support an encoder - at which point, the code might just diverge.  Before such divergence, this repository checkpoints the RL code so it can be run on a donkeycar with minimal changes.
 
#### The RL Code

The RL code is mostly in files starting with RL inside the [donkeycar/donkeycar/parts](https://github.com/downingbots/RLDonkeycar/tree/master/donkeycar/donkeycar/parts) directory:

* [RLControlPi.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/parts/RLControlPi.py): the logic that changes the mode between the OpenCV code and the RL code. Sends/receives message to/from the RLPi.

* [RL.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/parts/RL.py): a wrapper around some RL classes to integrate with manage.py.

* [RLPi.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/parts/RLPi.py): the main code executed at the RLPi. Loops around receiving and caching messages from the ControlPi, processing batches of messages ready for PPO, and sending the updated weights to the ControlPi.

* [RLMsg.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/parts/RLMsg.py): the ZMQ messages sent/received between the two pis. ZMQ must be installed on the pis.

* [RLConfig.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/parts/RLConfig.py): a config file used on both pi's for RL parameters. See below for details.

* [RLOpenCV.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/parts/RLOpenCV.py): the OpenCV line-following code. HoughLinesP are used to find lines. These lines are bundled together into Left/Center/Right lines. If a vanishing point is found (e.g., during straightaways on the track), yellow and white colors are computed for the center lines and lane lines respectively.

* [RLPPO.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/parts/RLPPO.py): the Keras PPO algorithm as implemented by others with minimal changes. The code is derived from [the initial framework](https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py) and [the PPO implementation.](https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py)
* [RLKeras.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/parts/RLKeras.py): the code create the actor and critic models, to get weights, run fit, randomize output for the Keras models, etc.
* [RLControlPi2.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/parts/RLControlPi2.py) : Obsolete. Used for debugging an earlier version of the code. The code not run as a result of a parameter in RLConfig. However, further functional parallelism may be feasible by adding yet another pi and another python process similar to RLControlPi2.py.

Other modified files:
* [manage.py](https://github.com/downingbots/RLDonkeycar/blob/master/d2/manage.py):  modified to support the command "python manage.py drive --model rl" and to integrate the RL code.
* [donkeygymsim.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/management/donkeygymsim.py): supports the newest openGym-compatible Unity simulator.
* [tcp_server.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/management/tcp_server.py): a tcp socket server (by Tawn Kramer) to talk to the unity donkey simulator.

#### RL Model and State Files

The [d2/model](https://github.com/downingbots/RLDonkeycar/tree/master/d2/models) directory is the repository for state stored between runs.

The PPO models for the actor are stored in: rlpilot_actor  rlpilot_critic

donkeystate.json is a human-readable and editable json file storing track-specific state and throttle starting points.  Track-specific state includes white/yellow color computations and lane width. The "minThrottle" is the initial throttle when using Optical Flow to determine whether moving. In the current implementation, the value can change dramatically based upon whether you are running in simulation (set to near-zero), or running with a bad or drained battery.  If you change batteries, you probably need to manually edit the file to reset the minThrottle (to around 30 with my car / battery.)

#### Configuration Parameters in [RLConfig.py](https://github.com/downingbots/RLDonkeycar/blob/master/donkeycar/donkeycar/parts/RLConfig.py)

Lots of parameters in this file are obsolete, constants, or already reasonably tuned. Others may require changes as outlined here.

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

The current reward function for an individual message/image is reward(distance_from_center) * (1 + cfg.THROTTLE_BOOST * throttle). See code in RLKeras.py for details. Feel free to experiment with tweaking THROTTLE_BOOST or replacing the reward function altogether:
* THROTTLE_BOOST = .05  

