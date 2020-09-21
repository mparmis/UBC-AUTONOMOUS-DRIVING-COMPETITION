# ENPH 353 - Simulated Autonomous Driving Competition

This project is the final result of the ENPH 353 course competition. ~20 pairs of students built code to move a robot in a simulated environment. The robots had to navigate a course, avoid moving obstacles, and collect information from 'parked cars' in the environment. 

![env](/docs/env.png)

---
## Setup

The following repo was developed on a Ubuntu 18.04 distribution. Additionally, these exact setup steps are untested.

1. Install [ros-melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) and [gym-gazebo](https://github.com/erlerobot/gym-gazebo/blob/master/INSTALL.md#ubuntu-1804). 
2. Install and activate the conda environment from the `env.yml` file.  
3. The code requires the simulated environment to run. This can be found in the [2019F_competition_student](https://github.com/ENPH353/2019F_competition_student) repository for the course. Please setup the repository, source the environment, and run the simulation.  
4. Run `python full_stack/main.py`. The robot in the gym-gazebo environment should now be moving and recording license plates!

## Components 

- Built a Convolutional Neural Network with three different networks to identify and read the license plates on the cars.

- Heavily augmented the training data and did error analysis using confusion matrix to get robust results. The robot was able to read all the license plates correctly at the competition.

- Used classical Computer Vision techniques (OpenCV) to navigate the robot, the algorithm successfully stayed on the road, detected the cross walk and avoided collision with the pedestrians.

- Used filtering and clustering computer vision algorithms to detect and avoid pedestrian and truck obstacles

![robo_view](/docs/robo_view.png)
