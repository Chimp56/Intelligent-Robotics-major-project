# Major Project - Tour Guide Robot

Act as a tour guide

1. Maintain a 2d metric map
2. D* path planning to navigate around dynamic obstacles
3. Determine it's location including the floor (localization)
4. Navigate point A to point B

## Requirements
- Python2
- Ros melodic
- Gazebo
- Rviz

## Instructions

Clone the repository into your catkin workspace src:

```bash
git clone https://github.com/Chimp56/Intelligent-Robotics-major-project ~/catkin_ws/src/tour_guide
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

```bash
chmod +x ~/catkin_ws/src/tour_guide/scripts/*.py
```
## Help



### keyboard movement

https://wiki.ros.org/cmd_vel_mux

### Vincent quick start up

```
git stash
git pull
chmod +x ~/catkin_ws/src/tour_guide/scripts/*.py
source ~/catkin_ws/devel/setup.bash
roslaunch tour_guide mapping.launch


```