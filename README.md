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
cd ~/catkin_ws/src/tour_guide/scripts
sed -i 's/\r$//' *.py
chmod +x ~/catkin_ws/src/tour_guide/scripts/*.py
```
## Help



### keyboard movement

To control turtlebot using keyboard, open a new terminal and run:
```bash
roslaunch turtlebot_teleop keyboard_teleop.launch
```

https://wiki.ros.org/cmd_vel_mux

### Vincent quick start up

```
git stash
git pull
cd ~/catkin_ws/src/tour_guide/scripts
sed -i 's/\r$//' *.py
chmod +x ~/catkin_ws/src/tour_guide/scripts/*.py
source ~/catkin_ws/devel/setup.bash
roslaunch tour_guide autonomous_map.launch
```

### Testing Basic Movement

To test the controller with simple circular movement:

```bash
# With simulation
roslaunch tour_guide test_circle.launch

# Minimal (real robot, no simulation)
roslaunch tour_guide test_circle_minimal.launch
```

The robot will move in a circle. Adjust speed via parameters:
- `linear_speed`: Forward speed (default: 0.15 m/s)
- `angular_speed`: Rotation speed (default: 0.3 rad/s)

```