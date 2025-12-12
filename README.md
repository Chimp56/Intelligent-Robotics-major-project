# Major Project - Tour Guide Robot

Act as a tour guide

1. Maintain a 2d metric map
2. D* path planning to navigate around dynamic obstacles
3. Determine it's location including the floor (localization)
4. Navigate point A to point B

## Requirements
- Python2
- Ros1 melodic
- Gazebo
- Rviz
- TurtleBot2

## Instructions

Clone the repository into your catkin workspace src:

```bash
git clone https://github.com/Chimp56/Intelligent-Robotics-major-project ~/catkin_ws/src/tour_guide
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

save map

```bash
rosrun map_server map_saver -f ~/catkin_ws/src/tour_guide/maps/my_map
```

To control turtlebot using keyboard (this will engage manual mode), open a new terminal and run:

```bash
roslaunch turtlebot_teleop keyboard_teleop.launch
```

Start and stop mapping using commands

```bash
# Start mapping mode
rosservice call /tour_guide/start_mapping

# Stop mapping mode  
rosservice call /tour_guide/stop_mapping
```

Go to idle state
```bash
rosservice call /tour_guide/go_to_idle
```

```bash
rosrun tour_guide waypoint_recorder.py
```

## Exploration Algorithms

The tour guide supports two exploration algorithms:

### Frontier-based Exploration (Default)
Uses frontier detection to explore boundaries between known and unknown space. This is the default algorithm.

```bash
roslaunch tour_guide controller.launch exploration_algorithm:=frontier
```

### RRT-based Exploration
Uses Rapidly Exploring Random Tree (RRT) algorithm to randomly sample waypoints in unexplored areas. Based on the approach from [ros_autonomous_slam](https://github.com/fazildgr8/ros_autonomous_slam).

```bash
roslaunch tour_guide controller.launch exploration_algorithm:=rrt
```

```bash
roslaunch tour_guide controller.launch exploration_algorithm:=rrt_opencv
```

The RRT algorithm:
- Randomly samples waypoints in the exploration region (automatically set from map bounds)
- Prefers waypoints in unknown space with nearby free space for navigation
- Avoids revisiting recently visited areas
- Uses move_base for navigation to waypoints



## Help


https://wiki.ros.org/cmd_vel_mux

### References

https://github.com/fazildgr8/ros_autonomous_slam
https://www.youtube.com/watch?v=Zjb_2krr1Xg

(Potentially) simulate pedestrians
https://github.com/robotics-upo/gazebo_sfm_plugin

### Vincent quick start up

```
git stash
git pull
cd ~/catkin_ws/src/tour_guide/scripts
sed -i 's/\r$//' *.py
chmod +x ~/catkin_ws/src/tour_guide/scripts/*.py
source ~/catkin_ws/devel/setup.bash
roslaunch tour_guide controller.launch exploration_algorithm:=rrt_opencv

roslaunch tour_guide controller.launch

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