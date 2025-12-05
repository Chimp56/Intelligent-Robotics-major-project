# Debugging Auto Explore

If the kobuki is not moving when state is MAPPING, follow these debugging steps:

## 1. Check ROS Topics

Open a terminal and check if topics are being published:

```bash
# Check if state is MAPPING
rostopic echo /tour_guide/state

# Check if map is being published
rostopic echo /map -n 1

# Check if robot pose is available
rostopic echo /odom -n 1

# Check if move_base is running
rostopic list | grep move_base
```

## 2. Check ROS Logs

Look at the auto_explore node output for debug messages:

```bash
# The node should output messages like:
# - "Auto Explore: In MAPPING state, handling exploration..."
# - "Auto Explore: Map size: ..."
# - "Auto Explore: Robot pose: ..."
# - "Auto Explore: Searching for frontiers..."
# - "Auto Explore: Found X frontier clusters"
```

## 3. Common Issues

### Issue: "Waiting for map and robot pose..."
**Solution:** 
- Make sure gmapping is running (should be launched automatically)
- Wait a few seconds for the map to initialize
- Check `/map` topic: `rostopic hz /map`

### Issue: "move_base action server not available"
**Solution:**
- Move_base should be launched automatically in controller.launch
- If not, manually launch: `roslaunch turtlebot_navigation move_base.launch.xml`
- Check if move_base is running: `rostopic list | grep move_base`

### Issue: "No frontiers found"
**Possible causes:**
- Map is fully explored (all unknown cells are gone)
- Map hasn't been built yet (wait for gmapping to create map)
- All frontiers are too small (check MIN_FRONTIER_SIZE parameter)
- Check map in RViz to see if there are unknown areas

### Issue: "Could not select a valid frontier"
**Solution:**
- All frontiers may have been visited
- Try resetting visited_frontiers or restart the node
- Check if frontiers are being found but filtered out

## 4. Manual Testing

### Test move_base directly:
```bash
# Send a test goal
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped "{
  header: {frame_id: 'map'},
  pose: {position: {x: 1.0, y: 1.0}, orientation: {w: 1.0}}
}"
```

### Test map subscription:
```bash
# Check map info
rostopic echo /map -n 1 | grep -A 5 "info:"
```

### Test robot pose:
```bash
# Check transform from map to base_link
rosrun tf tf_echo map base_link
```

## 5. Enable More Verbose Logging

The code now includes extensive logging. Check the terminal where auto_explore is running for:
- State changes
- Map statistics (unknown/free/occupied counts)
- Frontier detection results
- Goal status updates

## 6. Quick Fixes

If nothing works, try:
1. Restart the launch file
2. Check that all nodes are running: `rosnode list`
3. Verify state is actually MAPPING: `rostopic echo /tour_guide/state`
4. Check for errors in all node outputs

