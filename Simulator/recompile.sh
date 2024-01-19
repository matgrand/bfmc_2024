#!/bin/bash
curr_path=$(pwd)

# Remove the devel and build directories
rm -rf devel build

#remove the .gazebo directory in the home folder
rm -rf ~/.gazebo
rm -rf ~/.gazebo_gui
rm -rf ~/.ros

#set 2024 map
bash $curr_path/set_2024_map.sh

# Run catkin_make for the utils package
catkin_make --pkg utils
#set 2024 map
bash $curr_path/set_2024_map.sh
# Run catkin_make for the entire workspace
catkin_make

# Append lines to setup.bash
echo export GAZEBO_MODEL_PATH=\""$curr_path/src/models_pkg:\$GAZEBO_MODEL_PATH"\" >> devel/setup.bash
echo export ROS_PACKAGE_PATH=\""$curr_path/src:\$ROS_PACKAGE_PATH"\" >> devel/setup.bash

#you will have to do source devel/setup.bash for every new terminal you open and want to use the
#simulator, or you can add the above two lines to your ~/.bashrc file so that it is automatically 
#sourced everytime you open a new terminal
echo " "
echo "=================================================================================="
echo "To automatically source the sim, add the following lines to your ~/.bashrc file"
echo "source $curr_path/devel/setup.bash"
echo "=================================================================================="
echo " "