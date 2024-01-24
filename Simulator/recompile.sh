#!/bin/bash
# This script will recompile the simulator and source the devel/setup.bash file

#check if ros is installed and with the correct verison (noetic) and if it's sourced
#ros installed?
if ! [ -x "$(command -v rosversion)" ]; then 
  echo 'Error: ROS is not installed.' >&2
  exit 1
fi
#ros should be sourced
if [ -z "$ROS_DISTRO" ]; then
  echo 'Error: ROS is not sourced. Remeber to add "source /opt/ros/noetic/setup.bash" to your ~/.bashrc file' >&2
  exit 1
fi

curr_wd=$(pwd) #get the current working directory
#check if the script is being run from the correct directory (Simulator)
if [ ${curr_wd##*/} != "Simulator" ]; then
  echo 'Error: This script should be run from the Simulator directory.' >&2
  exit 1
fi


# Remove the devel and build directories
rm -rf devel build

#remove the .gazebo directory in the home folder
rm -rf ~/.gazebo
rm -rf ~/.gazebo_gui
rm -rf ~/.ros

#set 2024 map
bash $curr_wd/set_2024_map.sh

# Run catkin_make for the utils package
catkin_make --pkg utils
#set 2024 map
bash $curr_wd/set_2024_map.sh
# Run catkin_make for the entire workspace
catkin_make

# Append lines to setup.bash
echo export GAZEBO_MODEL_PATH=\""$curr_wd/src/models_pkg:\$GAZEBO_MODEL_PATH"\" >> devel/setup.bash
echo export ROS_PACKAGE_PATH=\""$curr_wd/src:\$ROS_PACKAGE_PATH"\" >> devel/setup.bash

#automatically source the devel/setup.bash file by adding the following lines to 

#you will have to do source devel/setup.bash for every new terminal you open and want to use the
#simulator, or you can add the above two lines to your ~/.bashrc file so that it is automatically 
#sourced everytime you open a new terminal
echo " "
echo "================================================================================"
echo "To automatically source the sim, add the following lines to your ~/.bashrc file"
echo "source $curr_wd/devel/setup.bash"
echo "================================================================================"
echo " "

#check if the line is already present in the bashrc file
if grep -Fxq "source $curr_wd/devel/setup.bash" ~/.bashrc
then
    echo "The line is already present in the ~/.bashrc file"
else
    # ask the user if they want to add the line to the bashrc file
    read -p "Do you want to add the line to the ~/.bashrc file? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "source $curr_wd/devel/setup.bash" >> ~/.bashrc
        echo "The line has been added to the ~/.bashrc file"
    fi
fi
