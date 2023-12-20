# BFMC 2024
Repo for Bosch Future Mobility Challenge 2024, Dei-Unipd Team

## How to use this repo
### Installations
Initial steps/requirements:
- Operating system: [Ubuntu 20.04](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview)
- Python >= 3.8 (install in virtual environment recommended)
- [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)

Install python packages, open a terminal in this directory (with the virtual environment activated)
and run:
```bash
pip install -r REQUIREMENTS.txt
```

To compile the simulator, in a terminal:
```bash
cd Simulator
bash recompile.sh
```

At the end of the compilation, you should see the following message:
```
To automatically source the sim, add the following lines to your ~/.bashrc file
source /your/repo/dir/bfmc_2024/Simulator/devel/setup.bash
```
with ```/your/repo/dir/``` being the path to the directory where you cloned the repo.

In order to use the simulator, you need to source the workspace, using the line above.
To do so, open a terminal and run:
```bash
source /your/repo/dir/bfmc_2024/Simulator/devel/setup.bash
```

To avoid sourcing the workspace every time, you can add the line to your ~/.bashrc file. (You can do
so by running `nano ~/.bashrc` and adding the line at the end of the file, note: you need to close
and reopen the terminal for the changes to take effect).

## Running the simulator
To run the simulator, open a terminal, source the workspace and run:
```bash
roslaunch sim_pkg car_with_map.launch 
```
This is the simplest launch file, it will start the simulator with the car and the map, without any
visualization. You can try ```car_with_map_vis.launch``` to see the car and the map and
```map_with_all_objects.launch``` to see the map with all the objects (Note1: this will open gazebo,
which is disabled by default in the other 2 launch files to reduce the computational burden. Note2:all the objects are in
the wrong positions, since they are referred to the old map).

### Visualization
To visualize the car and the map, open a terminal, source the workspace and run:
```bash
rosrun example visualizer.py
```
Press R on the image to reset the visualization, and ESC to close it.

### Manual control
To manually control the car, you can use the keyboard. To do so, open a terminal, source the
workspace and run:
```bash
rosrun example control.py
```

## Competition example
To run the competition example, close all terminals, open a new one, source the workspace and run:
```bash
roslaunch sim_pkg car_with_map.launch 
```
In another terminal, wait for the simulation to start, then source the workspace and run:
```bash
python main_brain.py
```
You should see the car going around the map, performing various tasks.

