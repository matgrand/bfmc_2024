<img src="https://github.com/ECC-BFMC/Simulator/blob/main/Picture1.png" width=30% height=30%>

# BFMC Simulator Project

The project contains the entire Gazebo simulator. 
- It can also be used to get a better overview of how the competition is environment looks like
- It can be used in to develop the vehicle state machine
- It can be used to simulate the path planning
- It can be used to set-up a configuration procedure for the real track
- Not suggested for image processing
- Try not to fall in the "continuous simulator developing" trap

To run the simulation, install ros noetic on ubuntu 20.04; clone the repository; cd to Simulator directory and run the recompile.sh with the command "bash recompile.sh". 
Source devel/setup.bash with "source devel/setup.bash" in every terminal you want to use the sim. 
To run a simulation you can run for example: "roslaunch sim_pkg map_with_car.launch" 


Tips on how to install and work on it, can be found in the 
## The documentation is available in details here:
[Documentation](https://bosch-future-mobility-challenge-documentation.readthedocs-hosted.com/data/simulator.html)
