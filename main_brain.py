#!/usr/bin/python3
from brain import SIMULATOR_FLAG, SHOW_IMGS
import numpy as np, cv2 as cv, signal, subprocess as sp
from time import sleep, time

sp.run('clear')
print('Main brain starting...')
# from automobile_data import Automobile_Data
if SIMULATOR_FLAG:
    from car_sim import CarSim
    from stuff import *
else: #PI
    raise NotImplementedError('Not implemented for PI')
    from control.automobile_data_pi import CarPi
    from control.helper_functions import *

from path_planning import PathPlanning
from controller import Controller
from controllerSP import ControllerSpeed
from detection import Detection
from brain import Brain
from environmental_data_simulator import EnvironmentalData

map, _ = load_map()

# PARAMETERS
TARGET_FPS = 30.0
sample_time = 0.01 # [s]
DESIRED_SPEED = 0.4# [m/s] #.35
SP_SPEED = 0.8 # [m/s]
CURVE_SPEED = 0.6# [m/s]
path_step_length = 0.01 # [m]
# CONTROLLER
k1 = 0.0 #0.0 gain error parallel to direction (speed)
k2 = 0.0 #0.0 perpenddicular error gain   #pure paralllel k2 = 10 is very good at remaining in the center of the lane
k3 = 1.0 #1.0 yaw error gain .8 with ff 
k3D = 0.08 #0.08 derivative gain of yaw error
dist_point_ahead= 0.35 #distance of the point ahead in m

#dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain

#load camera with opencv
if not SIMULATOR_FLAG:
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv.CAP_PROP_FPS, 30)

if __name__ == '__main__':
    d = DebugStuff(show_imgs=SHOW_IMGS) #init debug stuff

    if SIMULATOR_FLAG: 
        ros_check_run(launch='car_with_map.launch', map='2024')
        sp.run('rosservice call gazebo/pause_physics', shell=True) 
        sp.run('rosservice call /gazebo/reset_simulation', shell=True) 
        #init car
        car = CarSim()
    else: 
        raise NotImplementedError('Not implemented for PI')
        car = CarPi()
    sleep(.5)
    car.encoder_distance = 0.0

    # sp.run('rosrun example visualizer.py &', shell=True) #run visualization node

    #stop the car with ctrl+c
    def handler(signum, frame):
        print("Exiting...")
        car.stop()
        sp.run('rosservice call gazebo/pause_physics', shell=True) if SIMULATOR_FLAG else None 
        cv.destroyAllWindows()
        sleep(.9)
        exit()
    signal.signal(signal.SIGINT, handler)
    
    # init trajectory
    pp = PathPlanning() 

    # init env
    env = EnvironmentalData(trig_v2v=True, trig_v2x=True, trig_semaphore=True)

    # init controller
    cc = Controller(k1=k1, k2=k2, k3=k3, k3D=k3D, dist_point_ahead=dist_point_ahead, ff=ff_curvature)
    cc_sp = ControllerSpeed(desired_speed=SP_SPEED, curve_speed=CURVE_SPEED)

    #initiliaze all the neural networks for detection and lane following
    dd = Detection()

    #initiliaze the brain
    brain = Brain(car=car, controller=cc, controller_sp=cc_sp, detection=dd, env=env, 
                  path_planner=pp, desired_speed=DESIRED_SPEED, debug_stuff=d)

    car.stop()
    if SIMULATOR_FLAG: sp.run('rosservice call gazebo/unpause_physics', shell=True)
    
    loop_time = 1.0 / TARGET_FPS
    # main loop
    while not ros.is_shutdown():
        loop_start_time = time() #start time of the loop

        if not SIMULATOR_FLAG: #PI
            ret, frame = cap.read()
            brain.car.frame = frame
            if not ret:
                print("No image from camera")
                frame = np.zeros((240, 320, 3), np.uint8)
                continue

        # RUN BRAIN
        brain.run()
            
        ## DEBUG INFO
        if SHOW_IMGS: d.show(brain.car.pose) #show all debug images
        print(f'Lane detection time = {dd.avg_lane_detection_time:.1f} [ms]')
        # print(f'Sign detection time = {dd.avg_sign_detection_time:.1f} [ms]')
        print(f'FPS = {1/loop_time:.1f}, capped at {TARGET_FPS}')
        loop_time = time() - loop_start_time
        sleep(max((0.5/TARGET_FPS) if SIMULATOR_FLAG else 0.0, 1.0 / TARGET_FPS - loop_time))
      