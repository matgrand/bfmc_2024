#!/usr/bin/python3
from stuff import MyRandomGenerator, diff_angle, ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion
import collections, os, signal
from std_msgs.msg import Float32, Bool
import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep

# from helper_functions import *
from vicon import Vicon

# VICON settings
VICON_TOPIC = '/vicon/bfmc_car/bfmc_car'
YAW_OFFSET = np.deg2rad(-90.0)
X0 = -1.3425
Y0 = -2.35

# general settings
DIST_AHEAD = .35 #pure pursuit controller distance ahead
TARGET_FPS = 30.0

LAPS = 9
LAP_Y_TRSH = 2.54

## CONTROLLER
class Controller():
    def __init__(self, k=1.0, noise_std=np.deg2rad(20)):
        #controller paramters
        self.k = k
        self.cnt = 0
        self.noise = 0.0
    def get_control(self, alpha, dist_point_ahead):
        d = dist_point_ahead#POINT_AHEAD_CM/100.0 #distance point ahead, matched with lane_detection
        delta = np.arctan((2*0.26*np.sin(alpha))/d)
        return  - self.k * delta
K = 1.0 #1.0 yaw error gain .8 with ff 
CONTROLLER = Controller(K)



REVERSED_PATH = False #if True the path is Anti-clockwise
STEER_NOISE_STD = np.deg2rad(18.0) # [rad] noise in the steering angle



STEER_FRAME_CHAMGE_MEAN = 10 #avg frames after which the steering noise is changed
STEER_FRAME_CHAMGE_STD = 8 #frames max "deviation"
STEER_NOISE = MyRandomGenerator(0.0, STEER_NOISE_STD, STEER_FRAME_CHAMGE_MEAN, STEER_FRAME_CHAMGE_STD)
DESIRED_SPEED = .3#0.15# [m/s]
SPEED_NOISE_STD = 0.0  #[m/s] noise in the speed
SPEED_FRAME_CHAMGE_MEAN = 30 #avg frames after which the speed noise is changed
SPEED_FRAME_CHAMGE_STD = 20 #frames max "deviation"
SPEED_NOISE = MyRandomGenerator(-SPEED_NOISE_STD, SPEED_NOISE_STD, SPEED_FRAME_CHAMGE_MEAN, SPEED_FRAME_CHAMGE_STD, np.random.uniform)

VI = Vicon() #intialize vicon class, with publishers and subscribers

if __name__ == '__main__':
    
    path = np.load('sparcs_path.npy').T # load path from file
    if REVERSED_PATH:
        path = np.flip(path, axis = 0)
    print(f'path: {path.shape}')

    ros.init_node('pc_controller', anonymous=False) #initialize ros node

    def handler(signum, frame):
        print("Exiting ...")
        VI.stop()
        sleep(.3)
        exit()
    signal.signal(signal.SIGINT, handler)

    lap = 0
    prev_y = VI.y

    pc_locs = []

    while not ros.is_shutdown():
        loop_start_time = time()

        #get heading error
        p = np.array([VI.x,VI.y]).T #current position of the car
        min_index = np.argmin(np.linalg.norm( path-p,axis=1)) #index of clostest point on path
        pa =  path[(min_index + int(100*DIST_AHEAD)) % len( path)] #point ahead
        yaw_ref = np.arctan2(pa[1]-p[1],pa[0]-p[0]) #yaw reference in world frame
        he = diff_angle(yaw_ref, VI.yaw) #heading error

        pc_locs.append(np.array([VI.x,VI.y,VI.yaw]))

        #controller
        steer_angle = CONTROLLER.get_control(he, DIST_AHEAD) #get steering angle

        #noise
        steer_angle = steer_angle + STEER_NOISE.get_noise() #add noise to steer angle
        speed = DESIRED_SPEED + SPEED_NOISE.get_noise() #add noise to speed
        print(f'x: {VI.x} \ny: {VI.y} \nyaw: {np.rad2deg(VI.yaw)}\nhe: {np.rad2deg(he)}\nlap: {lap+1}/{LAPS}\n')

        #actuation
        VI.drive(speed, steer_angle)

        #laps
        curr_y = p[1]
        if (prev_y > LAP_Y_TRSH and curr_y < LAP_Y_TRSH and not REVERSED_PATH) or (prev_y < LAP_Y_TRSH and curr_y > LAP_Y_TRSH and REVERSED_PATH): #count lap
            lap+=1
            if lap>=LAPS:
                # saving locations
                np.savez_compressed('new_tests_locs/RENAME.npz', locs=np.array(pc_locs))
                #stop the car
                print('Stopping')
                VI.stop()
                sleep(.3)
                exit()


        prev_y = p[1]

        # wait for next loop, to get the desired fps
        loop_time = time() - loop_start_time
        if loop_time < 1/TARGET_FPS:
            sleep(1/TARGET_FPS - loop_time)








        