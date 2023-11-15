#!/usr/bin/python3
from brain import SIMULATOR_FLAG, SHOW_IMGS
import os, signal
import cv2 as cv
import rospy
import numpy as np
from time import sleep, time
os.system('clear')
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
from shutil import get_terminal_size as gts

map, _ = load_map()

# PARAMETERS
TARGET_FPS = 30.0
sample_time = 0.01 # [s]
DESIRED_SPEED = 0.3# [m/s] #.35
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
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv.CAP_PROP_FPS, 30)

if __name__ == '__main__':

    if SHOW_IMGS:
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame',640,480)
        # show windows
        cv.namedWindow('Path', cv.WINDOW_NORMAL)
        cv.resizeWindow('Path', 600, 600)
        cv.namedWindow('Map', cv.WINDOW_NORMAL)
        cv.resizeWindow('Map', 600, 600)


    # init the car data
    os.system('rosservice call /gazebo/reset_simulation') if SIMULATOR_FLAG else None
    os.system('rosservice call gazebo/unpause_physics') if SIMULATOR_FLAG else None
    # sleep(1.5)
    if SIMULATOR_FLAG: car = CarSim()
    else: car = CarPi()
    sleep(1.5)
    car.encoder_distance = 0.0
    
    #stop the car with ctrl+c
    def handler(signum, frame):
        print("Exiting ...")
        car.stop()
        os.system('rosservice call gazebo/pause_physics') if SIMULATOR_FLAG else None 
        cv.destroyAllWindows()
        sleep(.99)
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
    brain = Brain(car=car, controller=cc, controller_sp=cc_sp, detection=dd, env=env, path_planner=pp, desired_speed=DESIRED_SPEED)

    if SHOW_IMGS:
        map1 = map.copy()
        draw_car(map1, car.pose())
        cv.imshow('Map', cv.flip(map1, 0))
        cv.waitKey(1)
    try:
        car.stop()
        fps_avg = 0.0
        fps_cnt = 0
        while not rospy.is_shutdown():

            loop_start_time = time()
            # os.system('cls' if os.name=='nt' else 'clear') #0.1 sec
            print('\n' * gts().lines, end='')
            print('\033[F' * gts().lines, end='')

            if SHOW_IMGS:
                map1 = map.copy()
                draw_car(map1, car.est_pose(), color=(180,0,0))
                color=(255,0,255) if car.trust_gps else (100,0,100)
                draw_car(map1, car.pose(), color=(0,180,0))
                draw_car(map1, car.est_pose(), color=color)
                if len(brain.pp.path) > 0: 
                    cv.circle(map1, m2pix(brain.pp.path[int(brain.car_dist_on_path*100)]), 10, (150,50,255), 3) 
                # else:
                #     print('No path')
                cv.imshow('Map', cv.flip(map1, 0))
                cv.waitKey(1)

            if not SIMULATOR_FLAG:
                ret, frame = cap.read()
                brain.car.frame = frame
                if not ret:
                    print("No image from camera")
                    frame = np.zeros((240, 320, 3), np.uint8)
                    continue

            # RUN BRAIN
            brain.run()

            ## DEBUG INFO
            print(car)
            print(f'Lane detection time = {dd.avg_lane_detection_time:.1f} [ms]')
            # print(f'Sign detection time = {dd.avg_sign_detection_time:.1f} [ms]')
            print(f'FPS = {fps_avg:.1f},  loop_cnt = {fps_cnt}, capped at {TARGET_FPS}')

            if SHOW_IMGS:
                frame = car.frame.copy()
                # if brain.stop_line_distance_median is not None:
                #     dist = brain.stop_line_distance_median - car.encoder_distance + 0.1
                #     angle_to_stopline = diff_angle(car.yaw, get_yaw_closest_axis(car.yaw))
                #     frame, _ = project_stopline(frame, car, dist, angle_to_stopline, color=(0,200,0))
                cv.imshow('frame', frame)
                if cv.waitKey(1) == 27:
                    cv.destroyAllWindows()
                    break
            
            loop_time = time() - loop_start_time
            fps_avg = (fps_avg * fps_cnt + 1.0 / loop_time) / (fps_cnt + 1)
            fps_cnt += 1
            if loop_time < 1.0 / TARGET_FPS:
                sleep(1.0 / TARGET_FPS - loop_time)



    except KeyboardInterrupt:
        print("Shutting down")
        car.stop()
        sleep(.5)
        cv.destroyAllWindows()
        exit(0)
    except rospy.ROSInterruptException:
        pass
      