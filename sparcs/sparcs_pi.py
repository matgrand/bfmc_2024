#!/usr/bin/python3

import rospy
import numpy as np
import cv2 as cv
from time import sleep, time

from vicon import Vicon

TARGET_FPS = 30.0
REVERSED_PATH = False

LAPS = 3
LAP_Y_TRSH = 2.54

imgs = []
locs = []

if __name__ == '__main__':

    #initialize ros node
    rospy.init_node('pi_logger', anonymous=False) #initialize ros node
    VI = Vicon() #intialize vicon class, with publishers and subscribers

    #camera
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320) #320
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240) #240
    cap.set(cv.CAP_PROP_FPS, 30)

    lap = 0
    prev_y = VI.y

    #wait until we get an image
    while not cap.read()[0]:
        pass

    while not rospy.is_shutdown():
        loop_start_time = time()

        if lap >=1:
            #get heading error
            locs.append(np.array([VI.x,VI.y,VI.yaw]))

            #camera
            ret, img = cap.read()
            #convert to grayscale
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #create numpy 8 bit array for image
            img = np.array(img, dtype=np.uint8)
            imgs.append(img)
        else:
            print('Not saving anything first lap')

        #laps
        curr_y = VI.y
        if (prev_y > LAP_Y_TRSH and curr_y < LAP_Y_TRSH and not REVERSED_PATH) or (prev_y < LAP_Y_TRSH and curr_y > LAP_Y_TRSH and REVERSED_PATH): #count lap
            lap+=1
            if lap>=LAPS:
                print('Saving images and locations...')
                assert len(imgs) == len(locs), f'Number of images and locations do not match ({len(imgs)} != {len(locs)})'
                imgs = np.array(imgs)
                locs = np.array(locs)
                np.savez_compressed('sparcs_log.npz', imgs=imgs, locs=locs)
                sleep(.3)
                print('Saved')
                exit()
        prev_y = curr_y

        # wait for next loop, to get the desired fps
        loop_time = time() - loop_start_time
        if loop_time < 1/TARGET_FPS:
            sleep(1/TARGET_FPS - loop_time)
        print(f'x: {VI.x:.2f} - y: {VI.y:.2f} - yaw: {np.rad2deg(VI.yaw):.2f} -- lap: {lap+1}/{LAPS} - fps: {1/loop_time:.1f}')

    