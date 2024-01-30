#!/usr/bin/python3

import numpy as np, cv2 as cv
from time import time
from detection.stopline import detect_angle
from os.path import join, dirname, realpath

this_dir = dirname(realpath(__file__))

LANE_KEEPER_PATH = join(this_dir, 'models/lane_keeper_small.onnx')
DISTANCE_POINT_AHEAD = 0.35
LK_CORRECTION = 1.1

LANE_KEEPER_AHEAD_PATH = join(this_dir,'models/lane_keeper_ahead.onnx')
DISTANCE_POINT_AHEAD_AHEAD = 0.6

STOP_LINE_ESTIMATOR_PATH = join(this_dir,'models/stop_line_estimator.onnx')

STOP_LINE_ESTIMATOR_ADV_PATH = join(this_dir,'models/stop_line_estimator_advanced.onnx')
PREDICTION_OFFSET = 0.34

IMG_SIZE = (32,32)  #match with trainer

class Detection:

    #init 
    def __init__(self, advanced_sl=True) -> None:
        self.advanced_sl = advanced_sl

        #lane following
        self.lane_keeper = cv.dnn.readNetFromONNX(LANE_KEEPER_PATH)
        self.lane_cnt = 0
        self.avg_lane_detection_time = 0

        #speed challenge
        self.lane_keeper_ahead = cv.dnn.readNetFromONNX(LANE_KEEPER_AHEAD_PATH)
        self.lane_ahead_cnt = 0
        self.avg_lane_ahead_detection_time = 0

        #stop line detection
        if not advanced_sl: self.stop_line_estimator = cv.dnn.readNetFromONNX(STOP_LINE_ESTIMATOR_PATH) # simple 
        else: self.stop_line_estimator = cv.dnn.readNetFromONNX(STOP_LINE_ESTIMATOR_ADV_PATH) # advanced
        self.est_dist_to_stop_line = 1.0
        self.avg_stop_line_detection_time = 0
        self.stop_line_cnt = 0

    def preprocess(self, frame, cutoff=1/3, faster=False):
        """
        Preprocesses the frame for the lane keeper
        """
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #convert to gray
        frame = frame[int(frame.shape[0]*cutoff):,:] # cutoff
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200) #edge detection
        frame = cv.blur(frame, (3,3), 0) #worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE) #resize
        if faster: blob = cv.dnn.blobFromImage(frame, 1.0, IMG_SIZE, 0, swapRB=True, crop=False) 
        else:
            frame_flip = cv.flip(frame, 1) #flip
            frames = np.stack((frame, frame_flip), axis=0) #stack the 2 images
            blob = cv.dnn.blobFromImages(frames, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        return blob
        

    def detect_lane(self, frame, show_ROI=True, faster=False):
        """
        Estimates:
        - the lateral error wrt the center of the lane (e2), 
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        blob = self.preprocess(frame, cutoff=1/3, faster=faster)
        self.lane_keeper.setInput(blob)
        out = -self.lane_keeper.forward() * LK_CORRECTION #### NOTE: MINUS SIGN IF OLD NET
        if faster: e2, e3 = out[0,0], out[0,1]
        else : e2, e3 = (out[0,0] - out[1,0]) / 2.0, (out[0,1] - out[1,1]) / 2.0

        #calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])
        print(f"est_point_ahead: {est_point_ahead}")
        # print(f"lane_detection: {1000*(time()-start_time):.2f} ms")
        lane_detection_time = 1000*(time()-start_time)
        self.avg_lane_detection_time = (self.avg_lane_detection_time*self.lane_cnt + lane_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        return e2, e3, est_point_ahead

    def detect_lane_ahead(self, frame, show_ROI=True, faster=False):
        """
        Estimates:
        - the lateral error wrt the center of the lane (e2), 
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        blob = self.preprocess(frame, cutoff=1/3, faster=faster)
        self.lane_keeper_ahead.setInput(blob)
        out = self.lane_keeper_ahead.forward() #### NOTE: MINUS SIGN IF OLD NET
        if faster: e3 = out[0][0]
        else : e3 = (out[0][0] - out[1][0]) / 2.0

        #calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])
        
        # print(f"lane_detection: {1000*(time()-start_time):.2f} ms")
        lane_detection_time = 1000*(time()-start_time)
        self.avg_lane_detection_time = (self.avg_lane_detection_time*self.lane_cnt + lane_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        return e3, est_point_ahead

    def detect_stop_line(self, frame, show_ROI=True):
        if self.advanced_sl: return self.detect_stop_line2(frame, show_ROI)
        else: return self.detect_stop_line1(frame, show_ROI)

    def detect_stop_line1(self, frame, show_ROI=True):
        """
        Estimates the distance to the next stop line
        """
        start_time = time()
        blob = self.preprocess(frame, cutoff=2/5, faster=True)
        self.stop_line_estimator.setInput(blob)
        dist = self.stop_line_estimator.forward()[0,0] + PREDICTION_OFFSET
        self.est_dist_to_stop_line = dist
        stop_line_detection_time = 1000*(time()-start_time)
        self.avg_stop_line_detection_time = (self.avg_stop_line_detection_time*self.lane_cnt + stop_line_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        print(f"stop_line_detection dist: {dist:.2f}, in {stop_line_detection_time:.2f} ms")
        return dist

    def detect_stop_line2(self, frame, show_ROI=True):
        """
        Estimates the distance to the next stop line
        """
        start_time = time()
        blob = self.preprocess(frame, cutoff=2/5, faster=True)
        self.stop_line_estimator.setInput(blob)
        output = self.stop_line_estimator.forward()
        stopline_x = dist = output[0][0] + PREDICTION_OFFSET
        stopline_y = output[0][1]
        stopline_angle = output[0][2]
        self.est_dist_to_stop_line = dist
        stop_line_detection_time = 1000*(time()-start_time)
        self.avg_stop_line_detection_time = (self.avg_stop_line_detection_time*self.lane_cnt + stop_line_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        print(f"stop_line_detection dist: {dist:.2f}, in {stop_line_detection_time:.2f} ms")
        return stopline_x, stopline_y, stopline_angle

    #helper functions
    def detect_yaw_stopline(self, frame, show_ROI=False):
        return detect_angle(original_frame=frame, plot=show_ROI)

