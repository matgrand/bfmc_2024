#!/usr/bin/python3
# Functional libraries
import rospy, numpy as np
from time import time
from collections import deque
from stuff import *

class Car():
    def __init__(self) -> None:
        '''Manage flow of data with the car'''        
        # CAR POSITION
        self.x_true = START_X                   # [m]       true:x coordinate (used in simulation and SPARCS)
        self.y_true = START_Y                   # [m]       true:y coordinate (used in simulation and SPARCS)
        self.x_gps = START_X                    # [m]       GPS:x global coordinate
        self.y_gps = START_Y                    # [m]       GPS:y global coordinate
        # IMU           
        self.yaw_offset = YAW_GLOBAL_OFFSET     # [rad]     IMU:yaw offset
        self.roll = 0.0                         # [rad]     IMU:roll angle of the car
        self.roll_deg = 0.0                     # [deg]     IMU:roll angle of the car
        self.pitch = 0.0                        # [rad]     IMU:pitch angle of the car
        self.pitch_deg = 0.0                    # [deg]     IMU:pitch angle of the car
        self.yaw = 0.0                          # [rad]     IMU:yaw angle of the car
        self.yaw_deg = 0.0                      # [deg]     IMU:yaw angle of the car
        self.accel_x = 0.0                      # [m/ss]    IMU:accelx angle of the car
        self.accel_y = 0.0                      # [m/ss]    IMU:accely angle of the car
        self.accel_z = 0.0                      # [m/ss]    IMU:accelz angle of the car
        self.gyrox = 0.0                        # [rad/s]   IMU:gyrox angular vel of the car
        self.gyroy = 0.0                        # [rad/s]   IMU:gyroy angular vel of the car
        self.gyroz = 0.0                        # [rad/s]   IMU:gyroz angular vel of the car
        # ENCODER                           
        self.encoder_velocity = 0.0             # [m/s]     ENC:speed measure of the car from encoder
        self.filtered_encoder_velocity = 0.0    # [m/s]     ENC:filtered speed measure of the car from encoder
        self.encoder_distance = 0.0             # [m]       total absolute distance measured by the encoder, it never get reset
        self.prev_dist = 0.0                    # [m]       previous distance
        self.prev_gps_dist = 0.0
        # CAR POSE ESTIMATION
        self.x_est = 0.0                        # [m]       EST:x EKF estimated global coordinate
        self.y_est = 0.0                        # [m]       EST:y EKF estimated global coordinate
        self.yaw_est = self.yaw_offset          # [rad]     EST:yaw EKF estimated
        self.gps_cnt = 0
        self.trust_gps = True                  # [bool]    EST:variable set to true if the EKF trust the GPS
        self.buffer_gps_positions_still_car = []
        # LOCAL POSITION
        self.x_loc = 0.0                        # [m]       local:x local coordinate
        self.y_loc = 0.0                        # [m]       local:y local coordinate
        self.yaw_loc = 0.0                      # [rad]     local:yaw local
        self.yaw_loc_o = 0.0                    # [rad]     local:yaw origin wrt to global yaw from IMU
        self.dist_loc = 0.0                     # [m]       local:absolute distance, length of local trajectory
        self.dist_loc_o = 0.0                   # [m]       local:absolute distance origin, wrt global encoder distance
        self.last_gps_sample_time = time() 
        self.new_gps_sample_arrived = True
        # SONAR
        self.sonar_distance = 3.0                   # [m]   SONAR: unfiltered distance from the front sonar
        self.filtered_sonar_distance = 3.0          # [m]   SONAR: filtered distance from the front sonar
        self.lateral_sonar_distance = 3.0           # [m]   SONAR: unfiltered distance from the lateral sonar
        self.filtered_lateral_sonar_distance = 3.0  # [m]   SONAR: filtered distance from the lateral sonar
        # CAMERA
        self.frame = np.zeros((FRAME_WIDTH, FRAME_HEIGHT)) # [ndarray] CAM:image of the camera
        # CONTROL ACTION
        self.speed = 0.0            # [m/s]     MOTOR:speed
        self.steer = 0.0            # [rad]     SERVO:steering angle

        self.past_encoder_distances = deque(maxlen=BUFFER_PAST_MEASUREMENTS_LENGTH)
        self.past_yaws = deque(maxlen=BUFFER_PAST_MEASUREMENTS_LENGTH)
        self.yaws_between_updates = deque(maxlen=int(round(ENCODER_POS_FREQ/GPS_FREQ)))

        # I/O interface
        rospy.init_node('car', anonymous=False)
        print('Car node started')

        # SUBSCRIBERS AND PUBLISHERS
        # to be implemented in the specific class
        # they need to refer to the specific callbacks
        pass

    # DATA CALLBACKS
    def camera_callback(self, data) -> None:
        '''Receive and store camera frame
        :acts on: self.frame
        '''        
        pass

    def sonar_callback(self, data) -> None:
        '''Receive and store distance of an obstacle ahead 
        :acts on: self.sonar_distance, self.filtered_sonar_distance
        '''        
        pass

    def lateral_sonar_callback(self, data) -> None:
        '''Receive and store distance of a lateral obstacle 
        :acts on: self.lateral_sonar_distance, self.filtered_lateral_sonar_distance
        '''        
        pass

    def position_callback(self, data) -> None:
        '''Receive and store global coordinates from GPS
        :acts on: self.x_gps, self.y_gps
        :needs to: call update_estimated_state
        '''        
        pass

    def imu_callback(self, data) -> None:
        '''Receive and store rotation from IMU 
        :acts on: self.roll, self.pitch, self.yaw, self.roll_deg, self.pitch_deg, self.yaw_deg
        :acts on: self.accel_x, self.accel_y, self.accel_z, self.gyrox, self.gyroy, self.gyroz
        '''        
        pass

    def encoder_distance_callback(self, data) -> None:
        '''Callback when an encoder distance mself.yessage is received
        :acts on: self.encoder_distance
        :needs to: call update_rel_position
        '''        
        pass

    def update_rel_position(self) -> None:
        '''Update relative pose of the car
        right-hand frame of reference with x aligned with the direction of motion
        '''  
        self.yaw_loc = self.yaw - self.yaw_loc_o
        curr_dist = self.encoder_distance
        #add curr_dist to distance buffer
        self.past_encoder_distances.append(curr_dist)
        #update yaw buffer
        if len(self.past_yaws) > BUFFER_PAST_MEASUREMENTS_LENGTH-1:
            self.yaws_between_updates.append(self.past_yaws.popleft())
        self.past_yaws.append(self.yaw)
        self.dist_loc = np.abs(curr_dist - self.dist_loc_o)
        signed_L = curr_dist - self.prev_dist
        L = np.abs(signed_L)
        dx, dy = L * np.cos(self.yaw_loc), L * np.sin(self.yaw_loc)
        self.x_loc += dx
        self.y_loc += dy
        self.prev_dist = curr_dist       

    def encoder_velocity_callback(self, data) -> None:
        '''Callback when an encoder velocity message is received
        :acts on: self.encoder_velocity
        '''        
        pass 

    # COMMAND ACTIONS
    def drive_speed(self, speed=0.0) -> None:
        '''Set the speed of the car
        :acts on: self.speed 
        :param speed: speed of the car [m/s], defaults to 0.0
        '''       
        raise NotImplementedError()         

    def drive_angle(self, angle=0.0) -> None:
        '''Set the steering angle of the car
        :acts on: self.steer
        :param angle: [deg] desired angle, defaults to 0.0
        '''    
        raise NotImplementedError()    
    
    def drive_distance(self, dist=0.0):
        '''Drive the car a given distance forward or backward
        from the point it has been called and stop there, 
        it uses control in position and not in velocity, 
        used for precise movements
        :param dist: distance to drive, defaults to 0.0
        '''
        raise NotImplementedError()

    def drive(self, speed=0.0, angle=0.0) -> None:
        '''Command a speed and steer angle to the car
        :param speed: [m/s] desired speed, defaults to 0.0
        :param angle: [deg] desired angle, defaults to 0.0
        '''        
        self.drive_speed(speed)
        self.drive_angle(angle)
    
    def stop(self, angle=0.0) -> None:
        '''Hard/Emergency stop the car
        :acts on: self.speed, self.steer
        :param angle: [deg] stop angle, defaults to 0.0
        '''
        raise NotImplementedError()

    def reset_rel_pose(self) -> None:
        '''Set origin of the local frame to the actual pose'''        
        self.x_loc = 0.0
        self.y_loc = 0.0
        self.yaw_loc_o = self.yaw
        self.prev_yaw = self.yaw
        self.yaw_loc = 0.0
        self.prev_yaw_loc = 0.0
        self.dist_loc = 0.0
        self.dist_loc_o = self.encoder_distance

    def est_pose(self):
        return Pose(self.x_est, self.y_est, self.yaw_est)
    
    def pose(self):
        return Pose(self.x_true, self.y_true, self.yaw)
    
    # STATIC METHODS
    def normalizeSpeed(val):       
        return np.clip(val, -MAX_SPEED, MAX_SPEED) 

    def normalizeSteer(val):      
        return np.clip(val, -MAX_STEER, MAX_STEER)
    
    def __str__(self):
        description = '''
{:#^65s} 
(x,y):\t\t\t\t({:.2f},{:.2f})\t\t[m]
{:#^65s} 
(x_est,y_est,yaw_est):\t\t({:.2f},{:.2f},{:.2f})\t[m,m,deg]
{:#^65s} 
(x_loc,y_loc,yaw_loc):\t\t({:.2f},{:.2f},{:.2f})\t[m,m,deg]
dist_loc:\t\t\t{:.2f}\t\t\t[m]
{:#^65s} 
roll, pitch, yaw:\t\t{:.2f}, {:.2f}, {:.2f}\t[deg]
ax, ay, az:\t\t\t{:.2f}, {:.2f}, {:.2f}\t[m/s^2]
wx, wy, wz:\t\t\t{:.2f}, {:.2f}, {:.2f}\t[rad/s]
{:#^65s}
encoder_distance:\t\t{:.3f}\t\t\t[m]
encoder_velocity (filtered):\t{:.2f} ({:.2f})\t\t[m/s]
{:#^65s}
sonar_distance (filtered):\t{:.3f} ({:.3f})\t\t[m]
'''
        return description.format(' POSITION ', self.x_gps, self.y_gps,\
                                    ' ESTIMATION ', self.x_est, self.y_est, np.rad2deg(self.yaw_est),\
                                    ' LOCAL POSITION ', self.x_loc, self.y_loc, np.rad2deg(self.yaw_loc), self.dist_loc,\
                                    ' IMU ', self.roll_deg, self.pitch_deg, self.yaw_deg, self.accel_x, self.accel_y, self.accel_z, self.gyrox, self.gyroy, self.gyroz,\
                                    ' ENCODER ', self.encoder_distance, self.encoder_velocity, self.filtered_encoder_velocity,
                                    ' SONAR ', self.sonar_distance, self.filtered_sonar_distance)
                                    



    










