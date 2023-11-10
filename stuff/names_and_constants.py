#!/usr/bin/python3

#file to import in the other files

#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################

SIMULATOR_FLAG = True


#====================== VEHICLE PARAMETERS ======================
import numpy as np
from os.path import join, exists, dirname

YAW_GLOBAL_OFFSET = 0.0#np.deg2rad(-5)

START_X = 8.509 # 0.2
START_Y = 2.016# 14.8

GPS_DELAY = 0.45 # [s] delay for gps message to arrive
ENCODER_POS_FREQ = 100.0 # [Hz] frequency of encoder position messages
GPS_FREQ = 10.0 # [Hz] frequency of gps messages
BUFFER_PAST_MEASUREMENTS_LENGTH = int(round(GPS_DELAY * ENCODER_POS_FREQ))

# Vehicle driving parameters
MIN_SPEED = -0.3                    # [m/s]     minimum speed
MAX_SPEED = 2.5                     # [m/s]     maximum speed
MAX_ACCEL = 5.5                     # [m/ss]    maximum accel
MAX_STEER = 28.0                    # [deg]     maximum steering angle

# Vehicle parameters
CM2WB = 0.22                        # [m]       distance from center of mass to wheel base
LENGTH = 0.45  			            # [m]       car body length 
WIDTH = 0.18   			            # [m]       car body width 0.18, 0.2
BACKTOWHEEL = 0.10  		        # [m]       distance of the wheel and the car body
WHEEL_LEN = 0.03  			        # [m]       wheel raduis
WHEEL_WIDTH = 0.03  		        # [m]       wheel thickness
WB = 0.26  			                # [m]       wheelbase

# Camera parameters
FRAME_WIDTH = 320#640           # [pix]     frame width
FRAME_HEIGHT = 240#480          # [pix]     frame height
# position and orientation wrt the car frame
CAM_X = 0.0                 # [m]
CAM_Y = 0.0                 # [m]
CAM_Z = 0.2                 # [m]
CAM_ROLL = 0.0              # [rad]
CAM_PITCH = np.deg2rad(20)  # [rad]
CAM_YAW =  0.0              # [rad]
CAM_FOV = 1.085594795       # [rad]
CAM_F = 1.0                 # []        focal length
# scaling factors
CAM_Sx = 10.0               # [pix/m]
CAM_Sy = 10.0               # [pix/m]
CAM_Ox = 10.0               # [pix]
CAM_Oy = 10.0               # [pix]
CAM_K = np.array([[CAM_F*CAM_Sx,      0.0,            CAM_Ox],
                  [ 0.0,              CAM_F*CAM_Sy,   CAM_Oy],
                  [ 0.0,              0.0,            1.0]])
# Estimator parameters
EST_INIT_X      = 3.0               # [m]
EST_INIT_Y      = 3.0               # [m]
EST_INIT_YAW    = 0.0               # [rad] 

EKF_STEPS_BEFORE_TRUST = 10 #10 is fine, 15 is safer


# BRAIN
#========================= STATES ==========================
START_STATE = 'start_state'
END_STATE = 'end_state'
DOING_NOTHING = 'doing_nothing'
LANE_FOLLOWING = 'lane_following'
APPROACHING_STOP_LINE = 'approaching_stop_line'
INTERSECTION_NAVIGATION = 'intersection_navigation'
TURNING_RIGHT = 'turning_right'
TURNING_LEFT = 'turning_left'
GOING_STRAIGHT = 'going_straight'
TRACKING_LOCAL_PATH = 'tracking_local_path'
ROUNDABOUT_NAVIGATION = 'roundabout_navigation'
WAITING_FOR_PEDESTRIAN = 'waiting_for_pedestrian'
WAITING_FOR_GREEN = 'waiting_for_green'
WAITING_AT_STOPLINE = 'waiting_at_stopline'
OVERTAKING_STATIC_CAR = 'overtaking_static_car'
OVERTAKING_MOVING_CAR = 'overtaking_moving_car'
TAILING_CAR = 'tailing_car'
AVOIDING_ROADBLOCK = 'avoiding_roadblock'
PARKING = 'parking'
CROSSWALK_NAVIGATION = 'crosswalk_navigation'
CLASSIFYING_OBSTACLE = 'classifying_obstacle'
BRAINLESS = 'brainless'

#======================== ROUTINES ==========================
FOLLOW_LANE = 'follow_lane'
DETECT_STOP_LINE = 'detect_stop_line'
SLOW_DOWN = 'slow_down'
ACCELERATE = 'accelerate'
CONTROL_FOR_SIGNS = 'control_for_signs'
CONTROL_FOR_OBSTACLES = 'control_for_obstacles'
UPDATE_STATE = 'update_state'
DRIVE_DESIRED_SPEED = 'drive_desired_speed'

#========================== EVENTS ==========================
EVENT_POINTS_PATH = join(dirname(__file__), 'event_points.txt')
EVENT_TYPES_PATH = join(dirname(__file__), 'event_types.txt')

INTERSECTION_STOP_EVENT = 'intersection_stop_event'
INTERSECTION_TRAFFIC_LIGHT_EVENT = 'intersection_traffic_light_event'
INTERSECTION_PRIORITY_EVENT = 'intersection_priority_event'
JUNCTION_EVENT = 'junction_event'
ROUNDABOUT_EVENT = 'roundabout_event'
CROSSWALK_EVENT = 'crosswalk_event'
PARKING_EVENT = 'parking_event'
END_EVENT = 'end_event'
HIGHWAY_EXIT_EVENT = 'highway_exit_event'

EVENT_TYPES = [INTERSECTION_STOP_EVENT, INTERSECTION_TRAFFIC_LIGHT_EVENT, INTERSECTION_PRIORITY_EVENT,
                JUNCTION_EVENT, ROUNDABOUT_EVENT, CROSSWALK_EVENT, PARKING_EVENT, HIGHWAY_EXIT_EVENT]
                

#======================== ACHIEVEMENTS ========================
#consider adding all the tasks, may be too cumbersome
PARK_ACHIEVED = 'park_achieved'

#======================== CONDITIONS ==========================
CAN_OVERTAKE = 'can_overtake'
HIGHWAY = 'highway'
TRUST_GPS = 'trust_gps'
CAR_ON_PATH = 'car_on_path'
REROUTING = 'rerouting'
BUMPY_ROAD = 'bumpy_road'


#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################


# DETECTION
#PARKING SIGNS
SIGN_POINTS_PATH = join(dirname(__file__), 'sign_points.txt')
SIGN_TYPES_PATH = join(dirname(__file__), 'sign_types.txt')

PARK = 'park'
CLOSED_ROAD = 'closed_road'
HW_EXIT = 'hw_exit'
HW_ENTER = 'hw_enter'
STOP = 'stop'
ROUNDABOUT = 'roundabout'
PRIORITY = 'priority'
CROSSWALK = 'cross_walk'
ONE_WAY = 'one_way'
NO_SIGN = 'NO_sign'
TRAFFIC_LIGHT = 'traffic_light'
SIGN_NAMES = [PARK, CLOSED_ROAD, HW_EXIT, HW_ENTER, STOP, ROUNDABOUT, PRIORITY, CROSSWALK, ONE_WAY, NO_SIGN]
# SIGN_NAMES = [PARK, CLOSED_ROAD, HW_EXIT, HW_ENTER, STOP, ROUNDABOUT, PRIORITY, CROSSWALK, ONE_WAY, NO_SIGN, TRAFFIC_LIGHT]

#obstacles 
CAR = 'car'
PEDESTRIAN = 'pedestrian'
ROADBLOCK = 'roadblock'

#ENVIROMENTAL SERVER
STATIC_CAR_ON_ROAD = 'static_car_on_road'
STATIC_CAR_PARKING = 'static_car_parking'
PEDESTRIAN_ON_CROSSWALK = 'pedestrian_on_crosswalk'
PEDESTRIAN_ON_ROAD = 'pedestrian_on_road'
ROADBLOCK = 'roadblock'
BUMPY_ROAD = 'bumpy_road'

#sempahores
MASTER = 'master'
SLAVE = 'slave'
ANTIMASTER = 'antimaster'
START = 'start'
#semaphore states
GREEN = 2
YELLOW = 1
RED = 0

#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################

#=============== MAP/GRAPH STUFF =======================
BIG = 'big'
MEDIUM = 'medium'
SMALL = 'small'
VERY_SMALL = 'very_small'

MAP_H_M = 13.6 # map height in meters, should match with Simulator/src/models_pkg/track/model.sdf
MAP_HW_RATIO = 1.0/0.66323 # 20.5
MAP_W_M = MAP_H_M*MAP_HW_RATIO # map width in meters
K_BIG = 19897/MAP_H_M
K_MEDIUM = 12011/MAP_H_M
K_SMALL = 8107/MAP_H_M
K_VERYSMALL = 3541/MAP_H_M

#maps paths
MAP_VERY_SMALL_PATH = join(dirname(dirname(__file__)), 'Simulator/src/models_pkg/track/materials/textures/2024_VerySmall.png')
MAP_SMALL_PATH = join(dirname(dirname(__file__)), 'Simulator/src/models_pkg/track/materials/textures/2024_Small.png')
MAP_MEDIUM_PATH = join(dirname(dirname(__file__)), 'Simulator/src/models_pkg/track/materials/textures/2024_Medium.png')
MAP_BIG_PATH = join(dirname(dirname(__file__)), 'Simulator/src/models_pkg/track/materials/textures/2024_Big.png')
assert exists(MAP_VERY_SMALL_PATH), f'very small map file not found: {MAP_VERY_SMALL_PATH}'
assert exists(MAP_SMALL_PATH), f'small map file not found: {MAP_SMALL_PATH}'
assert exists(MAP_MEDIUM_PATH), f'medium map file not found: {MAP_MEDIUM_PATH}'
assert exists(MAP_BIG_PATH), f'big map file not found: {MAP_BIG_PATH}'
#graph paths
GRAPH_PATH = join(dirname(__file__), 'final_graph.graphml')
assert exists(GRAPH_PATH), f'graph file not found: {GRAPH_PATH}'

INT_IN_PATH = join(dirname(__file__), 'int_in.txt')
INT_OUT_PATH = join(dirname(__file__), 'int_out.txt')
INT_MID_PATH = join(dirname(__file__), 'int_mid.txt')
RA_IN_PATH = join(dirname(__file__), 'ra_in.txt')
RA_OUT_PATH = join(dirname(__file__), 'ra_out.txt')
RA_MID_PATH = join(dirname(__file__), 'ra_mid.txt')
HW_PATH = join(dirname(__file__), 'hw.txt')
STOPLINES_PATH = join(dirname(__file__), 'stop_line.txt')


################### PATH PLANING ###################
PATH_STEP_LENGTH = 0.01 # [m] interpolation step length for the path