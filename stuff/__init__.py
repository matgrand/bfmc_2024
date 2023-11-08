# this is utils
#!/usr/bin/python3

import cv2 as cv, numpy as np

# common definitions for the project

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

EKF_STEPS_BEFORE_TRUST = 10 #10 is fine, 15 is safe

# HELPER FUNCTIONS

class Pose:
    def __init__(self, x, y, ψ):
        self.x, self.y, self.ψ = x, y, ψ
    def update(self, x, y, ψ):
        self.x, self.y, self.ψ = x, y, ψ
    def __str__(self):
        return f'x: {self.x}, y: {self.y}, ψ: {self.ψ}'

def diff_angle(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

MAP_H_M = 13.6 # map height in meters, should match with Simulator/src/models_pkg/track/model.sdf
MAP_HW_RATIO = 1.0/0.66323 # 20.5
MAP_W_M = MAP_H_M*MAP_HW_RATIO # map width in meters
K_BIG = 19897/MAP_H_M
K_MEDIUM = 12011/MAP_H_M
K_SMALL = 8107/MAP_H_M
K_VERYSMALL = 3541/MAP_H_M

def m2pix(m, k=K_VERYSMALL): #meters to pixel
    return np.int32(m*k)

def pix2m(pix, k=K_VERYSMALL): #pixel to meters
    return pix/k

def xy2cv(x, y, k=K_VERYSMALL):
    x,y = m2pix(x,k), m2pix(y,k)
    return (int(x), int(y))

#function to draw the car on the map
def draw_car(map, cp:Pose, color=(0, 255, 0),  draw_body=True):
    x, y, ψ = cp.x, cp.y, cp.ψ
    cl, cw = LENGTH-CM2WB, WIDTH #car length and width
    #find 4 corners not rotated cw
    corners = np.array([[-CM2WB,cw/2],[cl,cw/2],[cl,-cw/2],[-CM2WB,-cw/2]])
    #rotate corners
    rot_matrix = np.array([[np.cos(ψ), -np.sin(ψ)],[np.sin(ψ), np.cos(ψ)]])
    corners = corners @ rot_matrix.T
    #add car position
    corners = corners + np.array([x,y]) #add car position
    if draw_body: cv.polylines(map, [m2pix(corners)], True, color, 3, cv.LINE_AA) #draw car body
    return map

def project_onto_frame(frame, cp:Pose, points, align_to_car=True, color=(0,255,255), thickness=2):
    #check if its a single point
    single_dim = False
    if points.ndim == 1:
        points = points[np.newaxis,:]
        single_dim = True
    num_points = points.shape[0]
    if points[0].shape == (2,): #if points are 2d, add z coordinate
        points = np.concatenate((points, -CAM_Z*np.ones((num_points,1))), axis=1) 

    #rotate the points around the z axis
    if align_to_car: points_cf = to_car_frame(points, cp, 3)
    else: points_cf = points

    #get points in front of the car
    angles = np.arctan2(points_cf[:,1], points_cf[:,0])
    diff_angles = diff_angle(angles, 0.0) #car yaw
    rel_pos_points = []
    for i, point in enumerate(points):
        if np.abs(diff_angles[i]) < CAM_FOV/2:
            rel_pos_points.append(points_cf[i])
    rel_pos_points = np.array(rel_pos_points)
    if len(rel_pos_points) == 0: return frame, None #no points in front of the car
    #add diffrence com to back wheels
    rel_pos_points = rel_pos_points - np.array([0.18, 0.0, 0.0])
    #rotate the points around the relative y axis, pitch
    beta = -CAM_PITCH
    rot_matrix = np.array([[np.cos(beta), 0, np.sin(beta)],
                            [0, 1, 0],
                            [-np.sin(beta), 0, np.cos(beta)]])
    rotated_points = rel_pos_points @ rot_matrix.T
    #project the points onto the camera frame
    proj_points = np.array([[-p[1]/p[0], -p[2]/p[0]] for p in rotated_points])
    #convert to pixel coordinates
    # proj_points = 490*proj_points + np.array([320, 240]) #640x480
    proj_points = 240*proj_points + np.array([320//2, 240//2]) #320x240
    # draw the points
    for i in range(proj_points.shape[0]):
        p = proj_points[i]
        assert p.shape == (2,), f"projection point has wrong shape: {p.shape}"
        # print(f'p = {p}')
        p1 = (int(round(p[0])), int(round(p[1])))
        # print(f'p = {p}')
        #check if the point is in the frame
        if p1[0] >= 0 and p1[0] < 320 and p1[1] >= 0 and p1[1] < 240:
            try:
                cv.circle(frame, p1, thickness, color, -1)
            except Exception as e:
                print(f'Error drawing point {p}')
                print(p1)
                print(e)
    if single_dim:
        return frame, proj_points[0]
    return frame, proj_points

def to_car_frame(points, cp:Pose, return_size=3):
    #check if its a single point
    single_dim = False
    if points.ndim == 1:
        points = points[np.newaxis,:]
        single_dim = True
    x, y, ψ = cp.x, cp.y, cp.ψ
    if points.shape[1] == 3:
        points_cf = points - np.array([x,y,0])
        rot_matrix = np.array([[np.cos(ψ), -np.sin(ψ), 0],[np.sin(ψ), np.cos(ψ), 0 ], [0,0,1]])
        out = points_cf @ rot_matrix
        if return_size == 2:
            out = out[:,:2]
        assert out.shape[1] == return_size, "wrong size, got {}".format(out.shape[1])
    elif points.shape[1] == 2:
        points_cf = points - np.array([x,y]) 
        rot_matrix = np.array([[np.cos(ψ), -np.sin(ψ)],[np.sin(ψ), np.cos(ψ)]])
        out = points_cf @ rot_matrix
        if return_size == 3:
            out = np.concatenate((out, np.zeros((out.shape[0],1))), axis=1)
        assert out.shape[1] == return_size, "wrong size, got {}".format(out.shape[1])
    else: raise ValueError("points must be (2,), or (3,)")
    if single_dim: return out[0]
    else: return out

def draw_bounding_box(frame, bounding_box, color=(0,0,255)):
    x,y,x2,y2 = bounding_box
    x,y,x2,y2 = round(x), round(y), round(x2), round(y2)
    cv.rectangle(frame, (x,y), (x2,y2), color, 2)
    return frame

def my_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def project_stopline(frame, cp:Pose, stopline_x, stopline_y, car_angle_to_stopline, color=(0,200,0)):
    points = np.zeros((50,2), dtype=np.float32)
    points[:,1] = np.linspace(-0.19, 0.19, 50)
    slp_cf = np.array([stopline_x+0.35, stopline_y])
    rot_matrix = np.array([[np.cos(car_angle_to_stopline), -np.sin(car_angle_to_stopline)], [np.sin(car_angle_to_stopline), np.cos(car_angle_to_stopline)]])
    points = points @ rot_matrix #rotation
    points = points + slp_cf #translation
    frame, proj_points = project_onto_frame(frame, cp, points, align_to_car=False, color=color)
    # frame = cv.polylines(frame, [proj_points], False, color, 2)
    return frame, proj_points

def get_yaw_closest_axis(angle):
    """
    Returns the angle multiple of pi/2 closest to the given angle 
    e.g. returns one of these 4 possible options: [-pi/2, 0, pi/2, pi]
    """
    int_angle = round(angle/(np.pi/2))
    assert int_angle in [-2,-1,0,1,2], f'angle: {int_angle}'
    if int_angle == -2: int_angle = 2
    return int_angle*np.pi/2









