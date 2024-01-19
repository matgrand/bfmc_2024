#!/usr/bin/python3

# HELPER FUNCTIONS
from math import dist
import numpy as np
import cv2 as cv

def diff_angle(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

#const_simple = 196.5
#const_med = 14164/15.0
const_verysmall = 3541/15.0
M_R2L = np.array([[ 1.0, 0.0], [ 0.0, -1.0]])
T_R2L = np.array([0, 15.0])

def mL2pix(ml): #meters to pixel (left frame)
    return np.int32(ml*const_verysmall)

def pix2mL(pix): #pixel to meters (left frame)
    return pix/const_verysmall

def mL2mR(m): # meters left frame to meters right frame
   return m @ M_R2L + T_R2L

def mR2mL(m): #meters right frame to meters left frame
    return (m - T_R2L) @ M_R2L 

def mR2pix(mr): # meters to pixel (right frame), return directly a cv point
    if mr.size == 2:
        pix =  mL2pix(mR2mL(mr))
        return (pix[0], pix[1])
    else:
        return mL2pix(mR2mL(mr))
    
def pix2mR(pix): #pixel to meters (right frame)
    return mL2mR(pix2mL(pix))

#function to draw the car on the map
def draw_car(map, x, y, angle, color=(0, 255, 0),  draw_body=True):
    car_length = 0.45-0.22 #m
    car_width = 0.2 #m
    #match angle with map frame of reference
    # angle = yaw2world(angle)
    #find 4 corners not rotated car_width
    corners = np.array([[-0.22, car_width/2],
                        [car_length, car_width/2],
                        [car_length, -car_width/2],
                        [-0.22, -car_width/2]])
    #rotate corners
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    corners = corners @ rot_matrix.T
    #add car position
    corners = corners + np.array([x,y])
    #draw body
    if draw_body:
        cv.polylines(map, [mR2pix(corners)], True, color, 3, cv.LINE_AA) 
    return map

def draw_angle(frame, angle, color=(0,0,255)):
    l = frame.shape[0]/3
    angle = angle*3
    #get bottom center of the frame
    xc,yc = frame.shape[1]/2, frame.shape[0]
    if -np.pi/2 < angle < np.pi/2:
        x = xc - l*np.sin(angle)
        y = yc - l*np.cos(angle)
        assert x > 0 and y > 0, f'x: {x}, y: {y}'
        frame = cv.line(frame, (int(xc), int(yc)), (int(x), int(y)), color, 2)
    return frame

def draw_seq_points_ahead(frame, car, seq_relative_angles, path_ahead_distances, color=(0,0,255)):
    assert len(seq_relative_angles) == len(path_ahead_distances)
    #create numpy array of the heading errors adding a zero at the beginning
    rel_angles = np.array(seq_relative_angles)
    # add a zero at the beginning
    dists = np.concatenate(([0], path_ahead_distances))
    rel_dists = dists[1:] - dists[:-1]
    pts = np.zeros((len(rel_angles),2))
    p = np.array([0,0])
    for i in range(len(rel_angles)):
        p = p + rel_dists[i]*np.array([np.cos(rel_angles[i]), np.sin(rel_angles[i])])
        pts[i] = p
    #draw the points
    frame, proj = project_onto_frame(frame, car, pts, align_to_car=False, color=color)
    if proj is not None and len(proj)>1:
        for i in range(len(proj)-1):
            p1 = (int(proj[i][0]), int(proj[i][1]))
            p2 = (int(proj[i+1][0]), int(proj[i+1][1]))
            frame = cv.line(frame, p1, p2, color, 2)
    return frame

def project_onto_frame(frame, car, points, align_to_car=True, color=(0,255,255), thickness=2):
    #check if its a single point
    single_dim = False
    if points.ndim == 1:
        points = points[np.newaxis,:]
        single_dim = True
    num_points = points.shape[0]
    #check shape
    if points[0].shape == (2,):
        #add 0 Z component
        points = np.concatenate((points, -car.CAM_Z*np.ones((num_points,1))), axis=1)

    #rotate the points around the z axis
    if align_to_car:
        points_cf = to_car_frame(points, car, 3)
    else:
        points_cf = points

    angles = np.arctan2(points_cf[:,1], points_cf[:,0])
    #calculate angle differences
    diff_angles = diff_angle(angles, 0.0) #car.yaw
    #get points in front of the car
    rel_pos_points = []
    for i, point in enumerate(points):
        if np.abs(diff_angles[i]) < car.CAM_FOV/2:
            # front_points.append(point)
            rel_pos_points.append(points_cf[i])

    #convert to numpy
    rel_pos_points = np.array(rel_pos_points)

    if len(rel_pos_points) == 0:
        # return frame, proj
        return frame, None

    #add diffrence com to back wheels
    rel_pos_points = rel_pos_points - np.array([0.18, 0.0, 0.0])

    #rotate the points around the relative y axis, pitch
    beta = -car.CAM_PITCH
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

def to_car_frame(points, car, return_size=3):
    #check if its a single point
    single_dim = False
    if points.ndim == 1:
        points = points[np.newaxis,:]
        single_dim = True
    gamma = car.yaw
    if points.shape[1] == 3:
        points_cf = points - np.array([car.x_true, car.y_true, 0])
        rot_matrix = np.array([[np.cos(gamma), -np.sin(gamma), 0],[np.sin(gamma), np.cos(gamma), 0 ], [0,0,1]])
        out = points_cf @ rot_matrix
        if return_size == 2:
            out = out[:,:2]
        assert out.shape[1] == return_size, "wrong size, got {}".format(out.shape[1])
    elif points.shape[1] == 2:
        points_cf = points - np.array([car.x_true, car.y_true]) 
        rot_matrix = np.array([[np.cos(gamma), -np.sin(gamma)],[np.sin(gamma), np.cos(gamma)]])
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

def get_curvature(points, v_des=0.0):
    #OLD VERSION
    # # calculate curvature 
    # local_traj = points
    # #get length
    # path_length = 0
    # for i in range(len(points)-1):
    #     x1,y1 = points[i]
    #     x2,y2 = points[i+1]
    #     path_length += np.hypot(x2-x1,y2-y1) 
    # #time
    # tot_time = path_length / v_des
    # local_time = np.linspace(0, tot_time, len(local_traj))
    # dx_dt = np.gradient(local_traj[:,0], local_time)
    # dy_dt = np.gradient(local_traj[:,1], local_time)
    # dp_dt = np.gradient(local_traj, local_time, axis=0)
    # v = np.linalg.norm(dp_dt, axis=1)
    # ddx_dt = np.gradient(dx_dt, local_time)
    # ddy_dt = np.gradient(dy_dt, local_time)
    # curv = (dx_dt*ddy_dt-dy_dt*ddx_dt) / np.power(v,1.5)
    # avg_curv = np.mean(curv)
    # return avg_curv
    diff = points[1:] - points[:-1]
    distances = np.linalg.norm(diff, axis=1)
    d = np.mean(distances)
    angles = np.arctan2(diff[:,1], diff[:,0]) 
    alphas = diff_angle(angles[1:], angles[:-1])
    alpha = np.mean(alphas)
    curv = (2*np.sin(alpha*0.5)) / d
    COMPENSATION_FACTOR = 0.855072 
    return curv * COMPENSATION_FACTOR

# semi random generator 
class MyRandomGenerator:
    def __init__(self, value_mean, value_std, frame_change_mean, frame_change_std, rand_func=np.random.normal) -> None:
        """
        Note: if using np.ranodm.normal, the mean and std are the mean and std of the distribution
        if using np.random.uniform, the mean and std are the lower and upper bound of the uniform distribution
        """
        self.cnt = 0
        self.noise_value = 0.0
        self.next_reset = 0
        self.random_func = rand_func

        self.value_mean = value_mean
        self.value_std = value_std
        self.frame_change_mean = frame_change_mean
        self.frame_change_std = frame_change_std
    
    def get_noise(self):
        if self.cnt == self.next_reset:
            self.cnt = 0
            self.noise_value = self.random_func(self.value_mean, self.value_std)
            self.next_reset = np.random.randint(self.frame_change_mean - self.frame_change_std, self.frame_change_mean + self.frame_change_std)
        self.cnt += 1
        return self.noise_value

#detection functions
def wrap_detection(output_data):
    class_ids = []
    confidences = []
    boxes = []
    rows = output_data.shape[0]
    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.3:
            classes_scores = row[5:]
            _, _, _, max_indx = cv.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w))
                top = int((y - 0.5 * h))
                width = int(w)
                height = int(h)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 
    result_class_ids = []
    result_confidences = []
    result_boxes = []
    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])
    return result_class_ids, result_confidences, result_boxes

def my_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def project_curvature(frame, car, curv):
    color = (0,255,0) 
    start_from = 0.13
    d_ahead = 0.4 # [m]
    num_points = 20
    #multiply by constant, to be tuned
    curv = curv * 1.0 #30.
    # curv = curv * 30 #30.
    #get radius from curvature
    r = 1. / curv
    print("r: {}".format(r))
    x = np.linspace(start_from, start_from + d_ahead, num_points)
    y = - np.sqrt(r**2 - x**2) * np.sign(curv) + r
    #stack x and y into one array
    points = np.stack((x,y), axis=1)
    #project points onto car frame, note: they are already inn car frame
    frame, proj_points = project_onto_frame(frame=frame, car=car, points=points, align_to_car=False, color=color)
    #draw a line connecting the points
    if proj_points is not None:
        #convert proj to int32
        proj_points = proj_points.astype(np.int32)
        # print(proj_points)
        cv.polylines(frame, [proj_points], False, color, 2)
    return r

def project_stopline(frame, car, stopline_x, stopline_y, car_angle_to_stopline, color=(0,200,0)):
    points = np.zeros((50,2), dtype=np.float32)
    points[:,1] = np.linspace(-0.19, 0.19, 50)

    slp_cf = np.array([stopline_x+0.35, stopline_y])

    rot_matrix = np.array([[np.cos(car_angle_to_stopline), -np.sin(car_angle_to_stopline)], [np.sin(car_angle_to_stopline), np.cos(car_angle_to_stopline)]])
    points = points @ rot_matrix #rotation
    points = points + slp_cf #translation

    frame, proj_points = project_onto_frame(frame=frame, car=car, points=points, align_to_car=False, color=color)
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

def get_heading_error(x,y,yaw,path,dist_ahead):
    #check path shape
    assert path.shape[1] == 2, f'path.shape: {path.shape}'
    p = np.array([x,y]).T #current position of the car
    min_index = np.argmin(np.linalg.norm( path-p,axis=1)) #index of clostest point on path
    pa =  path[(min_index + int(100*dist_ahead)) % len( path)] #point ahead
    yaw_ref = np.arctan2(pa[1]-p[1],pa[0]-p[0]) #yaw reference in world frame
    he = diff_angle(yaw_ref, yaw) #heading error
    return he







