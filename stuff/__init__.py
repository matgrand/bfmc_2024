# this is utils
#!/usr/bin/
from stuff.names_and_constants import *
import cv2 as cv, numpy as np, networkx, subprocess as sp
from os.path import join, exists, dirname
from time import sleep, time

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

def m2pix(m, k=K_VERYSMALL): #meters to pixel
    return np.int32(m*k)

def pix2m(pix, k=K_VERYSMALL): #pixel to meters
    return pix/k

def xy2cv(x, y, k=K_VERYSMALL):
    x,y = m2pix(x,k), m2pix(y,k)
    return (int(x), int(y))

def p2cv(p, k=K_VERYSMALL): # point in the shape of np.array([x,y]) gets converted to cv2 point
    assert p.shape == (2,), f'p shape: {p.shape}'
    return xy2cv(p[0], p[1], k)

def get_point_ahead(x,y,path,d=PP_DA):
    """Returns the point ahead of the car on the path"""
    p = np.array([x,y], dtype=np.float32) #car position
    cp_idx = np.argmin(np.linalg.norm(path-p, axis=1)) #closest point index
    if cp_idx == len(path)-1: return path[-1] #end of path
    samples_ahead = int(d/PATH_STEP_LENGTH) #number of samples ahead in the path
    short_path = path[cp_idx:cp_idx+samples_ahead*5] #get a short path ahead for efficiency
    pa_idx = np.argmin(np.abs(np.linalg.norm(short_path-p, axis=1) - d)) #point ahead index
    pa = short_path[pa_idx] #point ahead
    return pa

def load_graph() -> networkx.Graph:
    from os.path import join, exists, dirname
    graph_path = join(dirname(__file__), 'final_graph.graphml')
    assert exists(graph_path), f'graph file not found: {graph_path}'
    graph = networkx.read_graphml(graph_path)
    return graph

def load_map(map_name=VERY_SMALL):
    if map_name == BIG: mp, k = MAP_BIG_PATH, K_BIG
    elif map_name == MEDIUM: mp, k = MAP_MEDIUM_PATH, K_MEDIUM
    elif map_name == SMALL: mp, k = MAP_SMALL_PATH, K_SMALL
    elif map_name == VERY_SMALL: mp, k = MAP_VERY_SMALL_PATH, K_VERYSMALL
    else: raise ValueError(f'invalid map name: {map_name}, possible values: BIG, MEDIUM, SMALL, VERY_SMALL')
    m = cv.flip(cv.imread(mp), 0) #flip image vertically
    return m, k

def load_nodes(nodes_path): 
    assert nodes_path in ALL_NODES_PATHS, f'invalid nodes path: {nodes_path}'
    return np.loadtxt(nodes_path, dtype=str)

def create_window(name, size=(1920,1080)):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, size[0], size[1])
    cv.moveWindow(name, 2560, 0)

#function to draw the car on the map
def draw_car(map, cp:Pose, color=(0, 255, 0),  draw_body=True):
    assert isinstance(cp, Pose)
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
    assert isinstance(cp, Pose)
    #check if its a single point
    assert points.shape[-1] == 2, f'points must be (2,), got {points.shape}'
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
    for i, _ in enumerate(points):
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

FRONT_CAM = {'fov':CAM_FOV,'θ':CAM_PITCH,'x':0.18,'z':CAM_Z, 'w':320, 'h':240}
TOP_CAM = {'fov':CAM_FOV,'θ':-0.028,'x':1-0.18,'z':3.5, 'w':240, 'h':240}

def project_onto_frame2(cp:Pose, points, align_to_car=True, cam=FRONT_CAM):
    '''function to project points onto the camera frame, returns the points in pixel coordinates'''
    assert isinstance(cp, Pose)
    assert isinstance(points, np.ndarray), f'points must be np.ndarray, got {type(points)}'
    assert points.shape[-1] == 2, f'points must be (something,2), got {points.shape}'
    assert points.ndim <= 2, f'points must be (something,2), got {points.shape}'
    x,y,ψ = cp.x, cp.y, cp.ψ #car position and yaw
    fov, θ, xc, zc, w, h = cam['fov'], cam['θ'], cam['x'], cam['z'], cam['w'], cam['h']
    psh, pndim = points.shape, points.ndim #points shape and dimension
    pts = points.reshape(-1,2) #flatten points
    # convert to 3d points
    pts = np.concatenate((pts, np.zeros((pts.shape[0],1))), axis=1)
    if align_to_car: pts = to_car_frame(pts, cp, 3) # move and rotate points to the car frame
    # rotate the points around the relative y axis, pitch
    R = np.array([[np.cos(θ), 0, np.sin(θ)], [0, 1, 0], [-np.sin(θ), 0, np.cos(θ)]])
    pts = pts @ R.T
    # project the points onto the camera frame
    ppts = np.array([[-p[1]/p[0], -p[2]/p[0]] for p in pts]) 
    # convert to pixel coordinates
    ppts = h*ppts + np.array([w//2, h//2]) 
    # convert to integer
    ppts = np.round(ppts).astype(np.int32)
    # check if the points are in the camera frame
    ppts = ppts[(ppts[:,0] >= 0) & (ppts[:,0] < w) & (ppts[:,1] >= 0) & (ppts[:,1] < h)]
    if len(ppts) == 0: return None #no points in front of the car
    if pndim == 1: return ppts[0] #return a single point
    return ppts #return multiple points



def project_onto_frame(frame, cp:Pose, points, align_to_car=True, color=(0,255,255), thickness=2):
    proj_points = project_onto_frame2(cp, points, align_to_car)
    if proj_points is None: return frame, None #no points in front of the car
    print(f'0 proj_points.shape = {proj_points.shape}\nproj_points = {proj_points}')
    #reshape the points
    psh = proj_points.shape
    ppts = proj_points.reshape(-1,2)
    # draw the points
    for p in ppts:
        assert p.shape == (2,), f"projection point has wrong shape: {p.shape}"
        # print(f'p = {p}')
        p1 = (int(round(p[0])), int(round(p[1])))
        # print(f'p = {p}')
        #check if the point is in the frame
        assert p1[0] >= 0 and p1[0] < 320 and p1[1] >= 0 and p1[1] < 240, f'point not in frame: {p1}'
        cv.circle(frame, p1, thickness, color, -1)
    print(f'1 proj_points.shape = {psh}\nproj_points = {proj_points}')
    return frame, proj_points.reshape(psh)

def to_car_frame(points, cp:Pose, return_size=3):
    assert isinstance(cp, Pose)
    assert isinstance(points, np.ndarray), f'points must be np.ndarray, got {type(points)}'
    assert points.shape[-1] == 2 or points.shape[-1] == 3, f'points must be (something,2 or 3), got {points.shape}'
    assert return_size == 2 or return_size == 3, f'return_size must be 2 or 3, got {return_size}'
    psh = points.shape #points shape
    points = points.reshape(-1, psh[-1]) #flatten points
    x, y, ψ = cp.x, cp.y, cp.ψ #car position and yaw
    if psh[-1] == 3: # if points are 3d, remove z coordinate
        zs = points[:,-1]
        points = points[:,:-1]
    else: zs = np.zeros(points.shape[0])
    points_cf = points - np.array([x,y]) #move points to the origin of the car frame
    rot_matrix = np.array([[np.cos(ψ), -np.sin(ψ)],[np.sin(ψ), np.cos(ψ)]]) #rotation matrix
    out = points_cf @ rot_matrix #rotate points
    if return_size == 3: out = np.concatenate((out, zs[:,np.newaxis]), axis=1) #add z coordinate
    return out.reshape(psh)

def draw_bounding_box(frame, bounding_box, color=(0,0,255)):
    x,y,x2,y2 = bounding_box
    x,y,x2,y2 = round(x), round(y), round(x2), round(y2)
    cv.rectangle(frame, (x,y), (x2,y2), color, 2)
    return frame

def my_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def get_curvature(points, v_des=0.0):
    diff = points[1:] - points[:-1]
    distances = np.linalg.norm(diff, axis=1)
    d = np.mean(distances)
    angles = np.arctan2(diff[:,1], diff[:,0]) 
    alphas = diff_angle(angles[1:], angles[:-1])
    alpha = np.mean(alphas)
    curv = (2*np.sin(alpha*0.5)) / d
    COMPENSATION_FACTOR = 0.855072 
    return curv * COMPENSATION_FACTOR

def get_yaw_closest_axis(α):
    """
    Returns the α multiple of pi/2 closest to the given α 
    e.g. returns one of these 4 possible options: [-pi/2, 0, pi/2, pi]
    """
    int_angle = round(α/(np.pi/2))
    assert int_angle in [-2,-1,0,1,2], f'α: {int_angle}'
    if int_angle == -2: int_angle = 2
    return int_angle*np.pi/2

def ros_check_run(launch='car_with_map.launch', map='2024'): 
    ''' Checks if the ros launch file is running, if not, it starts it 
    launch: name of the launch file (e.g. 'car_with_map.launch', see folder:
    Simulator/src/sim_pkg/launch/)
    map: map choice: options: '2024', 'test'
    '''
    assert map in ['2024', 'test'], f'invalid map: {map}'
    assert exists(join(dirname(dirname(__file__)), 'Simulator/src/sim_pkg/launch', launch)), f'launch file not found: {launch}'
    if 'ERROR' in sp.run('rostopic list', shell=True, capture_output=True).stderr.decode('utf-8'):
        print(f'Ros master not running, starting it...')
        #set the map and run ros master with the launch file
        cmd =f'bash {join(dirname(dirname(__file__)), "Simulator", f"set_{map}_map.sh")} && roslaunch sim_pkg {launch}'
        sp.Popen(['gnome-terminal', '--', 'bash', '-c', cmd])
        while True: #wait for the ros master to start
            print('waiting for ros master to start...', end='\r')
            if '/automobile/command' in sp.run('rostopic list', shell=True, capture_output=True).stdout.decode('utf-8'): 
                print('ros master started!                     ')
                break
            sleep(0.3)

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
            
    








