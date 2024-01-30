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
    @property
    def xy(self):
        return np.array([self.x, self.y])
    @property
    def xyp(self):
        return np.array([self.x, self.y, self.ψ])
    @property
    def yaw(self):
        return self.ψ
    def __str__(self):
        return f'x: {self.x}, y: {self.y}, ψ: {self.ψ}'

class DebugStuff:
    def __init__(self, show_imgs=True) -> None:
        self.gmap, self.gframe, self.gtopview = None, None, None
        self.show_imgs = show_imgs
        self.names2imgs = {'gmap':self.gmap, 'gframe':self.gframe, 'gtopview':self.gtopview}
        self.he, self.pa_color = 0.0, (0,0,0) # heading error & color
        if show_imgs:
            cv.namedWindow('gmap', cv.WINDOW_NORMAL)
            # cv.resizeWindow('gmap', 800, 600)
            WINDS = 600
            cv.resizeWindow('gmap', 2*WINDS, 2*WINDS)
            cv.namedWindow('gtopview', cv.WINDOW_NORMAL)
            cv.resizeWindow('gtopview', WINDS, WINDS)
            cv.namedWindow('gframe', cv.WINDOW_NORMAL)
            cv.resizeWindow('gframe', WINDS, int(240/320*WINDS))

    def show(self, cp:Pose=None):
        if self.gmap is not None and self.gframe is not None and self.gtopview is not None:
            if cp is not None:
                color = (0,200,0)
                x,y = m2pix(cp.xy)
                cv.circle(self.gmap, (x,y), 5, color, -1) #draw car position
                cc = get_car_corners(cp) #get car corners
                SHOW_DIST = 0.55 # distance ahead just to show
                pa = np.array([SHOW_DIST*np.cos(self.he),SHOW_DIST*np.sin(self.he)]) # point ahead
                cv.line(self.gframe, project_onto_frame(pa, cam=FRONT_CAM), (self.gframe.shape[1]//2, self.gframe.shape[0]), self.pa_color, 2) #pa frame
                cv.line(self.gtopview, project_onto_frame(pa, cam=TOP_CAM), project_onto_frame(np.array([0,0]), cam=TOP_CAM), self.pa_color, 2) #pa topview
                cv.polylines(self.gtopview, [project_onto_frame(cc, cp, TOP_CAM)], True, color, 1, cv.LINE_AA) #draw car body topview
                # move pa and orig to world frame
                R = np.array([[np.cos(cp.ψ), -np.sin(cp.ψ)],[np.sin(cp.ψ), np.cos(cp.ψ)]])
                pa = pa @ R.T + cp.xy
                #move points for plotting on map
                cn = m2pix(cc - cp.xy) #get car corners in the car frame
                pa = m2pix(pa - cp.xy) #get point ahead in the car frame
                B = 1200 # distance around the car to show
                tmap = np.zeros((2*B,2*B,3), np.uint8)
                mh, mw = self.gmap.shape[:2] #map height and width
                xmin, xmax = max(0, x-B), min(mw, x+B) #x min and max in the tmap
                ymin, ymax = max(0, y-B), min(mh, y+B) #y min and max in the tmap
                dt, db, dr, dl = ymin - (y-B), (y+B) - ymax, xmin - (x-B), (x+B) - xmax #top, bottom, right, left
                xc, yc = x-xmin, y-ymin #car position in the tmap
                tmap = self.gmap[ymin:ymax, xmin:xmax].copy() #copy the map to the tmap
                cn = cn + np.array([xc, yc]) #car corners in the tmap
                pa = pa + np.array([xc, yc]) #point ahead in the tmap
                cv.polylines(tmap, [cn], True, color, 3, cv.LINE_AA) #draw car body
                cv.circle(tmap, (xc, yc), 8, faint_color(color), -1) #draw car position
                cv.line(tmap, (xc, yc), tuple(pa), self.pa_color, 10) #draw point ahead
                tmap = cv.copyMakeBorder(tmap, dt, db, dr, dl, cv.BORDER_CONSTANT, value=(0, 0, 0)) #add borders

            else: tmap = self.gmap
            cv.imshow('gmap', cv.flip(tmap, 0))
        if self.gtopview is not None: cv.imshow('gtopview', self.gtopview)
        if self.gframe is not None: cv.imshow('gframe', self.gframe)
        if cv.waitKey(1) == 27: 
            #pause ros time
            sp.run('rosservice call /gazebo/pause_physics', shell=True, capture_output=True)
            exit(0) #press esc to exit
        

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
def draw_car(map, cp:Pose, color=(0, 255, 0)):
    cn = get_car_corners(cp) #get car corners
    cv.polylines(map, [m2pix(cn)], True, color, 3, cv.LINE_AA) #draw car body

def get_car_corners(cp:Pose):
    assert isinstance(cp, Pose)
    x, y, ψ = cp.xyp
    cl, cw, bw = LENGTH, WIDTH, BACKTOWHEEL #car length, car width, back to wheel
    # cn = np.array([[-CM2WB,cw/2],[cl,cw/2],[cl,-cw/2],[-CM2WB,-cw/2]]) #find 4 corners not rotated cw #COM
    cn = np.array([[-bw,cw/2],[cl-bw,cw/2],[cl-bw,-cw/2],[-bw,-cw/2]]) #find 4 corners not rotated cw # REAR AXIS
    cn = cn @ np.array([[np.cos(ψ), -np.sin(ψ)],[np.sin(ψ), np.cos(ψ)]]).T + np.array([x,y]) #rotate corners and add car position
    return cn

ZOFF = 0.03
FRONT_CAM = {'fov':CAM_FOV,'θ':CAM_PITCH,'x':0.0+CM2WB,'z':0.2+ZOFF, 'w':320, 'h':240}
TOP_CAM = {'fov':CAM_FOV,'θ':1.04,'x':-1+CM2WB,'z':3.5+ZOFF, 'w':480, 'h':480}

def project_onto_frame(points, cp:Pose=None, cam=FRONT_CAM):
    ''' function to project points onto a camera frame, returns the points in pixel coordinates '''
    if cp is not None: assert isinstance(cp, Pose)
    assert isinstance(points, np.ndarray), f'points must be np.ndarray, got {type(points)}'
    assert points.shape[-1] == 2, f'points must be (something,2), got {points.shape}'
    assert points.ndim <= 2, f'points must be (something,2), got {points.shape}'
    θ, xc, zc, w, h = cam['θ'], cam['x'], cam['z'], cam['w'], cam['h']
    pts = points.reshape(-1, 2) #flatten points
    pts = np.concatenate((pts, np.zeros((pts.shape[0],1))), axis=1) # and add z coordinate
    if cp is not None: pts = to_car_frame(pts, cp, 3) # move and rotate points to the car frame
    R, T = np.array([[np.cos(θ), 0, np.sin(θ)], [0, 1, 0], [-np.sin(θ), 0, np.cos(θ)]]), np.array([xc, 0, zc])
    pts = (pts - T) @ R # move and rotate points to the camera frame
    f = 2*np.tan(cam['fov']/2)*h/w# focal length
    ppts = - pts[:,1:] / pts[:,0:1] / f # project points onto the camera frame
    ppts = np.round(h*ppts + np.array([w//2, h//2])).astype(np.int32) # convert to pixel coordinates
    if points.ndim == 1: return ppts[0] #return a single point
    return ppts #return multiple points

def draw_points_on_frame(frame, points, cp:Pose=None, color=(0,255,255), thickness=2, cam=FRONT_CAM):
    ppts = project_onto_frame(points, cp, cam)
    for p in ppts.reshape(-1,2): cv.circle(frame, p, thickness, color, -1)

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
    R = np.array([[np.cos(ψ), -np.sin(ψ)],[np.sin(ψ), np.cos(ψ)]]) #rotation matrix
    out = points_cf @ R #rotate points
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

# car placement in simulator
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
def place_car(x,y,yaw):
    state_msg = ModelState()
    state_msg.model_name = 'automobile'
    state_msg.pose.position.z = 0.032939
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.orientation.z = np.sin(yaw/2)
    state_msg.pose.orientation.w = np.cos(yaw/2)
    set_state = ros.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_state(state_msg)
    sleep(0.02)

def get_stopline_coords(slx, sly=0.0, slyaw=0.0, cp:Pose=None):
    SL_WIDTH = 0.4 #stopline width [m]
    xc, yc, yawc = cp.xyp
    Rc = np.array([[np.cos(yawc), -np.sin(yawc)],[np.sin(yawc), np.cos(yawc)]]) #car rotation matrix
    Rsl = np.array([[np.cos(slyaw), -np.sin(slyaw)],[np.sin(slyaw), np.cos(slyaw)]]) #stopline rotation matrix
    sl = np.array([[0.0, -SL_WIDTH/2.0],[0.0, SL_WIDTH/2.0]]) #stopline in the stopline frame
    sl = sl @ Rsl.T #rotate stopline
    sl = sl + np.array([slx, sly]) #move stopline
    sl = sl @ Rc.T #rotate stopline to the car frame
    sl = sl + np.array([xc, yc]) #move stopline to the car position
    return sl

def faint_color(color, alpha=0.6):
    return tuple([int(alpha*c)%255 for c in color])

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
            
    








