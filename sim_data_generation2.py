
import os, signal
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import sleep, time
from stuff import *
from tqdm import tqdm # progress bar

LAPS = 3
imgs = []
locs = []

STEER_NOSIE_STDS_DEG = np.linspace(0, 20, 11)
POSITION_NOISE_STD = np.linspace(0, 0.1, 11)
file_name = 'training_dss/sim_ds_'
if not os.path.exists('training_dss'): os.makedirs('training_dss')

SKIP_ALREADY_EXISTING = False
TARGET_FPS = 30

spath = np.load('sparcs/sparcs_path.npy').T # small path for testing
# spath = np.load('sparcs/sparcs_path_precise.npy').T # precise path for training

spath[:,0:2] += 2.5 #add 2.5 to all x and y values

#concatenate path with itself to make it longer based on LAPS
spath = np.tile(spath, (LAPS,1))

#create yaw sequence
yaws = np.zeros(spath.shape[0])
for i in range(len(spath)-1):
    yaws[i] = np.arctan2(spath[i+1,1]-spath[i,1], spath[i+1,0]-spath[i,0])
yaws[-1] = yaws[-2]
spath = np.vstack((spath.T, yaws)).T
print(f'spath shape: {spath.shape}')
#decimate path
path = spath[::12]

map = cv.imread('Simulator/src/models_pkg/track/materials/textures/test_VerySmall.png')

#initializations
os.system('rosservice call /gazebo/reset_simulation') 
os.system('rosrun example visualizer.py &') #run visualization node

#car placement in simulator
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
ros.wait_for_service('/gazebo/set_model_state')
state_msg = ModelState()
state_msg.model_name = 'automobile'
state_msg.pose.position.x = 0
state_msg.pose.position.y = 0
state_msg.pose.position.z = 0.032939
state_msg.pose.orientation.x = 0
state_msg.pose.orientation.y = 0
state_msg.pose.orientation.z = 0
state_msg.pose.orientation.w = 0
def place_car(x,y,yaw):
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.orientation.z = np.sin(yaw/2)
    state_msg.pose.orientation.w = np.cos(yaw/2)
    set_state = ros.ServiceProxy('/gazebo/set_model_state', SetModelState)
    _ = set_state(state_msg)
    sleep(0.02)

def save_data(imgs,locs, name):
    imgs = np.array(imgs)
    locs = np.array(locs)
    np.savez_compressed(name, imgs=imgs, locs=locs)
    print(f'saved data to {name}')
    # print('not saving, testing')

frame = None
bridge = CvBridge()

ros.init_node('gazebo_move')
def camera_callback(data) -> None:
    """Receive and store camera frame
    :acts on: self.frame
    """        
    global frame, bridge
    frame = bridge.imgmsg_to_cv2(data, "bgr8")

camera_sub = ros.Subscriber('/automobile/image_raw', Image, camera_callback)

for sn_std, pos_std in zip(STEER_NOSIE_STDS_DEG, POSITION_NOISE_STD):
    print(f'noise std: {sn_std}, pos std: {pos_std}')
    ds_name = f'{file_name}{sn_std:.0f}.npz'
    if os.path.exists(ds_name) and SKIP_ALREADY_EXISTING:
        print(f'{ds_name} already exists, skipping')
        continue
    imgs = []
    locs = []

    sleep(0.9)

    LAP_Y_TRSH = 2.54 + 2.5
    START_X = 5.03
    START_Y = LAP_Y_TRSH
    START_YAW = np.deg2rad(90.0) + π

    while frame is None:
        print('waiting for frame')
        sleep(0.01)

    for i in tqdm(range(len(path))):
        loop_start = time()
        xp,yp,yawp = path[i]

        #add noise  
        y_error = np.random.normal(0, pos_std)
        yaw_error = np.random.normal(0, np.deg2rad(sn_std))
        e = np.array([0, y_error])
        R = np.array([[np.cos(yawp), -np.sin(yawp)], [np.sin(yawp), np.cos(yawp)]])
        e = R @ e
        x = xp + e[0]
        y = yp + e[1]
        yaw = yawp + yaw_error

        #place car
        place_car(x,y,yaw)

        locs.append(np.array([x-2.5, y-2.5, yaw]))

        tmp_frame = frame.copy() #copy frame to draw on
        
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (320, 240), interpolation=cv.INTER_AREA)
        imgs.append(img)

        #calculate fps
        loop_time = time() - loop_start
        fps = 1.0 / loop_time
        # print(f'NOISE: {sn_std}')
        # print(f'x: {x:.2f}, y: {y:.2f}, yaw: {np.rad2deg(yaw):.2f}, fps: {fps:.2f}')
        if loop_time < 1/TARGET_FPS:
            sleep(1/TARGET_FPS - loop_time)

    save_data(imgs, locs, ds_name)

    cv.destroyAllWindows()



