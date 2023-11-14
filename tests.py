from path_planning import PathPlanning
from car_sim import CarSim
from stuff import *
from controller import Controller
from time import time, sleep
import numpy as np, os

pp = PathPlanning()
cc = Controller()
car = CarSim()

path_nodes = [472,207]
pp.generate_path_passing_through(path_nodes)
pp.draw_path()
pp.augment_path()
pp.path = pp.path * 15/13.6 #TODO remove the 15/13.6 when the simulator is fixed

FPS = 30.0
TARGET_SPEED = 0.5

# os.system('rosservice call /gazebo/reset_simulation')

create_window('map')
while True:
    start_loop = time()
    # print(car)

    x,y = car.x_true, car.y_true
    yaw = car.yaw_true

    #get closest point in the path
    p = np.array([x,y], dtype=np.float32)
    cp_idx = np.argmin(np.linalg.norm(pp.path-p, axis=1))
    cp = pp.path[cp_idx]
    if cp_idx == len(pp.path)-1: break #end of path

    #draw closest point
    cv.circle(pp.map, xy2cv(cp[0],cp[1]), 10, (255,0,255), -1)
    #draw car
    cv.circle(pp.map, xy2cv(x,y), 10, (0,255,0), -1)

    e2 = np.linalg.norm(pp.path[cp_idx]-p)
    # e2 = 0.0
    e3 = -diff_angle(yaw, np.arctan2(pp.path[cp_idx,1]-y, pp.path[cp_idx,0]-x))
    # e3 = 0.0

    print(f'e2: {e2:.2f}, e3: {e3:.2f}')

    s, α = cc.get_control(e2, e3, 0.0, TARGET_SPEED)
    car.drive(s, np.rad2deg(α))
    
    cv.imshow('map', cv.flip(pp.map, 0))
    key = cv.waitKey(1)
    if key == 27: break
    #wait to mantain the desired FPS
    end_loop = time()
    elapsed_time = end_loop - start_loop
    if elapsed_time < 1.0/FPS: sleep(1.0/FPS - elapsed_time)

cv.destroyAllWindows()