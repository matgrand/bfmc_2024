from car_sim import CarSim
from path_planning import PathPlanning
from controller import Controller
from detection import Detection
from stuff import *
from time import time, sleep
import numpy as np, os
# os.system('rosservice call /gazebo/reset_simulation')

PP_DA = .15
car = CarSim()
pp = PathPlanning()
cc = Controller(dist_point_ahead=PP_DA)
dd = Detection()

path_nodes = [472,207]
path_nodes = [472,468]
# path_nodes = [472,207,450,322]

pp.generate_path_passing_through(path_nodes)
pp.draw_path()
pp.augment_path()

FPS = 60.0
TARGET_SPEED = .5

#run visualization node
os.system('rosrun example visualizer.py &')

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
    #get point ahead
    pa = get_point_ahead(x,y,pp.path, PP_DA)

    #draw closest point
    cv.circle(pp.map, xy2cv(cp[0],cp[1]), 4, (255,0,255), -1)
    #draw car
    cv.circle(pp.map, xy2cv(x,y), 4, (0,255,0), -1)
    #draw point ahead
    cv.circle(pp.map, xy2cv(pa[0],pa[1]), 4, (0,0,255), -1)

    # heading direction error: e3
    e3 = diff_angle(np.arctan2(pa[1]-y, pa[0]-x), yaw) # using global position
    # _, e3, _ = dd.detect_lane(car.frame) #network estimate

    s, α = cc.get_control(0.0, e3, 0.0, TARGET_SPEED)
    car.drive(s, np.rad2deg(α))
    
    cv.imshow('map', cv.flip(pp.map, 0))
    print(f'press esc to exit.', end='\r')
    key = cv.waitKey(1)
    if key == 27: break
    #wait to mantain the desired FPS
    end_loop = time()
    elapsed_time = end_loop - start_loop
    if elapsed_time < 1.0/FPS: sleep(1.0/FPS - elapsed_time)

cv.destroyAllWindows()
car.stop()
sleep(1.0)
print('Done')