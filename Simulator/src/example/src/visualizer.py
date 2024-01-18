#!/usr/bin/env python3
import rospy, cv2 as cv, numpy as np, os
from sensor_msgs.msg import Image
from utils.msg import pose
from cv_bridge import CvBridge
from time import time
from os.path import join, exists, dirname
import colorsys, sys

# orbslam3 download
# lsdslam 
CANV_WIDTH = 1920//2 # canvas width
IW, IH = 320, 240 # image width, image height
TW, TH = 240, 240 # top image width, top image height

# map img path
#get this repo directory
REPO_PATH = dirname(dirname(dirname(dirname(dirname(os.path.abspath(__file__))))))
sys.path.append(REPO_PATH)
from stuff import LENGTH, WIDTH, BACKTOWHEEL, WB
SIMLATOR_DIR = join(REPO_PATH, 'Simulator')

#check if we need to use the 2024 map or the test map
with open(join(SIMLATOR_DIR, 'src/models_pkg/track/materials/scripts/bfmc_track.material'), 'r') as f:
    MAP_IMG_PATH = join(SIMLATOR_DIR, 'src/models_pkg/track/materials/textures/test_VerySmall.png')
    for line in f:
        if '2024' in line:
            MAP_IMG_PATH = join(SIMLATOR_DIR, 'src/models_pkg/track/materials/textures/2024_VerySmall.png')
            break
assert exists(MAP_IMG_PATH), f'No map image found at {MAP_IMG_PATH}'

PA = 20 #padding around the image
FPS = 60.0 # visualizer fps

DRAW_POSE_EVERY = int(round(1000/2/FPS)) # draw pose twice every frame

RAINBOW_COLORS = 3000 # number of colors in the rainbow, how fast colors change in the path

def rainbow_c(idx, n=RAINBOW_COLORS): # return a rainbow color
    c_float = colorsys.hsv_to_rgb(idx/n, 1.0, 1.0)
    return tuple([int(round(255*x)) for x in c_float])

class Visualizer():
    # ===================================== INIT==========================================
    def __init__(self):
        """
        Creates a bridge for converting the image from Gazebo image intro OpenCv image
        """
        self.bridge = CvBridge()
        self.img = np.zeros((IH, IW, 3), np.uint8)
        self.top_img = np.zeros((TH, TW, 3), np.uint8)
        self.map_img = cv.imread(MAP_IMG_PATH)
        mh, mw = self.map_img.shape[:2] # map height, map width
        canv_w = 3*PA+2*(IW+TW)
        self.mcw = canv_w - 2*PA # map canvas width
        self.mch = int(self.mcw*mh/mw) # map canvas height
        self.m2pix = 15.0/self.mch # meters to pixels
        self.map_img = cv.resize(self.map_img, (self.mcw, self.mch)) # resize map
        self.map_img = cv.flip(self.map_img, 0) # flip map to correctlyplot x,y
        self.clean_map = self.map_img.copy()
        canv_h = 3*PA+2*max(IH,TH)+self.mch
        
        canv_shape = (canv_h, canv_w, 3)
        self.canvas = np.zeros(canv_shape, np.uint8)
        rospy.init_node('visualizer', anonymous=True)
        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.img_callback)
        self.top_img_sub = rospy.Subscriber("/automobile/image_top", Image, self.img_top_callback)
        # create a sub for the pose, with the message type pose
        self.pose_sub = rospy.Subscriber("/automobile/pose", pose, self.pose_callback)
        self. p_idx = 0
        # create a timer for the draw function
        self.timer = rospy.Timer(rospy.Duration(1.0/FPS), self.draw)
        self.c_idx = 0 # color index, to cycle through rainbow colors
        self.prev_time = time()
        rospy.spin()

    def xy2cv(self, x, y): #return x, y as a cv coordinate
        return (int(round(x/self.m2pix)), int(round(y/self.m2pix)))

    def img_callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazebo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.canvas[PA:PA+2*IH,2*(PA+TW):2*(PA+TW+IW)] = cv.resize(self.img, (2*IW, 2*IH))

    def img_top_callback(self, data):
        self.top_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.canvas[PA:PA+2*TH,PA:PA+2*TW] = cv.resize(self.top_img, (2*TW, 2*TH))

    def pose_callback(self, data):
        self.p_idx += 1
        if self.p_idx%DRAW_POSE_EVERY != 0:
            return
        tmp_map = self.map_img.copy()
        x, y, ψ = data.x, data.y, data.psi
        #draw an arrow for the car
        k = 3
        x2, y2 = x+k*LENGTH*np.cos(ψ), y+k*LENGTH*np.sin(ψ)
        cv.arrowedLine(tmp_map, self.xy2cv(x,y), self.xy2cv(x2,y2), (0,0,255), 1)
        # draw a rectangle for the car using LENGTH, WIDTH, BACKTOWHEEL
        corners = np.array([[-BACKTOWHEEL, WIDTH/2],[LENGTH-BACKTOWHEEL, WIDTH/2],[LENGTH-BACKTOWHEEL, -WIDTH/2],[-BACKTOWHEEL, -WIDTH/2]])
        rot_matrix = np.array([[np.cos(ψ), -np.sin(ψ)],[np.sin(ψ), np.cos(ψ)]])
        corners = corners @ rot_matrix.T + np.array([x,y]) # rotate and translate
        corners = (corners/self.m2pix).astype(np.int32) # convert to pixels
        cv.polylines(tmp_map, [corners], True, (0,255,0), 1, cv.LINE_AA) # draw car body
        cv.circle(self.map_img, self.xy2cv(x,y), 1, rainbow_c(self.c_idx), -1) # draw car path
        self.c_idx = (self.c_idx+1)%RAINBOW_COLORS
        self.canvas[2*PA+2*IH:2*PA+2*IH+self.mch,PA:PA+self.mcw] = cv.flip(tmp_map, 0) # draw map

    def draw(self, event=None):
        global CANV_WIDTH
        curr_time = time()
        if curr_time - self.prev_time > 1.0/FPS:
            self.prev_time += 1.0/FPS
            #resize canvas with same ratio and width CANV_WIDTH
            new_h = int(round(CANV_WIDTH*self.canvas.shape[0]/self.canvas.shape[1]))
            cv.imshow('Visualization', cv.resize(self.canvas, (CANV_WIDTH, new_h)))
            key = cv.waitKey(1)
            if key == 27:
                rospy.signal_shutdown('esc pressed')
                cv.destroyAllWindows()
                return
            elif key == ord('='):
                CANV_WIDTH *= 1.1
                CANV_WIDTH = int(min(CANV_WIDTH, 1920)) 
            elif key == ord('-'):
                CANV_WIDTH /= 1.1
                CANV_WIDTH = int(max(CANV_WIDTH, 320))
            elif key == ord('r'): # reset map
                os.system('clear')
                os.system('rosservice call /gazebo/reset_simulation')
                self.map_img = self.clean_map.copy()
            
if __name__ == '__main__':
    try:
        nod = Visualizer()
        cv.destroyAllWindows()
    except rospy.ROSInterruptException:
        rospy.loginfo("Visualizer node terminated.")
        cv.destroyAllWindows()
    except KeyboardInterrupt:
        cv.destroyAllWindows()

