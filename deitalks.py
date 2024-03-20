import cv2 as cv, os, numpy as np
import signal
from time import sleep, time
import rospy
from std_msgs.msg import Float32
from detection import Detection
from controller import Controller
from controllerSP import ControllerSpeed

IMSHOW = False
SPEED = False

DES_SPEED = 0.3 # 0.5 for speed

MAX_SPEED = 0.6
MAX_ANGLE = 23.0

FPS = 30.0

#load camera with opencv
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv.CAP_PROP_FPS, 30)

def stop(): 
    pub_stop.publish(0.0)

def drive(speed, steer):
    speed = np.clip(speed, -MAX_SPEED, MAX_SPEED)
    steer = np.clip(np.rad2deg(steer), -MAX_ANGLE, MAX_ANGLE)
    pub_speed.publish(speed)
    pub_steer.publish(steer)


if __name__ == '__main__':


    #initialize ros node
    rospy.init_node('deitalks', anonymous=False)

    pub_speed = rospy.Publisher('/automobile/command/speed', Float32, queue_size=1)
    pub_steer = rospy.Publisher('/automobile/command/steer', Float32, queue_size=1)
    pub_stop = rospy.Publisher('/automobile/command/stop', Float32, queue_size=1)

    if IMSHOW:
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame',320,240)

    #stop the car with ctrl+c
    def handler(signum, frame):
        print("Exiting ...")
        stop()
        cv.destroyAllWindows()
        sleep(.99)
        exit()
    signal.signal(signal.SIGINT, handler)

    det = Detection()

    if SPEED: ctrlSP = ControllerSpeed(desired_speed=DES_SPEED)
    else: ctrl = Controller()

    sleep(1.0)
    stop()
    print("Starting in 2 seconds ...")
    sleep(2.0)

    while not rospy.is_shutdown():
        start = time()
        ret, frame = cap.read()
        if not ret: continue

        #detection
        if SPEED: le3, _ = det.detect_lane_ahead(frame.copy())
        se2, se3, _ = det.detect_lane(frame.copy())

        #control
        if SPEED: speed, angle = ctrlSP.get_control_speed(se2, se3, le3)
        else: speed, angle = ctrl.get_control(se2, se3, 0, DES_SPEED)

        #drive
        drive(speed, angle)

        if IMSHOW:
            cv.imshow('frame', frame)
            if cv.waitKey(1) == 27: break
        loopt = time() - start
        print(f'fps: {1/loopt:.1f}', end='\r')
        if loopt < 1/FPS: sleep(1/FPS - loopt)

    stop()

    cap.release()
    cv.destroyAllWindows()
    sleep(.4)

    print("Exiting ...")
