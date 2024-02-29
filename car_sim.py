# implement here the Car simulator class
from stuff import *
from car import Car # Car interface
from std_msgs.msg import String # standard ROS messages
from utils.msg import pose # custom messages from the ROS simulator
from sensor_msgs.msg import Image, Range # standard ROS messages
from cv_bridge import CvBridge
import numpy as np, json, collections
from time import time, sleep

class CarSim(Car):
    def __init__(self) -> None:
        super().__init__()

        # ADDITIONAL VARIABLES
        self.sonar_distance_buffer = collections.deque(maxlen=20)
        self.lateral_sonar_distance_buffer = collections.deque(maxlen=20)
        self.t = 0.0
        self.prev_x_true = self.x_true
        self.prev_y_true = self.y_true
        self.prev_timestamp = 0.0
        self.velocity_buffer = collections.deque(maxlen=20)
        self.target_steer = 0.0
        self.curr_steer = 0.0
        self.steer_deque = collections.deque(maxlen=MAX_STEER_SAMPLES)
        self.time_last_steer_command = time()
        self.target_dist = 0.0
        self.arrived_at_dist = True
        self.yaw_true = 0.0
        self.x_buffer = collections.deque(maxlen=5)
        self.y_buffer = collections.deque(maxlen=5)
        self.top_frame = np.zeros((320, 320, 3), np.uint8)

        # PUBLISHERS AND SUBSCRIBERS
        # sub for commands to the simulated vehicle
        self.pub = ros.Publisher('/automobile/command', String, queue_size=1)
        self.steer_updater = ros.Timer(ros.Duration(1/STEER_UPDATE_FREQ), self.steer_update_callback)
        self.drive_dist_updater = ros.Timer(ros.Duration(ENCODER_TIMER), self.drive_distance_callback)
        self.sub_imu = ros.Subscriber('/automobile/pose', pose, self.imu_callback)
        self.reset_rel_pose()
        ros.Timer(ros.Duration(ENCODER_TIMER), self.encoder_distance_callback) #the callback will also do velocity
        self.sub_son = ros.Subscriber('/automobile/sonar1', Range, self.sonar_callback)
        self.sub_lateral_son = ros.Subscriber('/automobile/sonar2', Range, self.lateral_sonar_callback)
        self.bridge = CvBridge()
        self.sub_cam = ros.Subscriber("/automobile/image_raw", Image, self.camera_callback)
        self.sub_top = ros.Subscriber("/automobile/image_top", Image, self.top_camera_callback)

    def camera_callback(self, data:Image) -> None:
        '''Receive and store camera frame
        :acts on: self.frame
        '''        
        self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def top_camera_callback(self, data:Image) -> None:
        '''Receive and store top camera frame
        :acts on: self.top_frame
        '''        
        self.top_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def sonar_callback(self, data:Range) -> None:
        '''Receive and store distance of an obstacle ahead 
        :acts on: self.sonar_distance, self.filtered_sonar_distance
        '''        
        self.sonar_distance = data.range 
        self.sonar_distance_buffer.append(self.sonar_distance)
        self.filtered_sonar_distance = np.median(self.sonar_distance_buffer)

    def lateral_sonar_callback(self, data:Range) -> None:
        '''Receive and store distance of a lateral obstacle
        :acts on: self.lateral_sonar_distance, self.filtered_lateral_sonar_distance
        '''        
        self.lateral_sonar_distance = data.range 
        self.lateral_sonar_distance_buffer.append(self.lateral_sonar_distance)
        self.filtered_lateral_sonar_distance = np.median(self.lateral_sonar_distance_buffer)

    def imu_callback(self, data:pose) -> None:
        '''Receive and store rotation from IMU 
        :acts on: self.roll, self.pitch, self.yaw, self.roll_deg, self.pitch_deg, self.yaw_deg
        :acts on: self.accel_x, self.accel_y, self.accel_z, self.gyrox, self.gyroy, self.gyroz
        '''        
        self.roll = float(data.phi)
        self.roll_deg = np.rad2deg(self.roll)    
        self.pitch = float(data.teta)
        self.pitch_deg = np.rad2deg(self.pitch)
        self.yaw_true = float(data.psi)
        self.yaw = float(data.psi) + self.yaw_offset
        self.yaw_deg = np.rad2deg(self.yaw)
        self.accel_x = float(data.ax)
        self.accel_y = float(data.ay)
        self.accel_z = float(data.az)
        self.gyrox = float(data.vphi)
        self.gyroy = float(data.vteta)
        self.gyroz = float(data.vpsi)
        self.t = float(data.t)
        #true position NOTE: not in real car
        xcom, ycom = float(data.x), float(data.y) #center of mass
        # self.x_true = xcom # center of map reference frame
        # self.y_true = ycom # center of map reference frame
        self.x_true = xcom - np.cos(self.yaw_true)*CM2WB # rear axle reference frame
        self.y_true = ycom - np.sin(self.yaw_true)*CM2WB # rear axle reference frame

    def encoder_distance_callback(self, data) -> None:
        '''Callback when an encoder distance message is received
        :acts on: self.encoder_distance
        :needs to: call update_rel_position
        '''   
        curr_x = self.x_true
        curr_y = self.y_true
        prev_x = self.prev_x_true
        prev_y = self.prev_y_true
        curr_time = self.t
        prev_time = self.prev_timestamp
        delta = np.hypot(curr_x - prev_x, curr_y - prev_y)
        #get the direction of the movement: + or -
        motion_yaw = + np.arctan2(curr_y - prev_y, curr_x - prev_x)
        abs_yaw_diff = np.abs(diff_angle(motion_yaw, self.yaw_true))
        sign = 1.0 if abs_yaw_diff < np.pi/2 else -1.0
        dt = curr_time - prev_time
        if dt > 0.0:
            velocity = (delta * sign) / dt
            self.encoder_velocity_callback(data=velocity) 
            self.encoder_distance += sign*delta
            self.prev_x_true = curr_x
            self.prev_y_true = curr_y
            self.prev_timestamp = curr_time
            self.update_rel_position()

    def steer_update_callback(self, data) -> None:
        # check self.steer_deque is not empty
        if len(self.steer_deque) > 0:
            curr_time = time()
            angle, t = self.steer_deque.popleft()
            if curr_time - t < SERVO_DEAD_TIME_DELAY: #we need to w8, time has not passed yet
                self.steer_deque.appendleft((angle,t))
            else: self.target_steer = angle # enough time is passed, we can update the angle 
        diff = self.target_steer - self.curr_steer
        if diff > 0.0: incr = min(diff, DELTA_ANGLE)
        elif diff < 0.0: incr = max(diff, -DELTA_ANGLE)
        else : return
        self.curr_steer += incr
        self.pub_steer(self.curr_steer)

    def encoder_velocity_callback(self, data) -> None:
        '''Callback when an encoder velocity message is received
        :acts on: self.encoder_velocity
        '''    
        self.encoder_velocity = data
        self.velocity_buffer.append(self.encoder_velocity)
        self.filtered_encoder_velocity = np.median(self.velocity_buffer)

    # COMMAND ACTIONS
    def drive_speed(self, speed=0.0) -> None:
        '''Set the speed of the car
        :acts on: self.speed 
        :param speed: speed of the car [m/s], defaults to 0.0
        '''   
        self.arrived_at_dist = True #ovrride the drive distance        
        self.pub_speed(speed)

    def drive_angle(self, angle=0.0) -> None:
        '''Set the steering angle of the car
        :acts on: self.steer
        :param angle: [deg] desired angle, defaults to 0.0
        '''        
        angle = Car.normalizeSteer(angle)   # normalize steer
        curr_time = time()
        if curr_time - self.time_last_steer_command > 1/MAX_STEER_COMMAND_FREQ: #cannot receive commands too fast
            self.time_last_steer_command = curr_time
            self.steer_deque.append((angle, curr_time))
        else: print('Missed steer command...')

    def drive_distance(self, dist=0.0):
        '''Drive the car a given distance forward or backward
        from the point it has been called and stop there, 
        it uses control in position and not in velocity, 
        used for precise movements
        :param dist: distance to drive, defaults to 0.0
        '''
        self.target_dist = self.encoder_distance + dist
        self.arrived_at_dist = False

    def drive_distance_callback(self, data) -> None:
        if not self.arrived_at_dist:
            dist_error = self.target_dist - self.encoder_distance
            self.pub_speed(min(0.5 * dist_error, 0.2))
            if np.abs(dist_error) < 0.01:
                self.arrived_at_dist = True
                self.drive_speed(0.0)
        
    def stop(self, angle=0.0) -> None:
        '''Hard/Emergency stop the car
        :acts on: self.speed, self.steer
        :param angle: [deg] stop angle, defaults to 0.0
        '''
        self.steer_deque.append((angle, time()))
        self.speed = 0.0
        data = {}
        data['action']        =  '3'
        data['steerAngle']    =  float(angle)
        reference = json.dumps(data)
        self.pub.publish(reference)

    def pub_steer(self, angle):
        data = {}
        data['action']        =  '2'
        data['steerAngle']    =  float(angle)
        reference = json.dumps(data)
        self.pub.publish(reference)

    def pub_speed(self, speed):
        speed = Car.normalizeSpeed(speed) # normalize speed
        self.speed = speed
        data = {}
        data['action']        =  '1'
        data['speed']         =  float(speed)
        reference = json.dumps(data)
        self.pub.publish(reference)