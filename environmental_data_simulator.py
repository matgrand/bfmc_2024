#!/usr/bin/python3
from stuff import *
from brain import SIMULATOR_FLAG
import numpy as np, json

# Functional libraries
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

from std_msgs.msg import Byte
from utils.msg import environmental
from utils.msg import vehicles

if SIMULATOR_FLAG:
    from stuff import *
else:
    raise NotImplementedError('Not implemented for PI')
    from control.helper_functions import *

VEHICLE_CLOSE_RADIUS = 0.5  # [m]

SEMAPHORE_POSITIONS = {
                ANTIMASTER:     np.array([2.01,4.16]), 
                SLAVE:      np.array([3.69,4.54]), 
                MASTER: np.array([3.06,3.5]),
                START:      np.array([0.84,0.64])
                } 



class EnvironmentalData():
    def __init__(self,
                 trig_v2v = False,
                 trig_v2x = False,
                 trig_semaphore = False
                ) -> None:
        """Manage flow of data with the environmental server

        :param trig_control: trigger on commands, defaults to True
        :type trig_control: bool, optional
        """        
        # ==================== VARIABLES ====================
        # VEHICLE-TO-VEHICLE (V2V)
        self.other_vehicles = {}
        # VEHICLE-TO-EVERYTHING (V2X)
        self.obstacle_list = []
        self.obstacle_map = {
                                STOP:                   1,
                                PRIORITY:               2,
                                PARK:                   3,
                                CROSSWALK:              4,
                                HW_ENTER:               5,
                                HW_EXIT:                6,
                                ROUNDABOUT:             7,
                                ONE_WAY:                8,
                                TRAFFIC_LIGHT:          9,
                                STATIC_CAR_ON_ROAD:     10,
                                STATIC_CAR_PARKING:     11,
                                PEDESTRIAN_ON_CROSSWALK:12,
                                PEDESTRIAN_ON_ROAD:     13,
                                ROADBLOCK:              14,
                                BUMPY_ROAD:             15
                            }
        # SEMAPHORE
        self.semaphore_states = {MASTER:0, SLAVE:0, ANTIMASTER:0, START:0}# can be changed to 1,2,3,4
                
        # ==================== SUBSCRIBERS AND PUBLISHERS ====================
        # VEHICLE-TO-VEHICLE (V2V)
        if trig_v2v:
            self.sub_v2v = ros.Subscriber('/automobile/vehicles', vehicles, self.v2v_callback)
        # VEHICLE-TO-EVERYTHING (V2X)
        if trig_v2x:
            self.pub_v2x = ros.Publisher('/automobile/environment', environmental, queue_size=1)
        # SEMAPHORE
        if trig_semaphore:
            if SIMULATOR_FLAG:
                self.sub_semaphoremaster= ros.Subscriber("/automobile/trafficlight/master", Byte, self.semaphore_master_callback)
                self.sub_semaphoreslave = ros.Subscriber("/automobile/trafficlight/slave", Byte, self.semaphore_slave_callback)
                self.sub_semaphoreantimaster = ros.Subscriber("/automobile/trafficlight/antimaster", Byte, self.semaphore_antimaster_callback)
                self.sub_semaphorestart = ros.Subscriber("/automobile/trafficlight/start", Byte, self.semaphore_start_callback)
            else:
                self.sub_semaphoremaster= ros.Subscriber("/automobile/semaphore/master", Byte, self.semaphore_master_callback)
                self.sub_semaphoreslave = ros.Subscriber("/automobile/semaphore/slave", Byte, self.semaphore_slave_callback)
                self.sub_semaphoreantimaster = ros.Subscriber("/automobile/semaphore/antimaster", Byte, self.semaphore_antimaster_callback)
                self.sub_semaphorestart = ros.Subscriber("/automobile/semaphore/start", Byte, self.semaphore_start_callback)

        # I/O interface
        # ros.init_node('environmental_data', anonymous=False)

    # V2V CALLBACKS AND FUNCTIONS
    def v2v_callback(self, data) -> None:
        """Receive and store positions of other moving vehicles
        :acts on: self.other_vehicles
        """
        ID = data.ID
        self.other_vehicles[ID] = np.array([data.posA, data.posB])
    
    def get_closest_moving_vehicle(self, car_x, car_y):
        """Gives the ID and position of the closest vehicle

        :param car_x: [m] x coordinate of the car
        :type car_x: float
        :param car_y: [m] y coordinate of the car
        :type car_y: float
        :return: ID and position of the closest vehicle if there are, else -1 and the car position
        :rtype: (int, nd-array)
        """        
        if self.other_vehicles:
            ID, pos = min(self.other_vehicles.items(), key=lambda x: np.linalg.norm(np.array([car_x, car_y]) - x[1]))
            return ID, pos
        else:
            # dictionary is empty
            ros.loginfo("No moving vehicles detected")
            return -1, np.array([car_x, car_y])
    
    def is_other_vehicle_close(self, car_x, car_y):
        """checks if there are moving vehicles near the car

        :param car_x: [m] x coordinate of the car
        :type car_x: float
        :param car_y: [m] y coordinate of the car
        :type car_y: float
        :return: True if there are moving vehicles near the car, else False
        :rtype: bool
        """        
        ID, pos = self.get_closest_moving_vehicle(car_x, car_y)
        if ID < 0:
            return False
        else:
            return np.linalg.norm(np.array([car_x, car_y]) - pos) < VEHICLE_CLOSE_RADIUS

    # V2X CALLBACKS AND FUNCTIONS
    def publish_obstacle(self, type, x, y):
        assert type in self.obstacle_map.keys(), "Obstacle type not recognized"
        data = environmental()
        data.obstacle_id    = self.obstacle_map[type]
        pR = np.array([x,y])
        # pL = mR2mL(pR)
        pL = pR #hopefully we don't need to change reference frame in 2024
        data.x              = pL[0]
        data.y              = pL[1]
        self.pub_v2x.publish(data)
        self.obstacle_list.append(f'{type} found at position ({x},{y}) m')
    
    # SEMAPHORE CALLBACKS AND FUNCTIONS
    def semaphore_master_callback(self, data):
        self.semaphore_states['master'] = data.data

    def semaphore_slave_callback(self, data):
        self.semaphore_states['slave'] = data.data
    
    def semaphore_antimaster_callback(self, data):
        self.semaphore_states['antimaster'] = data.data
    
    def semaphore_start_callback(self, data):
        self.semaphore_states['start'] = data.data
    
    def get_semaphore_state(self, semaphore_key):
        """
        :param semaphore_key: key of the semaphore
        :type semaphore_key: string
        :return: state of the semaphore: 0:RED, 1:YELLOW, 2:GREEN
        :rtype: byte
        """             
        assert semaphore_key in self.semaphore_states.keys(), "Semaphore key not recognized"
        return self.semaphore_states[semaphore_key]

    def get_closest_semaphore_state(self, car_position):
        """
        :param car_position: position of the car
        :type car_position: nd-array
        :return: closest semaphre name and state of the closest semaphore: 0:RED, 1:YELLOW, 2:GREEN
        :rtype: byte
        """
        closest_semaphore = min(SEMAPHORE_POSITIONS.items(), key=lambda x: np.linalg.norm(car_position - x[1]))
        print(f'Closest semaphore: {closest_semaphore[0]}')
        print(f'Closest semaphore state: {self.get_semaphore_state(closest_semaphore[0])}')
        return closest_semaphore[0], self.get_semaphore_state(closest_semaphore[0])



    # STATIC METHODS
    

    def __str__(self):
        description = '''
{:#^65s} 
other_vehicles:
{:s}
{:#^65s} 
obstacle_list:
{:s}
{:#^65s} 
semaphore_states:
{:s}
'''
        return description.format(
                                    ' V2V ', json.dumps(self.other_vehicles, sort_keys=True, indent=2, cls=NumpyEncoder),\
                                    ' V2X ', str(self.obstacle_list),\
                                    ' SEMAPHORES ', json.dumps(self.semaphore_states, sort_keys=True, indent=1)
                                 )
                                    



    










