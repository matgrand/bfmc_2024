#!/usr/bin/python3
import numpy as np
WB = 0.26
PP_DA = 0.35 # [m] distance from the car to the point ahead
from time import time

class Controller():
    def __init__(self,k1=0.0,k2=0.0,k3=1.0,k3D=0.08,dist_point_ahead=PP_DA,ff=1.0):
        #controller paramters
        self.k1, self.k2, self.k3, self.k3D = k1, k2, k3, k3D
        self.dist_point_ahead = dist_point_ahead
        self.ff = ff
        self.e1, self.e2, self.e3 = 0.0, 0.0, 0.0
        self.prev_delta = 0.0
        self.prev_time = 0.0

    def get_control(self, e2, e3, curv, desired_speed, gains=None):
        self.e2, self.e3 = e2, e3 # pp error, lateral error
        k2 , k3 = self.k2, self.k3
        if gains is not None: k1, k2, k3, k3D = gains
        # proportional term
        d = self.dist_point_ahead
        delta = np.arctan((2*WB*np.sin(e3))/d)
        proportional_term = k3 * delta
        #derivative term
        k3D = self.k3D
        curr_time = time()
        dt = curr_time - self.prev_time
        self.prev_time = curr_time
        diff3 = (delta - self.prev_delta) / dt
        self.prev_delta = delta
        derivative_term = k3D * diff3
        #feedforward term
        k3FF = self.ff #0.2 #higher for high speeds
        ff_term = -k3FF * curv
        #output    
        output_angle = ff_term - proportional_term - k2 * e2 - derivative_term
        output_speed = desired_speed - self.k1 * self.e1
        return output_speed, output_angle
        