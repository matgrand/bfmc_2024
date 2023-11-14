#!/usr/bin/python3
import networkx, numpy as np, cv2 as cv, os
from time import time, sleep
from pyclothoids import Clothoid
from stuff import *
from stuff.names_and_constants import *
from os.path import join, exists
from numpy.linalg import norm

SHOW_IMGS = False

class PathPlanning(): 
    def __init__(self):
        self.G = load_graph() # load the map graph
        self.path = [] # list of points in the path

        self.int_mid =  list(np.loadtxt(INT_MID_PATH, dtype=str)) # mid intersection nodes
        self.int_in =   list(np.loadtxt(INT_IN_PATH, dtype=str)) # intersecion entry nodes
        self.int_out =  list(np.loadtxt(INT_OUT_PATH, dtype=str)) # intersecion exit nodes
        self.ra_mid =   list(np.loadtxt(RA_MID_PATH, dtype=str)) # mid roundabout nodes
        self.ra_in =    list(np.loadtxt(RA_IN_PATH, dtype=str)) # roundabout entry nodes
        self.ra_out =   list(np.loadtxt(RA_OUT_PATH, dtype=str)) # roundabout exit nodes
        self.highway_nodes = list(np.loadtxt(HW_PATH, dtype=str)) # highway nodes
        
        self.skip_nodes = [str(i) for i in [262,235,195,196,281,216,263,234,239,301,282,258]] # 469? # nodes to skip in route generation

        self.forbidden_nodes = self.int_mid + self.int_in + self.int_out + self.ra_mid + self.ra_in + self.ra_out 

        #event points
        self.event_points = np.loadtxt(EVENT_POINTS_PATH, dtype=np.float32)
        self.event_types = [EVENT_TYPES[int(i)] for i in np.loadtxt(EVENT_TYPES_PATH, dtype=np.int32)]
        assert len(self.event_points) == len(self.event_types), "event points and types are not the same length"

        # import nodes and edges
        self.all_nodes = list(self.G.nodes)
        self.all_edges = list(self.G.edges)

        #possible starting positions
        self.all_start_nodes = []
        for n in self.all_nodes:
            p = self.get_xy(n)
            min_dist = np.min(norm(p - self.event_points, axis=1))
            if n in self.forbidden_nodes or min_dist < 0.2:
                # print(n)
                pass
            else:
                self.all_start_nodes.append(n)
        self.all_nodes_coords = np.array([self.get_xy(node) for node in self.all_nodes])
        self.all_start_nodes_coords = np.array([self.get_xy(node) for node in self.all_start_nodes])

        self.map, _ = load_map()
    
    def compute_shortest_path(self, source=473, target=207):
        ''' Generates the shortest path between source and target nodes using Clothoid interpolation '''
        src, tgt = str(source), str(target)
        route_nx = []

        # generate the shortest route between source and target      
        route_nx = list(networkx.shortest_path(self.G, source=src, target=tgt)) 
        # generate a route subgraph       
        routeG = networkx.DiGraph() #reset the graph
        routeG.add_nodes_from(route_nx) # add nodes
        for i in range(len(route_nx)-1): # add edges
            routeG.add_edges_from( [ (route_nx[i], route_nx[i+1], self.G.get_edge_data(route_nx[i],route_nx[i+1]))])
        # convert the graph in a list of poses (x,y,θ)
        route = [] # list of nodes in (x,y,θ) format
        for n in routeG.nodes():
            x,y = self.get_xy(n) # node position
            pred = list(routeG.predecessors(n))
            succ = list(routeG.successors(n))
            pv = pred[0] if len(pred) > 0 else None # previous node
            nn = succ[0] if len(succ) > 0 else None # next node
            assert not (pv is None and nn is None), f'node {n} has no predecessors or successors'
            if pv is None: route.append((x,y,self.get_dir(n,nn))) #first node
            elif nn is None: route.append((x,y,self.get_dir(pv,n))) #last node
            elif n in self.skip_nodes: pass # skip nodes
            else: #general case
                if n in self.int_in or n in self.ra_in: # int/ra entry -> prev edge
                    route.append((x,y,self.get_dir(pv,n)))
                elif n in self.int_out or n in self.ra_out: # int/ra exit -> next edge
                    route.append((x,y,self.get_dir(n,nn)))
                elif n in self.ra_mid: # ra mid 
                    if pv in self.ra_in or nn in self.ra_out: pass # ra near ends -> skip
                    else: route.append((x,y,self.get_dir(pv,nn))) # general case -> avg edge
                elif n in self.int_mid: pass # int mid -> skip
                else: route.append((x,y,self.get_dir(pv,nn))) # general case -> avg edge
        # interpolate the route of nodes
        path = [] # list of points in the path
        for i in range(len(route)-1):
            xc,yc,θc = route[i] # current node
            xn,yn,θn = route[i+1] # next node
            clothoid_path = Clothoid.G1Hermite(xc, yc, θc, xn, yn, θn) # clothoid interpolation
            [X,Y] = clothoid_path.SampleXY(int(clothoid_path.length/PATH_STEP_LENGTH)) # resample the path
            for x,y in zip(X,Y): path.append(np.array([x,y])) # add points to the path
        px, py = 0.0, 0.0 # previous point
        for i, (x,y) in enumerate(path): # remove possible duplicates
            if np.hypot(x-px, y-py) < 0.1*PATH_STEP_LENGTH: path.pop(i) 
            px, py = x, y
        self.path = np.array(path, dtype=np.float32) # convert to numpy array
        return self.path
    
    def augment_path(self, draw=True):
        exit_points = self.int_out + self.ra_out
        exit_points = np.array([self.get_xy(x) for x in exit_points])
        path_exit_points = []
        path_exit_point_idx = []
        #get all the points the path intersects with the exit_points
        for i in range(len(exit_points)):
            p = exit_points[i]
            distances = norm(self.path - p, axis=1)
            index_min_dist = np.argmin(distances)
            min_dist = distances[index_min_dist]
            if min_dist < 0.1:
                p = self.path[index_min_dist]
                path_exit_points.append(p)
                path_exit_point_idx.append(index_min_dist)
                if draw: cv.circle(self.map, p2cv(p), 20, (0,150,0), 5)
        path_exit_point_idx = np.array(path_exit_point_idx)
        #reorder points by idx
        exit_points = []
        exit_points_idx = []
        if len(path_exit_point_idx) > 0:
            max_idx = max(path_exit_point_idx)
            for i in range(len(path_exit_points)):
                min_idx = np.argmin(path_exit_point_idx)
                exit_points.append(path_exit_points[min_idx])
                exit_points_idx.append(path_exit_point_idx[min_idx])
                path_exit_point_idx[min_idx] = max_idx+1
        #get all the points the path intersects with the stop_line_points
        path_event_points = []
        path_event_points_idx = []
        path_event_types = []
        for i in range(len(self.event_points)):
            p = self.event_points[i]
            distances = norm(self.path - p, axis=1)
            index_min_dist = np.argmin(distances)
            min_dist = distances[index_min_dist]
            if min_dist < 0.05:
                p = self.path[index_min_dist]
                path_event_points.append(p)
                path_event_points_idx.append(index_min_dist)
                path_event_types.append(self.event_types[i])
                if draw: cv.circle(self.map, p2cv(p), 20, (0,255,0), 5)
        path_event_points_idx = np.array(path_event_points_idx)
        #reorder
        self.path_event_points = []
        self.path_event_points_distances = []
        self.path_event_points_idx = []
        self.path_event_types = []
        if len(path_event_points) > 0:
            max_idx = np.max(path_event_points_idx)
            for i in range(len(path_event_points)):
                min_idx = np.argmin(path_event_points_idx)
                self.path_event_points.append(path_event_points[min_idx])
                self.path_event_points_idx.append(path_event_points_idx[min_idx])
                self.path_event_points_distances.append(0.01*path_event_points_idx[min_idx])
                self.path_event_types.append(path_event_types[min_idx])
                path_event_points_idx[min_idx] = max_idx + 1
        #add path ahead after intersections and roundabouts
        path_event_path_ahead = []
        local_idx = 0
        for i in range(len(path_event_points)):
            t = self.path_event_types[i]
            if t.startswith('intersection') or t.startswith('roundabout'):
                assert len(self.path) > 0
                end_idx = min(exit_points_idx[local_idx]+10, len(self.path))
                path_ahead = self.path[self.path_event_points_idx[i]:end_idx]
                local_idx += 1
                path_event_path_ahead.append(path_ahead)
                if draw:
                    for p in path_ahead: cv.circle(self.map, p2cv(p), 10, (200,150,0), 5)
            elif t.startswith('junction') or t.startswith('highway'):
                assert len(self.path) > 0
                path_ahead = self.path[self.path_event_points_idx[i]:min(self.path_event_points_idx[i]+140, len(self.path))]
                path_event_path_ahead.append(path_ahead)
                if draw:
                    for p in path_ahead: cv.circle(self.map, p2cv(p), 10, (200,150,0), 5)
            else:
                path_event_path_ahead.append(None)
        print(f'path_event_points_idx: {self.path_event_points_distances}')
        print(f'path_event_points: {self.path_event_points}')
        print(f'path_event_types: {self.path_event_types}')
        events = list(zip(self.path_event_types, self.path_event_points_distances, self.path_event_points, path_event_path_ahead))
        return events

    def generate_path_passing_through(self, lon):
        '''Extend the path generation from source-target to a sequence of nodes/locations
        lon: list of nodes/locations to pass through'''
        assert len(lon) >= 2, "List of nodes must have at least 2 nodes"
        assert all([isinstance(n, int) for n in lon]), f'List of nodes must be integers, not {type(lon[0])}'
        print("Generating path passing through: ", lon)
        complete_path = self.compute_shortest_path(source=lon[0], target=lon[1])
        for i in range(1,len(lon)-1):
            self.compute_shortest_path(source=lon[i], target=lon[i+1]) #generate path from node i to node i+1
            self.path = self.path[1:] #remove first element of self.path
            complete_path = np.concatenate((complete_path, self.path)) 
        self.path = complete_path

    def get_closest_stop_line(self, nx, ny, draw=False):
        ''' Returns the closest stop point to the given point '''
        index_closest = np.argmin(np.hypot(nx - self.stop_points[:,0], ny - self.stop_points[:,1]))
        print(f'Closest stop point is {self.stop_points[index_closest, :]}, Point is {nx, ny}')
        #draw a circle around the closest stop point
        if draw: cv.circle(self.map, m2pix(self.stop_points[index_closest]), 8, (0, 255, 0), 4)
        return self.stop_points[index_closest, 0], self.stop_points[index_closest, 1]

    def get_xy(self, node):
        return np.array([self.G.nodes.data()[node]['x'], self.G.nodes.data()[node]['y']])
    
    def get_dir(self, n1, n2):
        ''' Returns the direction of the edge going from n1 to n2 '''
        x1,y1 = self.get_xy(n1)
        x2,y2 = self.get_xy(n2)
        return np.arctan2(y2-y1, x2-x1)

    def draw_path(self):
        # draw nodes
        self.map = cv.flip(self.map, 0) # flip the map vertically
        for node in self.all_nodes:
            x,y = self.get_xy(node)
            x,y = x, MAP_H_M - y
            cv.circle(self.map, xy2cv(x,y), 5, (0, 0, 255), -1) # circle on node position
            cv.putText(self.map, str(node), xy2cv(x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # draw node name
        self.map = cv.flip(self.map, 0) # flip the map back

        # # draw edges
        # for edge in self.all_edges:
        #     p1 = self.get_xy(edge[0])
        #     p2 = self.get_xy(edge[1])
        #     cv.line(self.map, p2cv(p1), p2cv(p2), (0, 255, 255), 2)

        # draw trajectory
        cv.polylines(self.map, [m2pix(self.path)], False, (200, 200, 0), thickness=4, lineType=cv.LINE_AA)
        if SHOW_IMGS:
            cv.imshow('Path', self.map)
            cv.waitKey(1)

    def get_closest_start_node(self, x, y):
        ''' Returns the closes node to the given x,y coordinates '''
        p = np.array([x,y])
        diff = self.all_start_nodes_coords - p
        dist = norm(diff, axis=1)
        index_closest = np.argmin(dist)
        return self.all_start_nodes[index_closest], dist[index_closest]
    
    def get_closest_node(self, x, y):
        ''' Returns the closes node to the given x,y coordinates '''
        p = np.array([x,y])
        diff = self.all_nodes_coords - p
        dist = norm(diff, axis=1)
        index_closest = np.argmin(dist)
        return self.all_nodes[index_closest], dist[index_closest]

    def is_dotted(self, n):
        ''' Check if a node is close to a dotted line '''
        #get edge of the node going out
        edges = self.G.out_edges(n)
        for e in edges:
            if not self.G.get_edge_data(e[0],e[1])['dotted']:
                return False
        return True