#!/usr/bin/python3
import networkx, numpy as np, cv2 as cv, os
from time import time, sleep
from pyclothoids import Clothoid
from stuff import *
from stuff.names_and_constants import *
from os.path import join, exists

SHOW_IMGS = False

class PathPlanning(): 
    def __init__(self):
        # start and end nodes
        self.source = str(473)
        self.target = str(207)

        # initialize path
        self.path = []
        self.step_length = PATH_STEP_LENGTH # step length for interpolation

        # previous index of the closest point on the path to the vehicle
        self.prev_index = 0

        self.G = load_graph() # load the map graph

        self.route_list = [] # list of nodes to visit

        self.nodes_data = self.G.nodes.data()
        self.edges_data = self.G.edges.data()

        self.int_mid =  list(np.loadtxt(INT_MID_PATH, dtype=str)) # mid intersection nodes
        self.int_in =   list(np.loadtxt(INT_IN_PATH, dtype=str)) # intersecion entry nodes
        self.int_out =  list(np.loadtxt(INT_OUT_PATH, dtype=str)) # intersecion exit nodes
        self.ra_mid =   list(np.loadtxt(RA_MID_PATH, dtype=str)) # mid roundabout nodes
        self.ra_in =    list(np.loadtxt(RA_IN_PATH, dtype=str)) # roundabout entry nodes
        self.ra_out =   list(np.loadtxt(RA_OUT_PATH, dtype=str)) # roundabout exit nodes
        self.highway_nodes = list(np.loadtxt(HW_PATH, dtype=str)) # highway nodes
        
        self.skip_nodes = [str(i) for i in [262,235,195,196,281,216,263,234]] # 469? # nodes to skip in route generation

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
            min_dist = np.min(np.linalg.norm(p - self.event_points, axis=1))
            if n in self.forbidden_nodes or min_dist < 0.2:
                # print(n)
                pass
            else:
                self.all_start_nodes.append(n)
        self.all_nodes_coords = np.array([self.get_xy(node) for node in self.all_nodes])
        self.all_start_nodes_coords = np.array([self.get_xy(node) for node in self.all_start_nodes])

        self.map, _ = load_map()
    
    def compute_route_list(self, route_graph:networkx.DiGraph):
        ''' Augments the route stored rin oute_graph'''
        self.route_list = [] 
        for n in route_graph.nodes():
            x,y = self.get_xy(n)
            pred = list(route_graph.predecessors(n))
            succ = list(route_graph.successors(n))
            pv = pred[0] if len(pred) > 0 else None # previous node
            nn = succ[0] if len(succ) > 0 else None # next node
            assert not (pv is None and nn is None), f'node {n} has no predecessors or successors'
            if pv is None: self.route_list.append((x,y,self.get_dir(n,nn))) #first node
            elif nn is None: self.route_list.append((x,y,self.get_dir(pv,n))) #last node
            elif n in self.skip_nodes: pass # skip nodes
            else: #general case
                if n in self.int_in or n in self.ra_in: # int/ra entry -> prev edge
                    self.route_list.append((x,y,self.get_dir(pv,n)))
                elif n in self.int_out or n in self.ra_out: # int/ra exit -> next edge
                    self.route_list.append((x,y,self.get_dir(n,nn)))
                elif n in self.ra_mid:
                    if pv in self.ra_in or nn in self.ra_out: pass # ra near ends -> skip
                    else: self.route_list.append((x,y,self.get_dir(pv,nn))) # general case -> avg edge
                elif n in self.int_mid: pass # int mid -> skip
                else: self.route_list.append((x,y,self.get_dir(pv,nn))) # general case -> avg edge

    def compute_shortest_path(self, source=None, target=None, step_length=0.01):
        ''' Generates the shortest path between source and target nodes using Clothoid interpolation '''
        src = str(source) if source is not None else self.source
        tgt = str(target) if target is not None else self.target
        self.source = src
        self.target = tgt
        self.step_length = step_length
        route_nx = []

        # generate the shortest route between source and target      
        route_nx = list(networkx.shortest_path(self.G, source=src, target=tgt)) 
        # generate a route subgraph       
        route_graph = networkx.DiGraph() #reset the graph
        route_graph.add_nodes_from(route_nx) # add nodes
        for i in range(len(route_nx)-1): # add edges
            route_graph.add_edges_from( [ (route_nx[i], route_nx[i+1], self.G.get_edge_data(route_nx[i],route_nx[i+1]))])
        self.compute_route_list(route_graph)
        
        #remove duplicates
        prev_x, prev_y, prev_yaw = 0,0,0
        for i, (x,y,α) in enumerate(self.route_list):
            if np.linalg.norm(np.array([x,y]) - np.array([prev_x,prev_y])) < 0.001:
                #remove element from list
                self.route_list.pop(i)
            prev_x, prev_y, prev_yaw = x,y,α
            
        # interpolate the route of nodes
        self.path = PathPlanning.interpolate_route(self.route_list, step_length)

        return self.path
    
    def augment_path(self, draw=True):
        exit_points = self.int_out + self.ra_out
        exit_points = np.array([self.get_xy(x) for x in exit_points])
        path_exit_points = []
        path_exit_point_idx = []
        #get all the points the path intersects with the exit_points
        for i in range(len(exit_points)):
            p = exit_points[i]
            distances = np.linalg.norm(self.path - p, axis=1)
            index_min_dist = np.argmin(distances)
            min_dist = distances[index_min_dist]
            if min_dist < 0.1:
                p = self.path[index_min_dist]
                path_exit_points.append(p)
                path_exit_point_idx.append(index_min_dist)
                if draw:
                    cv.circle(self.map, p2cv(p), 20, (0,150,0), 5)
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
            distances = np.linalg.norm(self.path - p, axis=1)
            index_min_dist = np.argmin(distances)
            min_dist = distances[index_min_dist]
            if min_dist < 0.05:
                p = self.path[index_min_dist]
                path_event_points.append(p)
                path_event_points_idx.append(index_min_dist)
                path_event_types.append(self.event_types[i])
                if draw:
                    cv.circle(self.map, p2cv(p), 20, (0,255,0), 5)
    
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
                # print(f'local_idx = {local_idx} -- i = {i} -- {self.path_event_points_idx[i]} -- {exit_points_idx[local_idx]}')
                assert len(self.path) > 0
                end_idx = min(exit_points_idx[local_idx]+10, len(self.path))
                path_ahead = self.path[self.path_event_points_idx[i]:end_idx]
                local_idx += 1
                path_event_path_ahead.append(path_ahead)
                if draw:
                    for p in path_ahead:
                        cv.circle(self.map, p2cv(p), 10, (200,150,0), 5)
            elif t.startswith('junction') or t.startswith('highway'):
                assert len(self.path) > 0
                path_ahead = self.path[self.path_event_points_idx[i]:min(self.path_event_points_idx[i]+140, len(self.path))]
                path_event_path_ahead.append(path_ahead)
                if draw:
                    for p in path_ahead:
                        cv.circle(self.map, p2cv(p), 10, (200,150,0), 5)
        
            else:
                path_event_path_ahead.append(None)

        # print(f'local_idx: {local_idx}')
        print("path_event_points_idx: ", self.path_event_points_distances)
        print("path_event_points: ", self.path_event_points)
        print("path_event_types: ", self.path_event_types)
        # sleep(5.0)

        events = list(zip(self.path_event_types, self.path_event_points_distances, self.path_event_points, path_event_path_ahead))
        return events

    def generate_path_passing_through(self, list_of_nodes, step_length=0.01):
        '''Extend the path generation from source-target to a sequence of nodes/locations'''
        assert len(list_of_nodes) >= 2, "List of nodes must have at least 2 nodes"
        print("Generating path passing through: ", list_of_nodes)
        src = list_of_nodes[0]
        tgt = list_of_nodes[1]
        complete_path = self.compute_shortest_path(source=src, target=tgt, step_length=step_length)
        for i in range(1,len(list_of_nodes)-1):
            src = list_of_nodes[i]
            tgt = list_of_nodes[i+1]
            self.compute_shortest_path(source=src, target=tgt, step_length=step_length)
            self.path = self.path[1:] #remove first element of self.path
            complete_path = np.concatenate((complete_path, self.path)) 
        self.path = complete_path

    def get_path_ahead(self, index, look_ahead=100):
        assert index < len(self.path) and index >= 0
        return np.array(self.path[index:min(index + look_ahead, len(self.path)-1), :])

    def get_closest_stop_line(self, nx, ny, draw=False):
        ''' Returns the closest stop point to the given point '''
        index_closest = np.argmin(np.hypot(nx - self.stop_points[:,0], ny - self.stop_points[:,1]))
        print(f'Closest stop point is {self.stop_points[index_closest, :]}, Point is {nx, ny}')
        #draw a circle around the closest stop point
        if draw: cv.circle(self.map, m2pix(self.stop_points[index_closest]), 8, (0, 255, 0), 4)
        return self.stop_points[index_closest, 0], self.stop_points[index_closest, 1]

    def get_length(self, path=None):
        ''' Calculates the length of the trajectory '''
        if path is None:
            return 0
        length = 0
        for i in range(len(path)-1):
            x1,y1 = path[i]
            x2,y2 = path[i+1]
            length += np.hypot(x2-x1,y2-y1) 
        return length

    def get_xy(self, node):
        return np.array([self.nodes_data[node]['x'], self.nodes_data[node]['y']])
    
    def get_dir(self, n1, n2):
        ''' Returns the direction of the edge going from n1 to n2 '''
        x1,y1 = self.get_xy(n1)
        x2,y2 = self.get_xy(n2)
        return np.arctan2(y2-y1, x2-x1)

    def get_path(self):
        return self.path
    
    def compute_path(xi, yi, thi, xf, yf, thf, step_length):
        clothoid_path = Clothoid.G1Hermite(xi, yi, thi, xf, yf, thf)
        length = clothoid_path.length
        [X,Y] = clothoid_path.SampleXY(int(length/step_length))
        return [X,Y]
    
    def interpolate_route(route, step_length):
        path_X = []
        path_Y = []

        # interpolate the route of nodes
        for i in range(len(route)-1):
            xc,yc,thc = route[i]
            xn,yn,thn = route[i+1]
            # thc = np.deg2rad(thc)
            # thn = np.deg2rad(thn)

            #print("from (",xc,yc,thc,") to (",xn,yn,thn,")\n")

            # Clothoid interpolation
            [X,Y] = PathPlanning.compute_path(xc, yc, thc, xn, yn, thn, step_length)

            for x,y in zip(X,Y):
                path_X.append(x)
                path_Y.append(y)
        
        # build array for cv.polylines
        path = []
        prev_x = 0.0
        prev_y = 0.0
        for i, (x,y) in enumerate(zip(path_X, path_Y)):
            if not (np.isclose(x,prev_x, rtol=1e-5) and np.isclose(y, prev_y, rtol=1e-5)):
                path.append([x,y])
            # else:
            #     print(f'Duplicate point: {x}, {y}, index {i}')
            prev_x = x
            prev_y = y

        path = np.array(path, dtype=np.float32)
        return path

    def draw_path(self):
        # draw nodes
        # for node in self.all_nodes:
        #     p = p2cv(self.get_xy(node))
        #     cv.circle(self.map, p, 5, (0, 0, 255), -1) # circle on node position
        #     cv.putText(self.map, str(node), p, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # draw node name

        # # draw edges
        # for edge in self.all_edges:
        #     p1 = self.get_xy(edge[0])
        #     p2 = self.get_xy(edge[1])
        #     cv.line(self.map, p2cv(p1), p2cv(p2), (0, 255, 255), 2)

        # # draw all points in given path
        # for x,y,α in self.route_list:
        #     cv.circle(self.map, xy2cv(x,y), 5, (0, 0, 255), -1) # circle on node position
        #     d = 0.25
        #     nx, ny = x + d*np.cos(α), y + d*np.sin(α)
        #     cv.arrowedLine(self.map, xy2cv(x,y), xy2cv(nx,ny), (255, 0, 255), 2)

        # draw trajectory
        cv.polylines(self.map, [m2pix(self.path)], False, (200, 200, 0), thickness=4, lineType=cv.LINE_AA)
        if SHOW_IMGS:
            cv.imshow('Path', self.map)
            cv.waitKey(1)

    def get_closest_start_node(self, x, y):
        ''' Returns the closes node to the given x,y coordinates '''
        p = np.array([x,y])
        diff = self.all_start_nodes_coords - p
        dist = np.linalg.norm(diff, axis=1)
        index_closest = np.argmin(dist)
        return self.all_start_nodes[index_closest], dist[index_closest]
    
    def get_closest_node(self, x, y):
        ''' Returns the closes node to the given x,y coordinates '''
        p = np.array([x,y])
        diff = self.all_nodes_coords - p
        dist = np.linalg.norm(diff, axis=1)
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