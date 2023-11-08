#!/usr/bin/python3
import networkx as nx, numpy as np, cv2 as cv
from time import time, sleep
from pyclothoids import Clothoid
from stuff import *
from names_and_constants import *
from os.path import join, exists

SHOW_IMGS = False

class PathPlanning(): 
    def __init__(self, map_img):
        # start and end nodes
        self.source = str(86)
        self.target = str(300)

        # initialize path
        self.path = []
        self.navigator = []# set of instruction for the navigator
        self.path_data = [] #additional data for the path (e.g. curvature, 
        self.step_length = 0.01
        # state=following/intersection/roundabout, next_action=left/right/straight, path_params_ahead)

        # previous index of the closest point on the path to the vehicle
        self.prev_index = 0

        # read graph
        graph_path = 'data/Competition_track.graphml'
        assert exists(graph_path), f'Graph file {graph_path} does not exist'
        self.G = nx.read_graphml('data/Competition_track.graphml')
        # initialize route subgraph and list for interpolation
        self.route_graph = nx.DiGraph()
        self.route_list = []
        self.old_nearest_point_index = None # used in search target point

        self.nodes_data = self.G.nodes.data()
        self.edges_data = self.G.edges.data()


        # define intersection central nodes
        #self.intersection_cen = ['347', '37', '39', '38', '11', '29', '30', '28', '371', '84', '9', '20', '20', '82', '75', '74', '83', '312', '315', '65', '468', '10', '64', '57', '56', '66', '73', '424', '48', '47', '46']
        self.intersection_cen = ['468','12','37', '39', '38', '11', '29', '30', '28', '84', '9','19', '20', '21', '82', '75', '74', '83', '312', '315', '65', '10', '64', '57', '56','55', '66', '73', '48', '47', '46']

        # define intersecion entrance nodes
        self.intersection_in = [77,45,54,50,41,79,374,52,43,81,36,4,68,2,34,70,6,32,72,59,15,16,27,14,25,18,61,23,63]
        self.intersection_in = [str(i) for i in self.intersection_in]

        # define intersecion exit nodes
        self.intersection_out = [76,78,80,40,42,44,49,104,53,67,69,71,1,3,5,7,8,31,33,35,22,24,26,13,17,58,60,62]
        self.intersection_out = [str(i) for i in self.intersection_out]

        # define left turns tuples
        #self.left_turns = [(45,42),(43,40),(54,51),(36,33),(34,31),(6,3),(2,7),(2,8),(4,1),(70,67),(72,69),(27,24),(25,22),(16,13),(61,58),(63,60)]

        #define crosswalks 
        self.crosswalk = [92,96,276,295,170,266,177,162] #note: parking spots are included too (177,162)
        self.crosswalk = [str(i) for i in self.crosswalk]

        # define roundabout nodes
        self.ra = [267,268,269,270,271,302,303,304,305,306,307]
        self.ra = [str(i) for i in self.ra]

        self.ra_enter = [301,342,230] #[267,302,305]
        self.ra_enter = [str(i) for i in self.ra_enter]

        self.ra_exit = [272,231,343] # [271,304,306]
        self.ra_exit = [str(i) for i in self.ra_exit]

        self.junctions = [467,314]
        self.junctions = [str(i) for i in self.junctions]

        self.forbidden_nodes = self.intersection_cen + self.intersection_in + self.intersection_out + self.crosswalk + self.ra + self.ra_enter + self.ra_exit + self.junctions
        
        self.no_yaw_calibration_nodes = [str(i) for i in [340,341,342,464,465,466,467]]

        # self.near_crosswalk_nodes = [str(i) for i in [67,95,80,92,96,295,296,276,277,]]

        #event points
        self.event_points = np.load('data/event_points.npy') #created in R coord
        self.event_types = np.load('data/event_types.npy')
        assert len(self.event_points) == len(self.event_types), "event points and types are not the same length"
        event_type_names = EVENT_TYPES
        self.event_types = [event_type_names[int(i)] for i in self.event_types]

        # import nodes and edges
        self.list_of_nodes = list(self.G.nodes)
        self.list_of_edges = list(self.G.edges)

        #spossible starting positions
        all_start_nodes = list(self.G.nodes)
        self.all_start_nodes = []
        print(all_start_nodes)
        print(self.forbidden_nodes)
        for n in all_start_nodes:
            p = self.get_coord(n)
            min_dist = np.min(np.linalg.norm(p - self.event_points, axis=1))
            if n in self.forbidden_nodes or min_dist < 0.2:
                print(n)
            else:
                self.all_start_nodes.append(n)
        self.all_nodes_coords = np.array([self.get_coord(node) for node in self.all_start_nodes])

        #highway nodes
        self.highway_nodes = [str(i) for i in [*range(311,338), *range(375,398), *range(348,371), *range(400,424)]]

        #bumpy road nodes
        self.bumpy_road_nodes = [str(i) for i in [*range(427,466)]]

        # import map to plot trajectory and car
        self.map = map_img

    def roundabout_navigation(self, prev_node, curr_node, next_node):
        while next_node in self.ra:
            prev_node = curr_node
            curr_node = next_node
            if curr_node != self.target:
                next_node = list(self.route_graph.successors(curr_node))[0]
            else:
                next_node = None
                break
                
            #print("inside roundabout: ", curr_node)
            pp = self.get_coord(prev_node)
            xp,yp = pp[0],pp[1]
            pc = self.get_coord(curr_node)
            xc,yc = pc[0],pc[1]
            pn = self.get_coord(next_node)
            xn,yn = pn[0],pn[1]

            if curr_node in self.ra_enter:
                if not(prev_node in self.ra):
                    dx = xn - xp
                    dy = yn - yp
                    self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
                else:
                    if curr_node == '302':
                        continue
                    else:
                        dx = xn - xp
                        dy = yn - yp
                        self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
            elif curr_node in self.ra_exit:
                if next_node in self.ra:
                    # remain inside roundabout
                    if curr_node == '271':
                        continue
                    else:
                        dx = xn - xp
                        dy = yn - yp
                        self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
                else:
                    dx = xn - xp
                    dy = yn - yp
                    self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
            else:
                dx = xn - xp
                dy = yn - yp
                self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))

        prev_node = curr_node
        curr_node = next_node
        if curr_node != self.target:
            next_node = list(self.route_graph.successors(curr_node))[0]
        else:
            next_node = None
        
        self.navigator.append("exit roundabout at "+curr_node)
        self.navigator.append("go straight")
        return prev_node, curr_node, next_node
    
    def intersection_navigation(self, prev_node, curr_node, next_node):
        prev_node = curr_node
        curr_node = next_node
        if curr_node != self.target:
            next_node = list(self.route_graph.successors(curr_node))[0]
        else:
            next_node = None
            return prev_node, curr_node, next_node
        
        prev_node = curr_node
        curr_node = next_node
        if curr_node != self.target:
            next_node = list(self.route_graph.successors(curr_node))[0]
        else:
            next_node = None
        
        self.navigator.append("exit intersection at " + curr_node)
        self.navigator.append("go straight")
        return prev_node, curr_node, next_node

    def compute_route_list(self):
        ''' Augments the route stored in self.route_graph'''
        #print("source=",source)
        #print("target=",target)
        #print("edges=",self.route_graph.edges.data())
        #print("nodes=",self.route_graph.nodes.data())
        #print(self.route_graph)
        
        curr_node = self.source
        prev_node = curr_node
        next_node = list(self.route_graph.successors(curr_node))[0]

        #reset route list
        self.route_list = []

        #print("curr_node",curr_node)
        #print("prev_node",prev_node)
        #print("next_node",next_node)
        self.navigator.append("go straight")
        while curr_node != self.target:
            pp = self.get_coord(prev_node)
            xp,yp = pp[0],pp[1]
            pc = self.get_coord(curr_node)
            xc,yc = pc[0],pc[1]
            pn = self.get_coord(next_node)
            xn,yn = pn[0],pn[1]


            #print("from",curr_node,"(",xc,yc,")"," to ",next_node,"(",xn,yn,")")
            #print("edge: ",self.route_graph.get_edge_data(curr_node, next_node))
            #print("prev_node is",prev_node,"(",xp,yp,")")
            #print("*************************\n")
            
            curr_is_junction = curr_node in self.intersection_cen#len(adj_nodes) > 1
            next_is_intersection = next_node in self.intersection_cen#len(next_adj_nodes) > 1
            prev_is_intersection = prev_node in self.intersection_cen#len(prev_adj_nodes) > 1
            next_is_roundabout_enter = next_node in self.ra_enter
            curr_in_roundabout = curr_node in self.ra
            #print(f"CURR: {curr_node}, NEXT: {next_node}, PREV: {prev_node}")
            # ****** ROUNDABOUT NAVIGATION ******
            if next_is_roundabout_enter:
                if curr_node == "342":
                    self.route_list.append((xc,yc,np.rad2deg(np.arctan2(-1,0))))
                    dx = xc - xp
                    dy = yc - yp
                    self.route_list.append((xc,yc+0.3*dy,np.rad2deg(np.arctan2(-1,0))))
                else:
                    dx = xc-xp
                    dy = yc-yp
                    self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
                    # add a further node
                    dx = xc - xp
                    dy = yc - yp
                    self.route_list.append((xc+0.3*dx,yc+0.3*dy,np.rad2deg(np.arctan2(dy,dx))))
                # enter the roundabout
                self.navigator.append("enter roundabout at " + curr_node)
                prev_node, curr_node, next_node = self.roundabout_navigation(prev_node, curr_node, next_node)
                continue               
            elif next_is_intersection:
                dx = xc - xp
                dy = yc - yp
                self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
                # add a further node
                dx = xc - xp
                dy = yc - yp
                self.route_list.append((xc+0.3*dx,yc+0.3*dy,np.rad2deg(np.arctan2(dy,dx))))
                # enter the intersection
                self.navigator.append("enter intersection at " + curr_node)
                prev_node, curr_node, next_node = self.intersection_navigation(prev_node, curr_node, next_node)
                continue
            elif prev_is_intersection:
                # add a further node
                dx = xn - xc
                dy = yn - yc
                self.route_list.append((xc-0.3*dx,yc-0.3*dy,np.rad2deg(np.arctan2(dy,dx))))
                # and add the exit node
                dx = xn - xc
                dy = yn - yc
                self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
            else:
                dx = xn - xp
                dy = yn - yp
                self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
            
            prev_node = curr_node
            curr_node = next_node
            if curr_node != self.target:
                next_node = list(self.route_graph.successors(curr_node))[0]
            else:
                # You arrived at the END
                dx = xn - xp
                dy = yn - yp
                self.route_list.append((xn,yn,np.rad2deg(np.arctan2(dy,dx))))
                next_node = None
        
        self.navigator.append("stop")

    def compute_shortest_path(self, source=None, target=None, step_length=0.01):
        ''' Generates the shortest path between source and target nodes using Clothoid interpolation '''
        src = str(source) if source is not None else self.source
        tgt = str(target) if target is not None else self.target
        self.source = src
        self.target = tgt
        self.step_length = step_length
        route_nx = []

        # generate the shortest route between source and target      
        route_nx = list(nx.shortest_path(self.G, source=src, target=tgt)) 
        # generate a route subgraph       
        self.route_graph = nx.DiGraph() #reset the graph
        self.route_graph.add_nodes_from(route_nx)
        for i in range(len(route_nx)-1):
            self.route_graph.add_edges_from( [ (route_nx[i], route_nx[i+1], self.G.get_edge_data(route_nx[i],route_nx[i+1])) ] )         # augment route with intersections and roundabouts
        # expand route node and update navigation instructions
        self.compute_route_list()
        
        #remove duplicates
        prev_x, prev_y, prev_yaw = 0,0,0
        for i, (x,y,yaw) in enumerate(self.route_list):
            if np.linalg.norm(np.array([x,y]) - np.array([prev_x,prev_y])) < 0.001:
                #remove element from list
                self.route_list.pop(i)
            prev_x, prev_y, prev_yaw = x,y,yaw
            
        # interpolate the route of nodes
        self.path = PathPlanning.interpolate_route(self.route_list, step_length)

        return self.path
    
    def augment_path(self, draw=True):
        exit_points = self.intersection_out + self.ra_exit
        exit_points = np.array([self.get_coord(x) for x in exit_points])
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
                    cv.circle(self.map, m2pix(p), 20, (0,150,0), 5)
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
                    cv.circle(self.map, m2pix(p), 20, (0,255,0), 5)
    
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
                        cv.circle(self.map, m2pix(p), 10, (200,150,0), 5)
            elif t.startswith('junction') or t.startswith('highway'):
                assert len(self.path) > 0
                path_ahead = self.path[self.path_event_points_idx[i]:min(self.path_event_points_idx[i]+140, len(self.path))]
                path_event_path_ahead.append(path_ahead)
                if draw:
                    for p in path_ahead:
                        cv.circle(self.map, m2pix(p), 10, (200,150,0), 5)
        
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
        """
        Extend the path generation from source-target to a sequence of nodes/locations
        """
        assert len(list_of_nodes) >= 2, "List of nodes must have at least 2 nodes"
        print("Generating path passing through: ", list_of_nodes)
        src = list_of_nodes[0]
        tgt = list_of_nodes[1]
        complete_path = self.compute_shortest_path(source=src, target=tgt, step_length=step_length)
        for i in range(1,len(list_of_nodes)-1):
            src = list_of_nodes[i]
            tgt = list_of_nodes[i+1]
            self.compute_shortest_path(source=src, target=tgt, step_length=step_length)
            #now the local path is computed in self.path
            #remove first element of self.path
            self.path = self.path[1:]
            #we need to add the local path to the complete path
            complete_path = np.concatenate((complete_path, self.path))
        self.path = complete_path

    def get_path_ahead(self, index, look_ahead=100):
        assert index < len(self.path) and index >= 0
        return np.array(self.path[index:min(index + look_ahead, len(self.path)-1), :])

    def get_closest_stop_line(self, nx, ny, draw=False):
        """
        Returns the closest stop point to the given point
        """
        index_closest = np.argmin(np.hypot(nx - self.stop_points[:,0], ny - self.stop_points[:,1]))
        print(f'Closest stop point is {self.stop_points[index_closest, :]}, Point is {nx, ny}')
        #draw a circle around the closest stop point
        if draw: cv.circle(self.map, m2pix(self.stop_points[index_closest]), 8, (0, 255, 0), 4)
        return self.stop_points[index_closest, 0], self.stop_points[index_closest, 1]

    def print_path_info(self):
        prev_state = None
        prev_next_state = None
        for i in range(len(self.path_data)-1):
            curr_state = self.path_data[i][0]
            next_state = self.path_data[i][1]
            if curr_state != prev_state or next_state != prev_next_state:
                print(f'{i}: {self.path_data[i]}')
            prev_state = curr_state
            prev_next_state = next_state

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

    def get_coord(self, node):
        x = self.nodes_data[node]['x']
        y = self.nodes_data[node]['y']
        p = np.array([x,y])
        return p
    
    def get_path(self):
        return self.path
    
    def print_navigation_instructions(self):
        for i,instruction in enumerate(self.navigator):
            print(i+1,") ",instruction)
    
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
            thc = np.deg2rad(thc)
            thn = np.deg2rad(thn)

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
        for node in self.list_of_nodes:
            p = self.get_coord(node) #change to r coordinate
            cv.circle(self.map, m2pix(p), 5, (0, 0, 255), -1)
            #add node number
            cv.putText(self.map, str(node), m2pix(p), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw edges
        for edge in self.list_of_edges:
            p1 = self.get_coord(edge[0])
            p2 = self.get_coord(edge[1])
            #cv.line(self.map, m2pix(p1), m2pix(p2), (0, 255, 255), 2)

        # draw all points in given path
        for point in self.route_list:
            x,y,_ = point
            p = np.array([x,y])
            cv.circle(self.map, m2pix(p), 5, (255, 0, 0), 1)

        # draw trajectory
        cv.polylines(self.map, [m2pix(self.path)], False, (200, 200, 0), thickness=4, lineType=cv.LINE_AA)
        if SHOW_IMGS:
            cv.imshow('Path', self.map)
            cv.waitKey(1)

    def get_closest_node(self, p):
        '''
        Returns the closes node to the given point np.array([x,y])
        '''
        diff = self.all_nodes_coords - p
        dist = np.linalg.norm(diff, axis=1)
        index_closest = np.argmin(dist)
        return self.all_start_nodes[index_closest], dist[index_closest]

    def is_dotted(self, n):
        """
        Check if a node is close to a dotted line
        """
        #get edge of the node going out
        edges = self.G.out_edges(n)
        for e in edges:
            if not self.G.get_edge_data(e[0],e[1])['dotted']:
                return False
        return True