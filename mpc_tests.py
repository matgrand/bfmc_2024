from stuff import *
import matplotlib.pyplot as plt
from tqdm import tqdm

MAP_SIZE = (12223, 8107)
MAP_SIZE_M = np.array([22.6166, 15.0])

START_POINT = np.array([13.0, 1.5]) #np.array([12.99, 2.016])

# define a funcion tha kills ros masterwhen ctrl+c is pressed
import signal, sys
def signal_handler(sig, frame):
    print('Exiting.........')
    kill_ros_master()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def create_track(step_length=0.01, road_width=0.4, n=36, md=2.8, bd=2.5, start_p=START_POINT):
    '''
    step_length: step length for the clothoid interpolation
    road_width: width of the road
    n: number of random points to generate
    md: minimum distance between points
    bd: border distance
    start_p: start point of the track
    '''
    assert start_p[0] < bd or start_p[0] > MAP_SIZE_M[0] - bd or start_p[1] < bd or start_p[1] > MAP_SIZE_M[1] - bd, \
        'start point must be in the border'
    
    #define n random points within the map boundaries MAP_SIZE_M
    pts = [start_p]
    for i in range(1, n):
        for _ in range(1000):
            p = bd + np.random.rand(2) * (MAP_SIZE_M - 2*bd)
            if np.min(np.linalg.norm(pts[:i] - p, axis=1)) > md:
                pts.append(p)
                break
    pts = np.array(pts)
        
    #split the points in 3 consecutive convex hulls
    list_hulls = []
    for i in range(5): # will most likely be 3
        #get the convex hull of the points
        from scipy.spatial import ConvexHull
        if len(pts) < 6: break
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        list_hulls.append(hull_pts)

        #remove the points inside the convex hull
        mask = np.ones(len(pts), dtype=bool)
        mask[hull.vertices] = False
        pts = pts[mask]

    extch = list_hulls[0] # external convex hull
    intch = list_hulls[-1] # internal convex hull

    # select 3 sides from the internal convex hull
    si = np.sort(np.random.choice(len(intch), 3, replace=False)) #select 3 sides indices
    sides = np.array([[intch[si[i]], intch[(si[i]+1)%len(intch)]] for i in range(3)]) #select 3 sides
    track_points = extch #initialize the track points with the external convex hull

    #divide the external convex hull in nturns segments usign the internal convex hull
    v0, v1, v2 = sides[0,1] - sides[0,0], sides[1,1] - sides[1,0], sides[2,1] - sides[2,0]
    mask0 = np.sign(np.cross(v0, extch - sides[0,0])) != np.sign(np.cross(v0, v1))
    mask1 = np.sign(np.cross(v1, extch - sides[1,0])) != np.sign(np.cross(v1, v2))
    mask2 = np.sign(np.cross(v2, extch - sides[2,0])) != np.sign(np.cross(v2, v0))
    section = np.ones(len(extch))*-1
    section[mask0 & ~(mask1 | mask2)] = 0
    section[mask1 & ~(mask0 | mask2)] = 1
    section[mask2 & ~(mask0 | mask1)] = 2
    
    # for each section, if there are points, select one at random and add 
    # it to the track points, at the corresponding section
    split_and_poins = {}
    for i in range(3):
        candidates = np.where(section == i)[0]
        if len(candidates) > 0: #if there are points in the section
            idx = np.random.choice(candidates) #select one at random
            # p = 0.5 * (intch[i] + intch[(i+1)%3]) #add it to the track points
            p = 0.5 * (sides[i,0] + sides[i,1]) #add it to the track points
            split_and_poins[idx] = p #add the point to the dictionary
    split_and_poins = dict(sorted(split_and_poins.items())) #sort the dictionary by index
    for j, (i, p) in enumerate(split_and_poins.items()): #add the points to the track points
        track_points = np.vstack((track_points[0:i+j], p, track_points[i+j:]))
    track_points = np.vstack((track_points, track_points[0])) #close the track

    #interpolate using clothoids
    track_points = track_points[:-1] #remove the last point
    thetas = np.zeros(len(track_points))
    for i in range(len(track_points)):
        pp = track_points[(i-1)%len(track_points)] #previous point
        cp = track_points[i] #current point
        fp = track_points[(i+1)%len(track_points)] #following point
        a = np.arctan2(fp[1]-cp[1], fp[0]-cp[0]) 
        b = np.arctan2(cp[1]-pp[1], cp[0]-pp[0])
        #get a middle point ahead between a and b
        pa = cp + 0.5 * np.array([np.cos(a), np.sin(a)])
        pb = cp + 0.5 * np.array([np.cos(b), np.sin(b)])
        pm = 0.5 * (pa + pb) #middle point
        thetas[i] =  np.arctan2(pm[1]-cp[1], pm[0]-cp[0]) #angle of the middle point
    track_points = np.hstack((track_points, thetas.reshape(-1,1))) #add the thetas to the track points

    # set the theta of start_p to 0
    track_points[np.argmin(np.linalg.norm(track_points[:,:2] - start_p, axis=1)), 2] = 0
    

    # # add all the middle points between each pair of points
    # new_track_points = []    
    # for i in range(len(track_points)):
    #     new_track_points.append(track_points[i])
    #     xc,yc,thc = track_points[i]
    #     xf,yf,thf = track_points[(i+1)%len(track_points)]
    #     #add 2 middle points one at 1/3 and one at 2/3
    #     x1, y1 = (2*xc+xf)/3, (2*yc+yf)/3
    #     x2, y2 = (xc+2*xf)/3, (yc+2*yf)/3
    #     th = np.arctan2(yf-yc, xf-xc)
    #     new_track_points.append([x1, y1, th])
    #     new_track_points.append([x2, y2, th])
    # track_points = np.array(new_track_points)

    from pyclothoids import Clothoid
    track = []
    for i in range(len(track_points)):
        xc,yc,thc = track_points[i]
        xf,yf,thf = track_points[(i+1)%len(track_points)]
        cloth_path = Clothoid.G1Hermite(xc,yc,thc,xf,yf,thf)
        [xs,ys] = cloth_path.SampleXY(int(cloth_path.length/step_length))
        for x, y in zip(xs, ys): track.append([x, y])
    track = np.array(track)

    # remove duplicate points
    track = track[np.linalg.norm(np.diff(np.vstack((track, track[0])), axis=0), axis=1) >= 0.1 * step_length]

    #get the external lines
    th = np.arctan2(np.diff(track[:, 1]), np.diff(track[:, 0]))
    dxl, dyl = 0.5 * road_width * np.cos(th + np.pi / 2), 0.5 * road_width * np.sin(th + np.pi / 2)
    dxr, dyr = 0.5 * road_width * np.cos(th - np.pi / 2), 0.5 * road_width * np.sin(th - np.pi / 2)
    left_lane = track[:-1] + np.column_stack((dxl, dyl))
    right_lane = track[:-1] + np.column_stack((dxr, dyr))

    #check if there are points outside the map
    k = 0.5+0.1
    if np.any((track-k*road_width) < 0) or np.any((track+k*road_width) > MAP_SIZE_M):
        # print('Some points are outside the map, creating a new track')
        return create_track()
    
    # check for self intersection
    from shapely.geometry import LineString
    if not LineString(left_lane).is_simple or not LineString(right_lane).is_simple:
        # print('Self intersection detected, creating a new track')
        return create_track()
    
    return track, left_lane, right_lane

def draw_map(track, left_lane, right_lane, line_width=0.02):
    #define b/w image
    img = np.zeros((int(MAP_SIZE[1]), int(MAP_SIZE[0])), dtype=np.uint8)
    ll = m2pix(left_lane, k=K_SMALL)
    rl = m2pix(right_lane, k=K_SMALL)

    line_width = m2pix(line_width, k=K_SMALL)

    cv.polylines(img, [ll], False, 255, line_width)
    cv.polylines(img, [rl], False, 255, line_width)

    #flip image upside down
    img = cv.flip(img, 0)

    #save the image
    cv.imwrite('tmp/swap.png', img)

    return img



# if __name__ == '__main__':
#     cv.namedWindow('track', cv.WINDOW_NORMAL)
#     cv.resizeWindow('track', 800, 600)
#     for i in range(1000):
#         tr, ll, rl = create_track()
#         draw_map(tr, ll, rl)
#         # plot the track
#         fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#         ax.plot(tr[:,0], tr[:,1], 'k--')
#         ax.plot(ll[:,0], ll[:,1], 'k-')
#         ax.plot(rl[:,0], rl[:,1], 'k-')
#         ax.axis('equal')
#         ax.set_xlim(0, MAP_SIZE_M[0])
#         ax.set_ylim(0, MAP_SIZE_M[1])
#         ax.grid(True)
#         ax.set_xticks(np.arange(0, MAP_SIZE_M[0]+1, 2))
#         plt.tight_layout()
#         fig.savefig('tmp/track.png', dpi=100)
#         plt.close(fig)
#         img = cv.imread('tmp/track.png')
#         cv.imshow('track', img)
#         if cv.waitKey(0) == 27:
#             break
#     cv.destroyAllWindows()


if __name__ == '__main__':
    

    for _ in range(20):

    #create a random color map
        tr, ll, rl = create_track()
        new_map = draw_map(tr, ll, rl)

        change_track(new_map) #change the map

        run_ros_master('bare_car_with_map.launch') #run the simulator
        wait_for_ros_startup() #wait for the simulator to start

        #place car in start point
        place_car(START_POINT[0], START_POINT[1], 0)

        # while not ros.is_shutdown():
        #     print('running', end='\r')
        #     sleep(0.2)
        sleep(5)

        kill_ros_master() #kill the simulator
