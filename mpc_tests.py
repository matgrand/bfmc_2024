from stuff import *
import matplotlib.pyplot as plt

MAP_SIZE = (12223, 8107)
MAP_SIZE_M = np.array([22.6166, 15.0])

# define a funcion tha kills ros masterwhen ctrl+c is pressed
import signal, sys
def signal_handler(sig, frame):
    print('Exiting.........')
    kill_ros_master()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)



def create_track():
    FIG_SIZE = (10, 8)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    #define n random points within the map boundaries MAP_SIZE_M
    n = 36
    md = 2.0 # [m] maximum distance between points
    bd = 2.5 # [m] minimum distance from the boundary
    start_p = np.array([13.0, 1.5]) #np.array([12.99, 2.016])
    pts = [start_p]
    for i in range(1, n):
        for _ in range(1000):
            p = bd + np.random.rand(2) * (MAP_SIZE_M - 2*bd)
            if np.min(np.linalg.norm(pts[:i] - p, axis=1)) > md:
                pts.append(p)
                break
    pts = np.array(pts)
        
    #draw the points with matplotlib
    ax.plot(pts[:,0], pts[:,1], 'ko', markersize=3)
    ax.set_xlim(0, MAP_SIZE_M[0])
    ax.set_ylim(0, MAP_SIZE_M[1])
    ax.grid(True)
    ax.set_xticks(np.arange(0, MAP_SIZE_M[0], 2))
    ax.set_yticks(np.arange(0, MAP_SIZE_M[1], 2))
    ax.set_aspect('equal')

    #split the points in 3 consecutive convex hulls
    colors = ['g', 'y', 'm']
    list_hulls = []
    for i in range(3): # will most likely be 3
        #get the convex hull of the points
        from scipy.spatial import ConvexHull
        if len(pts) < 3: break
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        list_hulls.append(hull_pts)

        #remove the points inside the convex hull
        mask = np.ones(len(pts), dtype=bool)
        mask[hull.vertices] = False
        pts = pts[mask]

    # #draw the convex hulls
    # for i, hull in enumerate(list_hulls):
    #     ax.fill(hull[:,0], hull[:,1], colors[i], alpha=1)

    extch = list_hulls[0] # external convex hull
    intch = list_hulls[-1] # internal convex hull

    #reduce the internal convex hull to a triangle by randomly choosing 3 points
    intch = intch[np.sort(np.random.choice(len(intch), 3, replace=False))]

    #draw external and internal convex hulls
    ax.fill(extch[:,0], extch[:,1], colors[0], alpha=0.5)
    ax.fill(intch[:,0], intch[:,1], colors[2], alpha=0.5)

    # blue green red and gray
    colors = ['b', 'g', 'r']
    # shapes = ['1', '2', '3']
    # shapes = ['^', '>', '<']
    shapes = ['o', 'o', 'o']

    track_points = extch #initialize the track points with the external convex hull

    #divide the external convex hull in nturns segments usign the internal convex hull
    v0, v1, v2 = intch[1] - intch[0], intch[2] - intch[1], intch[0] - intch[2]
    mask0 = np.sign(np.cross(v0, extch - intch[0])) != np.sign(np.cross(v0, v1))
    mask1 = np.sign(np.cross(v1, extch - intch[1])) != np.sign(np.cross(v1, v2))
    mask2 = np.sign(np.cross(v2, extch - intch[2])) != np.sign(np.cross(v2, v0))

    section = np.ones(len(extch))*-1
    section[mask0 & ~(mask1 | mask2)] = 0
    section[mask1 & ~(mask0 | mask2)] = 1
    section[mask2 & ~(mask0 | mask1)] = 2
    
    #color the points according to the section
    for i in range(0,3):
        mask = section == i
        ax.plot(extch[mask,0], extch[mask,1], colors[i]+shapes[i], markersize=10)
        p1, p2 = intch[i], intch[(i+1)%3]
        #get the line passing through p1 and p2
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        q = p1[1] - m * p1[0]
        #draw the full line passing through p1 and p2
        ax.plot([0, MAP_SIZE_M[0]], [q, m*MAP_SIZE_M[0]+q], colors[i]+'-', markersize=3)

    # for each section, if there are points, select one at random and add 
    # it to the track points, at the corresponding section
    split_and_poins = {}
    for i in range(3):
        candidates = np.where(section == i)[0]
        if len(candidates) > 0: #if there are points in the section
            idx = np.random.choice(candidates) #select one at random
            p = 0.5 * (intch[i] + intch[(i+1)%3]) #add it to the track points
            split_and_poins[idx] = p #add the point to the dictionary
    #sort by index
    split_and_poins = dict(sorted(split_and_poins.items()))

    for j, (i, p) in enumerate(split_and_poins.items()):
        track_points = np.vstack((track_points[0:i+j], p, track_points[i+j:]))
    #add the first point to the end of the track
    track_points = np.vstack((track_points, track_points[0]))

    #draw the track points
    ax.plot(track_points[:,0], track_points[:,1], 'k--', markersize=3)
    
    #save plot in the tmp folder
    plt.savefig('tmp/track1.png')
    
    #create a new figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    #plot the track points
    ax.plot(track_points[:,0], track_points[:,1], 'ko', markersize=5)
    ax.set_xlim(0, MAP_SIZE_M[0])
    ax.set_ylim(0, MAP_SIZE_M[1])
    ax.grid(True)
    ax.set_xticks(np.arange(0, MAP_SIZE_M[0], 2))
    ax.set_yticks(np.arange(0, MAP_SIZE_M[1], 2))
    ax.set_aspect('equal')
    #draw the track points
    ax.plot(track_points[:,0], track_points[:,1], 'k--', markersize=3)


    # # interpolate the track points using a spline
    # from scipy.interpolate import splprep, splev
    # tck, u = splprep(track_points.T, s=0, per=True)
    # u_new = np.linspace(u.min(), u.max(), 1000)
    # x_new, y_new = splev(u_new, tck)
    # track = np.array([x_new, y_new]).T


    # # interpolate the points suing bezier curves
    # cps = get_bezier_parameters(track_points[:,0], track_points[:,1], degree=3)
    # track = np.array(bezier_curve(cps, nTimes=1000)).T

    #interpolate using clothoids
    # get the thetas for each point
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
        pm = 0.5 * (pa + pb)
        thetas[i] =  np.arctan2(pm[1]-cp[1], pm[0]-cp[0])

    track_points = np.hstack((track_points, thetas.reshape(-1,1)))
    from pyclothoids import Clothoid
    track = []
    sl = 0.01 # [m] step length
    for i in range(len(track_points)):
        xc,yc,thc = track_points[i]
        xf,yf,thf = track_points[(i+1)%len(track_points)]
        cloth_path = Clothoid.G1Hermite(xc,yc,thc,xf,yf,thf)
        [xs,ys] = cloth_path.SampleXY(int(cloth_path.length/sl))
        for x, y in zip(xs, ys): track.append([x, y])
    track = np.array(track)
    
    
    ax.plot(track[:,0], track[:,1], 'k-<', markersize=3)

    #save plot in the tmp folder
    plt.savefig('tmp/track2.png')
    plt.close('all')
    


if __name__ == '__main__':

    cv.namedWindow('track1', cv.WINDOW_NORMAL)
    cv.namedWindow('track2', cv.WINDOW_NORMAL)
    cv.resizeWindow('track1', 800, 600)
    cv.resizeWindow('track2', 800, 600)

    for i in range(1000):
        create_track()
        
        #load the image
        track1 = cv.imread('tmp/track1.png')
        track2 = cv.imread('tmp/track2.png')
        cv.imshow('track1', track1)
        cv.imshow('track2', track2)

        if cv.waitKey(0) == 27:
            break

    cv.destroyAllWindows()





# if __name__ == '__main__':
    
#     #create a random color map
#     new_map = create_track()
#     change_track(new_map) #change the map

#     run_ros_master('bare_car_with_map.launch') #run the simulator
#     wait_for_ros_startup() #wait for the simulator to start

#     while not ros.is_shutdown():
#         print('running', end='\r')
#         sleep(0.2)

#     kill_ros_master() #kill the simulator
