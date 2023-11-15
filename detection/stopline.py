import cv2 # Import the OpenCV library to enable computer vision
import numpy as np # Import the NumPy scientific computing library
import matplotlib.pyplot as plt # Used for plotting and error checking
 
# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Implementation of the Lane class 
 
#filename = 'advancestopline.jpg'
#filename = 'Stop-Bars-and-Center-Lines.jpg'
#filename = 'advance-yield.jpg'
#filename = 'stop_lines.jpg'
#filename = 'test.jpg'
filename = '1.jpg'

 
class StopLine:
  """
  Represents a lane on a road.
  """

  def __init__(self, orig_frame):
    """
      Default constructor
         
    :param orig_frame: Original camera image (i.e. frame)
    """
    self.orig_frame = orig_frame
 
    # This will hold an image with the lane lines       
    self.lane_line_markings = None
 
    # This will hold the image after perspective transformation
    self.warped_frame = None
    self.transformation_matrix = None
    self.inv_transformation_matrix = None
 
    # (Width, Height) of the original video frame (or image)
    self.orig_image_size = self.orig_frame.shape[::-1][1:]
 
    width = self.orig_image_size[0]
    height = self.orig_image_size[1]
    self.width = width
    self.height = height
     
    # Four corners of the trapezoid-shaped region of interest
    # You need to find these corners manually.
    # <++>
    self.roi_points = np.float32([ 
      (self.width//12, 3*self.height//5), # Top-left corner self.width//12, 2*self.height//3), # Top-left corner
      (self.width//12, self.height - self.height//24), # Bottom-left corner            
      (self.width - self.width//12, self.height - self.height//24), # Bottom-right corner
      (self.width - self.width//12, 3*self.height//5) # Top-right corner (self.width - self.width//12, 2*self.height//3) # Top-right corner
    ])
    #self.roi_points = np.float32([
    #  (0, 2*self.height//3), # Top-left corner
    #  (0, self.height), # Bottom-left corner            
    #  (self.width, self.height), # Bottom-right corner
    #  (self.width, 2*self.height//3) # Top-right corner
    #])

    # The desired corner locations  of the region of interest
    # after we perform perspective transformation.
    # Assume image width of 600, padding == 150.
    #self.padding = int(0.25 * self.width) # padding from side of the image in pixels
    self.padding = int(0 * self.width) # padding from side of the image in pixels
    self.desired_roi_points = np.float32([
      [self.padding, 0], # Top-left corner
      [self.padding, self.height], # Bottom-left corner         
      [self.width-self.padding, self.height], # Bottom-right corner
      [self.width-self.padding, 0] # Top-right corner
    ]) 
         
    # Histogram that shows the white pixel peaks for lane line detection
    self.histogram = None
         
    # Sliding window parameters
    # <++>
    self.no_of_windows = 50
    self.margin = int((1/8) * self.height)  # Window height is +/- margin
    self.minpix = int((1/24) * self.height)  # Min no. of pixels to recenter window
         
    # Best fit polynomial lines for left line and right line of the lane
    self.line_fit = None
    self.stop_line_inds = None
    self.plotx = None
    self.line_fity = None
    self.linex = None
    self.liney = None
         
    # Pixel parameters for x and y dimensions
    self.YM_PER_PIX = 10.0 / 1000 # meters per pixel in y dimension
    self.XM_PER_PIX = 3.7 / 781 # meters per pixel in x dimension
         
    # Radii of curvature and offset
    self.left_curvem = None
    self.right_curvem = None
    self.center_offset = None
 
  def calculate_histogram(self,frame=None,plot=True):
    """
    Calculate the image histogram to find peaks in white pixel count
         
    :param frame: The warped image
    :param plot: Create a plot if True
    """
    if frame is None:
      frame = self.warped_frame
             
    # Generate the histogram
    self.histogram = np.sum(frame[:,frame.shape[1]//3:2*frame.shape[1]//3], axis=1)
 
    if plot:
         
      # Draw both the image and the histogram
      figure, (ax1, ax2) = plt.subplots(2,1) # 2 row, 1 columns
      figure.set_size_inches(10, 5)
      ax1.imshow(frame, cmap='gray')
      ax1.set_title("Warped Binary Frame")
      ax2.plot(self.histogram)
      ax2.set_title("Histogram Peaks")
      plt.show()
             
    return self.histogram

  def get_lane_line_indices_sliding_windows(self, plot=False):
    """
    Get the indices of the lane line pixels using the 
    sliding windows technique.
         
    :param: plot Show plot or not
    :return: Best fit lines for the left and right lines of the current lane 
    """
    # Sliding window width is +/- margin
    margin = self.margin
 
    frame_sliding_window = self.warped_frame.copy()
 
    # Set the height of the sliding windows
    window_width = self.warped_frame.shape[1]//self.no_of_windows       
 
    # Find the x and y coordinates of all the nonzero 
    # (i.e. white) pixels in the frame. 
    nonzero = self.warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1]) 
         
    # Store the pixel indices for the left and right lane lines
    stop_line_inds = []
         
    # Current positions for pixel indices for each window,
    # which we will continue to update
    y_base = self.histogram_peak()
    y_current = y_base
 
    # Go through one window at a time
    no_of_windows = self.no_of_windows
         
    for window in range(no_of_windows):
       
      # Identify window boundaries in x and y (and right and left)
      # <++>
      win_x_low = self.warped_frame.shape[1] - (window + 1) * window_width
      win_x_high = self.warped_frame.shape[1] - window * window_width
      win_y_low = y_current - margin
      win_y_high = y_current + margin
      cv2.rectangle(frame_sliding_window,
                    (win_x_low,win_y_low),
                    (win_x_high,win_y_high),
                    (255,255,255), 2)

      # Identify the nonzero pixels in x and y within the window
      good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_x_low) & (
                           nonzerox < win_x_high)).nonzero()[0]
                                                         
      # Append these indices to the lists
      stop_line_inds.append(good_inds)
         
      # If you found > minpix pixels, recenter next window on mean position
      minpix = self.minpix
      if len(good_inds) > minpix:
        y_current = int(np.mean(nonzeroy[good_inds]))
                     
    # Concatenate the arrays of indices
    stop_line_inds = np.concatenate(stop_line_inds)
 
    # Extract the pixel coordinates for the left and right lane lines
    linex = nonzerox[stop_line_inds]
    liney = nonzeroy[stop_line_inds] 
 
    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    line_fit = np.polyfit(linex, liney, 1)
    #right_fit = np.polyfit(righty, rightx, 2) 
         
    self.line_fit = line_fit
 
    if plot==True:
         
      # Create the x and y values to plot on the image  
      plotx = np.linspace(0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
      line_fity = line_fit[0]*plotx + line_fit[1]
 
      # Generate an image to visualize the result
      out_img = np.dstack((frame_sliding_window,
                           frame_sliding_window,
                           (frame_sliding_window))) * 255
             
      # Add color to the left line pixels and right line pixels
      out_img[nonzeroy[stop_line_inds], nonzerox[stop_line_inds]] = [255, 0, 0]
      #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
                 
      # Plot the figure with the sliding windows
      figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(frame_sliding_window, cmap='gray')
      ax3.imshow(out_img)
      ax3.plot(plotx, line_fity, color='yellow')
      ax1.set_title("Original Frame")  
      ax2.set_title("Warped Frame with Sliding Windows")
      ax3.set_title("Detected Lane Lines with Sliding Windows")
      plt.show()        
             
    return self.line_fit
 
  def get_line_markings(self, frame=None):
    """
    Isolates lane lines.
   
      :param frame: The camera frame that contains the lanes we want to detect
    :return: Binary (i.e. black and white) image containing the lane lines.
    """
    if frame is None:
      frame = self.orig_frame
             
    # Convert the video frame from BGR (blue, green, red) 
    # color space to HLS (hue, saturation, lightness).
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
 
    ################### Isolate possible lane line edges ######################
         
    # Perform Sobel edge detection on the L (lightness) channel of 
    # the image to detect sharp discontinuities in the pixel intensities 
    # along the x and y axis of the video frame.             
    # sxbinary is a matrix full of 0s (black) and 255 (white) intensity values
    # Relatively light pixels get made white. Dark pixels get made black.
    _, sxbinary = cv2.threshold(hls[:, :, 1], 160, 255, cv2.THRESH_BINARY)
    sxbinary = cv2.GaussianBlur(sxbinary, (3, 3), 0) # Reduce noise
         
    # 1s will be in the cells with the highest Sobel derivative values
    # (i.e. strongest lane line edges)
    # <++>
    sxbinary = mag_thresh(sxbinary, sobel_kernel=3, thresh=(0, 255))
 
    ######################## Isolate possible lane lines ######################
   
    # Perform binary thresholding on the S (saturation) channel 
    # of the video frame. A high saturation value means the hue color is pure.
    # We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
    # and have high saturation channel values.
    # s_binary is matrix full of 0s (black) and 255 (white) intensity values
    # White in the regions with the purest hue colors (e.g. >80...play with
    # this value for best results).
    s_channel = hls[:, :, 2] # use only the saturation channel data
    # <++>
    _, s_binary = cv2.threshold(s_channel, 50, 255, cv2.THRESH_BINARY_INV)
    s_binary = cv2.GaussianBlur(s_binary, (3, 3), 0) # Reduce noise
     
    # Perform binary thresholding on the R (red) channel of the 
        # original BGR video frame. 
    # r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
    # White in the regions with the richest red channel values (e.g. >120).
    # Remember, pure white is bgr(255, 255, 255).
    # Pure yellow is bgr(0, 255, 255). Both have high red channel values.
    _, r_thresh = cv2.threshold(frame[:, :, 2], 170, 255, cv2.THRESH_BINARY)
    r_thresh = cv2.GaussianBlur(r_thresh, (3, 3), 0) # Reduce noise
 
    # Lane lines should be pure in color and have high red channel values 
    # Bitwise AND operation to reduce noise and black-out any pixels that
    # don't appear to be nice, pure, solid colors (like white or yellow lane 
    # lines.)       
    rs_binary = cv2.bitwise_and(s_binary, r_thresh)
 
    ### Combine the possible lane lines with the possible lane line edges ##### 
    # If you show rs_binary visually, you'll see that it is not that different 
    # from this return value. The edges of lane lines are thin lines of pixels.
    self.lane_line_markings = cv2.bitwise_and(rs_binary, sxbinary.astype(
                              np.uint8))    
    return self.lane_line_markings
         
  def histogram_peak(self):
    """
    Get the left and right peak of the histogram
 
    Return the x coordinate of the left histogram peak and the right histogram
    peak.
    """
    peak = np.argmax(self.histogram)
 
    # (y coordinate of the peak)
    return peak
         
  def perspective_transform(self, frame=None, plot=False):
    """
    Perform the perspective transform.
    :param: frame Current frame
    :param: plot Plot the warped image if True
    :return: Bird's eye view of the current lane
    """
    if frame is None:
      frame = self.lane_line_markings
             
    # Calculate the transformation matrix
    self.transformation_matrix = cv2.getPerspectiveTransform(
      self.roi_points, self.desired_roi_points)
 
    # Calculate the inverse transformation matrix           
    self.inv_transformation_matrix = cv2.getPerspectiveTransform(
      self.desired_roi_points, self.roi_points)
 
    # Perform the transform using the transformation matrix
    self.warped_frame = cv2.warpPerspective(
      frame, self.transformation_matrix, self.orig_image_size, flags=(
     cv2.INTER_LINEAR)) 
 
    # Convert image to binary
    (thresh, binary_warped) = cv2.threshold(
      self.warped_frame, 127, 255, cv2.THRESH_BINARY)           
    self.warped_frame = binary_warped
 
    # Display the perspective transformed (i.e. warped) frame
    if plot == True:
      warped_copy = self.warped_frame.copy()
      warped_plot = cv2.polylines(warped_copy, np.int32([
                    self.desired_roi_points]), True, (147,20,255), 3)
 
      # Display the image
      cv2.imshow('Warped Image', warped_plot)
             
      # Press any key to stop
      cv2.waitKey(0)
 
      # cv2.destroyAllWindows()   
             
    return self.warped_frame        
     
  def plot_roi(self, frame=None, plot=False):
    """
    Plot the region of interest on an image.
    :param: frame The current image frame
    :param: plot Plot the roi image if True
    """
    if plot == False:
      return
             
    if frame is None:
      frame = self.orig_frame.copy()
 
    # Overlay trapezoid on the frame
    this_image = cv2.polylines(frame, np.int32([
      self.roi_points]), True, (147,20,255), 3)
 
    # Display the image
    while(1):
      cv2.imshow('ROI Image', this_image)
             
      # Press any key to stop
      if cv2.waitKey(0):
        break
 
    cv2.destroyAllWindows()


def binary_array(array, thresh, value=0):
  """
  Return a 2D binary array (mask) in which all pixels are either 0 or 1
     
  :param array: NumPy 2D array that we want to convert to binary values
  :param thresh: Values used for thresholding (inclusive)
  :param value: Output value when between the supplied threshold
  :return: Binary 2D array...
           number of rows x number of columns = 
           number of pixels from top to bottom x number of pixels from
             left to right 
  """
  if value == 0:
    # Create an array of ones with the same shape and type as 
    # the input 2D array.
    binary = np.ones_like(array) 
         
  else:
    # Creates an array of zeros with the same shape and type as 
    # the input 2D array.
    binary = np.zeros_like(array)  
    value = 1
 
  # If value == 0, make all values in binary equal to 0 if the 
  # corresponding value in the input array is between the threshold 
  # (inclusive). Otherwise, the value remains as 1. Therefore, the pixels 
  # with the high Sobel derivative values (i.e. sharp pixel intensity 
  # discontinuities) will have 0 in the corresponding cell of binary.
  binary[(array >= thresh[0]) & (array <= thresh[1])] = value
 
  return binary
 
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
  """
  Implementation of Sobel edge detection
 
  :param image: 2D or 3D array to be blurred
  :param sobel_kernel: Size of the small matrix (i.e. kernel) 
                       i.e. number of rows and columns
  :return: Binary (black and white) 2D mask image
  """
  # Get the magnitude of the edges that are vertically aligned on the image
  sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
         
  # Get the magnitude of the edges that are horizontally aligned on the image
  sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))
 
  # Find areas of the image that have the strongest pixel intensity changes
  # in both the x and y directions. These have the strongest gradients and 
  # represent the strongest edges in the image (i.e. potential lane lines)
  # mag is a 2D array .. number of rows x number of columns = number of pixels
  # from top to bottom x number of pixels from left to right
  mag = np.sqrt(sobelx ** 2 + sobely ** 2)
 
  # Return a 2D array that contains 0s and 1s   
  return 255 * binary_array(mag, thresh)
 
def sobel(img_channel, orient='x', sobel_kernel=3):
  """
  Find edges that are aligned vertically and horizontally on the image
     
  :param img_channel: Channel from an image
  :param orient: Across which axis of the image are we detecting edges?
  :sobel_kernel: No. of rows and columns of the kernel (i.e. 3x3 small matrix)
  :return: Image with Sobel edge detection applied
  """
  # cv2.Sobel(input image, data type, prder of the derivative x, order of the
  # derivative y, small matrix used to calculate the derivative)
  if orient == 'x':
    # Will detect differences in pixel intensities going from 
        # left to right on the image (i.e. edges that are vertically aligned)
    sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  if orient == 'y':
    # Will detect differences in pixel intensities going from 
    # top to bottom on the image (i.e. edges that are horizontally aligned)
    sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
 
  return sobel
 

def detect_angle(original_frame=None, plot=False):
  frame = original_frame.copy()
  XL_OFFSET = 50
  XR_OFFSET = 50
  frame = frame[:,XL_OFFSET:-XR_OFFSET,:]

  # Create a Lane object
  lane_obj = StopLine(orig_frame=frame)
 
  # Perform thresholding to isolate lane lines
  _ = lane_obj.get_line_markings()
 
  # Plot the region of interest on the image
  lane_obj.plot_roi(plot=False)
 
  # Perform the perspective transform to generate a bird's eye view
  # If Plot == True, show image with new region of interest
  warped_frame = lane_obj.perspective_transform(plot=False)

  #extract lines
  img = warped_frame
  # # img = cv2.resize(img, (100,100))
  # print(img.shape)
  # cv2.namedWindow('lines', cv2.WINDOW_NORMAL)
  # cv2.imshow('lines',img)
  # cv2.waitKey(0) 
  lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=30, minLineLength=80, maxLineGap=5)
  angles = []
  if lines is None:
    return 0.0
  for line in lines:
    for x1,y1,x2,y2 in line:
      angle = np.arctan2(y2-y1,x2-x1)
      angles.append(angle)
    
  angles = np.array(angles)
  angle_median = np.median(angles)
  #keep only valid angles
  angles = angles[np.abs(angles-angle_median)<np.deg2rad(5.0)]
  final_angle = np.mean(angles)

  if np.abs(final_angle) > np.pi/4:
    return 0.0
  elif final_angle > np.deg2rad(31.0):
    return np.deg2rad(31.0)
  elif final_angle < np.deg2rad(-31.0):
    return np.deg2rad(-31.0)
  return final_angle


def main():
     
  # Load a frame (or image)
  original_frame = cv2.imread(filename)
 
  print(detect_angle(original_frame=original_frame, plot=False))
     
if __name__ == '__main__':
  main()
