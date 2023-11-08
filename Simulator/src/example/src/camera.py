#!/usr/bin/env python3
import rospy, cv2, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraHandler():
    # ===================================== INIT==========================================
    def __init__(self):
        """
        Creates a bridge for converting the image from Gazebo image intro OpenCv image
        """
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        rospy.init_node('CAMnod', anonymous=True)
        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.callback)
        rospy.spin()

    def callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazsbo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.imshow("Frame preview", self.cv_image)
        key = cv2.waitKey(1)
    
            
if __name__ == '__main__':
    try:
        nod = CameraHandler()
    except rospy.ROSInterruptException:
        # When the program is interrupted (e.g. shutdown) we'll print a message
        rospy.loginfo("Camera log  node terminated.")
        cv2.destroyAllWindows()
        pass