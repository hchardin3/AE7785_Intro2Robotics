import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from joblib import load
import cv2
import numpy as np
from skimage.feature import hog
from collections import Counter


def preprocess(img):
    image_size = (256, 256)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for green in HSV
    lower_green = np.array([30, 40, 40])  # Adjust these values as needed
    upper_green = np.array([90, 255, 255])  # Adjust these values as needed

    lower_blue = np.array([90, 140, 10])  
    upper_blue = np.array([130, 255, 250])

    lower_red1 = np.array([0, 100, 100])  # increase saturation and value for lighter reds
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])  # decrease hue for darker reds
    upper_red2 = np.array([179, 255, 255])

    # Create a mask for green
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Combine the individual color masks into one large mask
    combined_mask = cv2.bitwise_or(mask_blue, mask_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red)

    # Use the combined mask to isolate the blue, green, or red pixels in the original image
    isolated_colors = cv2.bitwise_and(img, img, mask=combined_mask)

    filtered_bgr = cv2.cvtColor(isolated_colors, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY)

    mask = gray > 0
    gray[mask] = 255


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


    contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    list_of_contours = []
    if hierarchy is not None:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 600:
                x, y, w, h = cv2.boundingRect(contour)
                list_of_contours.append([x, y, w, h])

    if len(list_of_contours) == 1:
        x, y, w, h = list_of_contours[0]
        x_min = max(x, 0)
        y_min = max(y, 0)
        x_max = min(x + w, gray.shape[1])
        y_max = min(y + h, gray.shape[0])

        final = gray[y_min:y_max, x_min:x_max]
    else:
        final = gray


    resized_gray = cv2.resize(final, image_size)

    hog_features = hog(resized_gray, orientations=12, pixels_per_cell=(2, 2), cells_per_block=(1, 1))

    return hog_features

class SignVision(Node):

    def __init__(self):
        super().__init__('sign_vision')
        self.clf = load('svm_class_2024_nb.joblib')
        self.detection_on = False
        self.detection_count = 0
        self.detection_results = [0, 0, 0, 0, 0]

        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(depth=5)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        #Declare that the minimal_video_subscriber node is subcribing to the /camera/image/compressed topic.
        self._video_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.camera_callback,
            10)
        self._video_subscriber

        self.laser_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            image_qos_profile)
        
        self.laser_subscriber
        

        self.sign_pub = self.create_publisher(
            Int32,
            'sign_detected',
            10
        )
        
        self.bridge = CvBridge()
        
        self.laser_subscriber

        self.laser_data = None

        self.distance_treshold = 0.3

    def camera_callback(self, msg):
        if not self.detection_on:
            return None
        
        elif self.detection_on:
            if self.detection_count < 5:
                imgHSV = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

                self.new_hog = preprocess(imgHSV)

                self.detection_results[self.detection_count] = self.clf.predict(self.new_hog)

                self.detection_count += 1

            else:
                self.detection_count = 0

                counter = Counter(self.detection_results)

                sign_detected, _ = counter.most_common(1)[0]

                self.sign_pub.publish(sign_detected)

                self.detection_results = [0, 0, 0, 0, 0]

                self.detection_on = False
        
        return



    def laser_callback(self, msg):
        self.laser_data = msg

        if self.laser_data.range[0] < self.distance_treshold:
            self.detection_on = True
        

def main():
    rclpy.init() #init routine needed for ROS2.
    sign_vision = SignVision() #Create class object to be used.

    rclpy.spin(sign_vision) # Trigger callback processing.		

    #Clean up and shutdown.
    sign_vision.destroy_node()  
    rclpy.shutdown()

if __name__ == "__main__":
    main()