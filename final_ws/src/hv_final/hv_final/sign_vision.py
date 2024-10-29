import rclpy
import os
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import Int32, String
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
        package_share_directory = get_package_share_directory('hv_final')
        joblib_path = os.path.join(package_share_directory, 'data', 'svm_class_2024_1.joblib')
        self.clf = load(joblib_path)
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

        qos_profile = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=10)

        self.vision_subscriber = self.create_subscription(
                String,
                '/prediction_request',
                self.vision_callback,
                qos_profile
            )
        
        self.vision_subscriber2 = self.create_subscription(
                String,
                '/prediction_request2',
                self.vision_callback,
                qos_profile
            )
        
        self.vision_subscriber3 = self.create_subscription(
                String,
                '/prediction_request3',
                self.vision_callback,
                qos_profile
            )
        
        self.vision_subscriber4 = self.create_subscription(
                String,
                '/prediction_request4',
                self.vision_callback,
                qos_profile
            )

        self.vision_subscriber5 = self.create_subscription(
                String,
                '/prediction_request5',
                self.vision_callback,
                qos_profile
            )
        
        
        
        self.vision_subscriber
        
        self.laser_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            image_qos_profile)
            
        

        self.sign_pub = self.create_publisher(
            Int32,
            '/sign_detected',
            10
        )
        
        self.bridge = CvBridge()

    def camera_callback(self, msg):
       #self.get_logger().info("First cam message")
       if self.detection_on:
            if self.detection_count < 5:

                self.get_logger().info(f"Detection number  {self.detection_count}")
                imgHSV = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

                hog_features = preprocess(imgHSV)

                hog_features = hog_features.reshape(1, -1)

                self.detection_results[self.detection_count] = self.clf.predict(hog_features)[0]

                self.detection_count += 1

                self.get_logger().info(f"Detection number  {self.detection_count - 1} done")

            else:
                self.detection_count = 0

                counter = Counter(self.detection_results)

                sign_detected, _ = counter.most_common(1)[0]

                self.get_logger().info(f"Sendeing detection {sign_detected}")

                self.sign_pub.publish(Int32(data=int(sign_detected)))

                

                self.detection_results = [0, 0, 0, 0, 0]

                self.detection_on = False
        

    def vision_callback(self, msg):
        self.get_logger().info("Received a message on /prediction_request.")
        if msg.data == "start":
            self.detection_on = True
            self.get_logger().info("Detection started.")
        elif msg.data == "stop":
            self.detection_on = False
            self.get_logger().info("Detection stopped.")

    def laser_callback(self, msg):
        letsgo = False
        if msg.ranges[0] < 0.5 and letsgo:
            self.move_forward = False

            self.get_logger().info(f"Let's detect anyway")

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