import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from joblib import load
import cv2
import numpy as np
from skimage.feature import hog
from collections import Counter
from transformations import quaternion_from_euler, euler_from_quaternion
import math


class RobotControl(Node):
    def __init__(self):
        
        self.sign_sub = self.create_subscription(
            Int32,
            'sign_detected',
            self.sign_callback,
            10
        )

        self.await_sign = False
        self.rot_dir = 0
        self.goal_reached = False

        image_qos_profile = QoSProfile(depth=5)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        self.laser_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            image_qos_profile)
        
        self.laser_subscriber

        self.control_publisher = self.create_publisher(
            Twist, 
            '/cmd_vel', 
            10
            )
        
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            10
            )

        self.initial_yaw = None
        self.target_yaw = None
        self.is_turning = False
        self.move_forward = False

        self.sign_dictionnary = {
            0: "Empty",
            1: "Left",
            2: "Right",
            3: "Do Not Enter",
            4: "Stop",
            5: "Goal"
        }

        self.distance_treshold = 0.3
    
    def sign_callback(self, msg):
        if self.await_sign:
            print("Message detected", self.sign_dictionnary[msg.data])
            sign = msg.data
            if sign == 1:
                self.rot_dir = 1
                self.initial_yaw = self.current_yaw  # Capture the current yaw as the starting point
                self.target_yaw = (self.initial_yaw + math.pi / 2) % (2 * math.pi)  # Calculate target yaw
                self.is_turning = True 
                self.turn()
            elif sign == 2:
                self.rot_dir = 2
                self.initial_yaw = self.current_yaw  # Capture the current yaw as the starting point
                self.target_yaw = (self.initial_yaw - math.pi / 2) % (2 * math.pi)  # Calculate target yaw
                self.is_turning = True 
                self.turn()
            elif sign == 5:
                print("Goal Reached, Mankind Shall Die")
                self.goal_reached = True
                twist = Twist()
                twist.angular.z = 0.2
                self.control_publisher.publish(twist)
            self.await_sign = False

    def turn(self):
        """Make the robot do a 90 degree turn, to the left if direction = 1 and the right if it is 2"""
        twist = Twist()
        angular_speed = 0.5  # Adjust the turning speed as necessary
        if self.rot_dir == 1:
            twist.angular.z = angular_speed
        elif self.rot_dir == 2:
            twist.angular.z = -angular_speed
        self.control_publisher.publish(twist)

    def go_straight(self, direction):
        """Make the robot go straight"""
        self.move_forward = True
        twist = Twist()
        twist.linear.x = 0.2
        self.control_publisher.publish(twist)

    def stop_moving(self):
        # Stop the robot
        twist = Twist()
        self.control_publisher.publish(twist)
        self.is_turning = False
        self.rot_dir = 0
        self.move_forward = False
        print("Turn completed")


    def odom_callback(self, msg):
        # Extract the current yaw from the odometry quaternion
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_yaw = yaw

        if self.is_turning:
            # Check if we have reached or exceeded the target yaw
            yaw_difference = (yaw - self.initial_yaw + 2 * math.pi) % (2 * math.pi)
            if self.rot_dir == 1 and yaw_difference >= math.pi / 2 or self.rot_dir == 2 and yaw_difference <= 3 * math.pi / 2:
                self.stop_moving()

                if not self.goal_reached:
                    self.move_forward()
                
    
    def laser_callback(self, msg):
        if msg.range[0] < self.distance_treshold:
            self.move_forward = False

            self.stop_moving()

            self.await_sign = True



def main():
    rclpy.init() #init routine needed for ROS2.
    robot_control = RobotControl() #Create class object to be used.

    rclpy.spin(robot_control) # Trigger callback processing.		

    #Clean up and shutdown.
    robot_control.destroy_node()  
    rclpy.shutdown()

if __name__ == "__main__":
    main()