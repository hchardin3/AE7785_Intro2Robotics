import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, String
from cv_bridge import CvBridge
from joblib import load
import cv2
import numpy as np
from skimage.feature import hog
from collections import Counter
from transformations import quaternion_from_euler, euler_from_quaternion
import math
import time


class RobotControl(Node):
    def __init__(self):
        super().__init__('robot_control')
        self.sign_sub = self.create_subscription(
            Int32,
            '/sign_detected',
            self.sign_callback,
            10
        )

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

        self.laser_subscriber2 = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            image_qos_profile
            )
        
        self.laser_subscriber2

        self.control_publisher = self.create_publisher(
            Twist, 
            '/cmd_vel', 
            10
            )
        
        qos_profile = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=10)
        
        self.vision_publisher = self.create_publisher(
                String,
                '/prediction_request',
                qos_profile
            )
        
        # self.vision_publisher2 = self.create_publisher(
        #         String,
        #         '/prediction_request2',
        #         qos_profile
        #     )
        
        # self.vision_publisher3 = self.create_publisher(
        #         String,
        #         '/prediction_request3',
        #         qos_profile
        #     )

        # self.vision_publisher4 = self.create_publisher(
        #         String,
        #         '/prediction_request4',
        #         qos_profile
        #     )
        

        # self.vision_publisher5 = self.create_publisher(
        #         String,
        #         '/prediction_request5',
        #         qos_profile
        #     )
        
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

        self.distance_treshold = 0.5
        self.await_next_detection = True
        self.demand_next_detection = True


        # PID parameters
        self.kp = 0.6
        self.ki = 0.1
        self.kd = 0.05
        self.previous_error = 0
        self.integral = 0
        self.desired_accuracy = 0.01  # Accuracy within 5 degrees in radians

        self.timer = self.create_timer(1, self.go_straight_timer_callback)

    def go_straight_timer_callback(self):
        self.go_straight()
        self.timer.cancel()
    
    def sign_callback(self, msg):
        self.get_logger().info(f"dddddddddddddddddddd")
        if self.await_next_detection:
            self.get_logger().info(f"Message detected: {self.sign_dictionnary[msg.data]}")
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
            elif sign == 3 or sign == 4:
                self.rot_dir = 3
                self.initial_yaw = self.current_yaw  # Capture the current yaw as the starting point
                self.target_yaw = (self.initial_yaw - math.pi) % (2 * math.pi)  # Calculate target yaw
                self.is_turning = True 
                self.turn()
            elif sign == 5:
                print("Goal Reached, Mankind Shall Die")
                self.goal_reached = True
                twist = Twist()
                twist.angular.z = 1.0
                self.control_publisher.publish(twist)
            self.await_next_detection = False

    def turn(self):
        """Make the robot do a 90 degree turn, to the left if direction = 1 and the right if it is 2"""
        twist = Twist()
        angular_speed = 0.5  # Adjust the turning speed as necessary
        if self.rot_dir == 1:
            twist.angular.z = angular_speed
        elif self.rot_dir == 2 or self.rot_dir == 3:
            twist.angular.z = -angular_speed
        self.control_publisher.publish(twist)

    def go_straight(self):
        """Make the robot go straight"""
        self.get_logger().info(f"Let's go straight")
        self.move_forward = True
        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = 0.0
        self.control_publisher.publish(twist)
        self.await_next_detection = True
        self.demand_next_detection = True

    def stop_moving(self):
        # Stop the robot
        twist = Twist()
        self.control_publisher.publish(twist)
        time.sleep(1.0)
        self.is_turning = False
        self.rot_dir = 0
        self.move_forward = False


    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        self.current_yaw = orientation

        if self.is_turning:
            yaw_target = self.target_yaw
            yaw_error = (yaw_target - self.current_yaw) % (2 * math.pi)
            if yaw_error > math.pi:
                yaw_error -= 2 * math.pi

            # Simple anti-windup: Cap the integral to not accumulate too much error
            integral_cap = 0.1  # Adjust based on your application
            self.integral = max(min(self.integral + yaw_error, integral_cap), -integral_cap)
            derivative = yaw_error - self.previous_error
            self.previous_error = yaw_error
            angular_speed = (self.kp * yaw_error  + self.kd * derivative + self.ki * self.integral) * (abs(math.sin(yaw_error))**(0.1) + 0.5)

            twist = Twist()
            twist.angular.z = angular_speed
            self.control_publisher.publish(twist)

            # Debugging output
            # self.get_logger().info(f'Current Yaw: {orientation:.2f}, Target Yaw: {yaw_target:.2f}, Error: {yaw_error:.2f}, Cmd: {angular_speed:.2f}')

            if abs(yaw_error) < self.desired_accuracy:
                self.get_logger().info("Turn completed.")
                self.stop_moving()
                self.is_turning = False
                if not self.goal_reached:
                    self.move_forward = True
                    self.go_straight()
    
    def laser_callback(self, msg):
        # self.get_logger().info(f"LIDAR LIDAR LIDAR")
        if msg.ranges[0] < self.distance_treshold and self.demand_next_detection:
            self.move_forward = False
            self.demand_next_detection = False

            self.stop_moving()

            self.vision_publisher.publish(String(data="start"))
            # self.vision_publisher2.publish(String(data="start"))
            # self.vision_publisher3.publish(String(data="start"))
            # self.vision_publisher4.publish(String(data="start"))
            # self.vision_publisher5.publish(String(data="start"))

            self.get_logger().info(f"Demanding new vision")

            self.await_next_detection = True

            



def main():
    rclpy.init() #init routine needed for ROS2.
    robot_control = RobotControl() #Create class object to be used.

    rclpy.spin(robot_control) # Trigger callback processing.		

    #Clean up and shutdown.
    robot_control.destroy_node()  
    rclpy.shutdown()

if __name__ == "__main__":
    main()