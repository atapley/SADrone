#!/usr/bin/env python

import roslib

roslib.load_manifest('fp_nav')
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tensorflow as tf
import time

# Hyperparameters
MODEL = "RedObstacle"
META_PATH = "./models/" + MODEL + "/model.cptk.meta"
VARIABLE_PATH = "./models/" + MODEL + "/model.cptk.index"
ROBOT_NAME = "****"
RATE = 10
TRAINING_DIMS = (320, 240)
SPEED = 1


class choose_action:

    def __init__(self):
        tf.reset_default_graph()
        self.graph = tf.train.import_meta_graph(META_PATH)
        self.cv_bridge = CvBridge()
        self.sess = tf.Session()
        self.graph.restore(self.sess, VARIABLE_PATH)

        self.joysub = rospy.Subscriber(ROBOT_NAME + '/joy', Joy, self.joy_callback, queue_size=1)
        self.imagesub = rospy.Subscriber(ROBOT_NAME + '/camera/rgb/image_raw', Image, self.image_callback, queue_size=1)

        self.pub = rospy.Publisher(ROBOT_NAME + '/cmd_vel', Twist, queue_size=1)
        self.rate = rospy.Rate(RATE)

        self.image = None
        self.joy = None
        self.running = 0
        self.count = 0

        self.g_lower_bound = np.array([75, 50, 0])
        self.g_upper_bound = np.array([130, 255, 255])
        self.g_kernel = np.ones((5, 5), np.uint8)

        self.o_lower_bound = np.array([15, 100, 20])
        self.o_upper_bound = np.array([28, 255, 200])
        self.o_kernel = np.ones((5, 5), np.uint8)

    # Set the image to the image_msg
    def image_callback(self, image_msg):
        if not self.running:
            self.image = image_msg
            self.callback()

    # Set the joy to the joy_msg
    def joy_callback(self, joy_msg):
        if not self.running:
            self.joy = joy_msg
            self.callback()

    # If button is pushed, then run through model and publish
    def callback(self):
        if (self.joy is not None and self.image is not None and self.joy.buttons[0] == 1 and not self.running):
            self.running = 1
            # Convert the camera image to a cv image
            try:
                cv_image = self.cv_bridge.imgmsg_to_cv2(self.image, "bgr8")
            except CvBridgeError as e:
                print(e)

            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Mask and denoise the goal in the image
            g_mask = cv2.inRange(hsv, self.g_lower_bound, self.g_upper_bound)
            g_denoise = cv2.morphologyEx(g_mask, cv2.MORPH_OPEN, self.g_kernel)

            # Mask and denoise the obstacles in the image
            o_mask = cv2.inRange(hsv, self.o_lower_bound, self.o_upper_bound)
            o_denoise = cv2.morphologyEx(o_mask, cv2.MORPH_OPEN, self.o_kernel)

            # Separate the RGB channels in the image
            blue_ch = cv_image[:, :, 0]
            green_ch = cv_image[:, :, 1]
            red_ch = cv_image[:, :, 2]

            # Overlay the masks with the corresponding channels
            g_masked = cv2.bitwise_or(blue_ch, g_denoise)
            o_masked = cv2.bitwise_or(red_ch, o_denoise)

            # Zero out the goal channels
            o_masked = cv2.bitwise_and(cv2.bitwise_not(g_denoise), o_masked)
            green_ch = cv2.bitwise_and(cv2.bitwise_not(g_denoise), green_ch)

            # Zero out the obstacle channels
            g_masked = cv2.bitwise_and(cv2.bitwise_not(o_denoise), g_masked)
            green_ch = cv2.bitwise_and(cv2.bitwise_not(o_denoise), green_ch)

            # Add an axis to concatenate onto
            g_masked = g_masked[:, :, np.newaxis]
            o_masked = o_masked[:, :, np.newaxis]
            blue_ch = blue_ch[:, :, np.newaxis]
            green_ch = green_ch[:, :, np.newaxis]
            red_ch = red_ch[:, :, np.newaxis]

            new_img = np.concatenate([red_ch, green_ch, g_masked], axis=2)

            obs = np.squeeze(new_img)
            new_img = cv2.resize(obs, TRAINING_DIMS, interpolation=cv2.INTER_AREA)

            np_img = new_img[np.newaxis, :, :, :]
            np_img = np_img / 255.0

            feed_dict = {"visual_observation_0:0": np_img, "batch_size:0": 1, "sequence_length:0": 1}

            action_probs = self.sess.run("action_probs:0", feed_dict=feed_dict)

            action = np.argmax(action_probs)
            # cv2.imwrite('/ros_node/run_imgs/pic_masked_' + str(self.count) + '.png', new_img)

            # Message to be published to /cmd_vel
            p_msg = Twist()

            # Move forward
            if (action == 0):
                p_msg.linear.x = 0.5 * SPEED

            # Move left
            if (action == 1):
                p_msg.angular.z = -1.0 * SPEED

            # Move right
            if (action == 2):
                p_msg.angular.z = 1.0 * SPEED

            # Publish the action msg to the cmd_vel topic so the robot moves
            self.pub.publish(p_msg)
            self.rate.sleep()
            self.count += 1
            self.image = None
            self.joy = None
            self.running = 0
        else:
            pass

    def main(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('action')
    model = choose_action()
    model.main()