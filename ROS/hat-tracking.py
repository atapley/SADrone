#!/usr/bin/env python

import roslib

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
# from sensor_msgs.msg import Joy
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile
import time

# Hyperparameters
MODEL = "Tracker-Test"
META_PATH = "./models/" + MODEL + "/model-300000.ckpt.meta"
VARIABLE_PATH = "./models/" + MODEL + "/raw_graph_def.pb"
ROBOT_NAME = "/pidrone"
RATE = 10
TRAINING_DIMS = (360, 240)
SPEED = 1


class choose_action:

    def __init__(self):
        tf.reset_default_graph()
        self.graph = tf.train.import_meta_graph(META_PATH)
        self.cv_bridge = CvBridge()
        self.sess = tf.Session()

        with gfile.FastGFile(VARIABLE_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # self.joysub = rospy.Subscriber(ROBOT_NAME + '/joy', Joy, self.joy_callback, queue_size=1)
        self.imagesub = rospy.Subscriber(ROBOT_NAME + '/picamera/image_raw', Image, self.image_callback, queue_size=1)

        self.pub = rospy.Publisher(ROBOT_NAME + '/desired/twist', Twist, queue_size=1)
        self.rate = rospy.Rate(RATE)

        self.image = None
        # self.joy = None
        self.running = 0
        self.count = 0

        self.g_lower_bound = np.array([160, 50, 50])
        self.g_upper_bound = np.array([255, 255, 255])
        self.g_kernel = np.ones((5, 5), np.uint8)

    # Set the image to the image_msg
    def image_callback(self, image_msg):
        if not self.running:
            self.image = image_msg
            self.callback()

    # # Set the joy to the joy_msg
    # def joy_callback(self, joy_msg):
    #     if not self.running:
    #         self.joy = joy_msg
    #         self.callback()

    # If button is pushed, then run through model and publish
    def callback(self):
        if (self.image is not None and not self.running):
            self.running = 1
            # Convert the camera image to a cv image
            try:
                cv_image = self.cv_bridge.imgmsg_to_cv2(self.image, "bgr8")
            except CvBridgeError as e:
                print(e)

            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Mask the goal in the image
            g_mask = cv2.inRange(hsv, self.g_lower_bound, self.g_upper_bound)
            res = cv2.bitwise_and(cv_image, cv_image, mask=g_mask)
            res[g_mask == 255] = (0, 0, 255)

            obs = np.squeeze(res)
            new_img = cv2.resize(obs, TRAINING_DIMS, interpolation=cv2.INTER_AREA)

            np_img = new_img[np.newaxis, :, :, :]
            np_img = np_img / 255.0

            feed_dict = {"visual_observation_0:0": np_img, "batch_size:0": 1, "sequence_length:0": 1}
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            action_probs = self.sess.run("action_probs:0", feed_dict=feed_dict)

            action = action_probs
            # cv2.imwrite('/ros_node/run_imgs/pic_masked_' + str(self.count) + '.png', new_img)

            p_msg = Twist()

            p_msg.linear.x = 0.5 * action[0]

            p_msg.linear.z = 0.5 * action[1]

            # Publish the action msg to the twist topic so the robot moves
            self.pub.publish(p_msg)
            self.rate.sleep()
            self.count += 1
            self.image = None
            # self.joy = None
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