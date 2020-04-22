#!/usr/bin/env python

import sys
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# from tensorflow.python.framework import ops
import PIL
from tensorflow.python.platform import gfile
GRAPH_PB_PATH = 'C:/Users/drewb/PycharmProjects/SADrone/Tracker-5/Tracker-Test/raw_graph_def.pb'


MODEL = "Tracker-Test"
META_PATH = "C:/Users/drewb/PycharmProjects/SADrone/Tracker-5/" + MODEL + "/model-300000.ckpt.meta"
VARIABLE_PATH = "C:/Users/drewb/PycharmProjects/SADrone/Tracker-5/" + MODEL + "/raw_graph_def.pb"
ROBOT_NAME = "****"
RATE = 10
TRAINING_DIMS = (360, 240)
SPEED = 1



class choose_action:

    def __init__(self):

        tf.reset_default_graph()
        self.graph = tf.train.import_meta_graph(META_PATH)

        self.sess = tf.Session()
        print("load graph")
        with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')


        # self.graph.restore(self.sess, VARIABLE_PATH)


        self.image = None
        self.joy = None
        self.running = 0
        self.count = 0

        self.g_lower_bound = np.array([160, 50, 50])
        self.g_upper_bound = np.array([180, 255, 255])
        self.g_kernel = np.ones((5, 5), np.uint8)


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

    def callback(self):
        # Convert the camera image to a cv image
        pil_image = PIL.Image.open('picture.jpg').convert('RGB')
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        cv_image = open_cv_image[:, :, ::-1].copy()
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Mask the goal in the image
        g_mask = cv2.inRange(hsv, self.g_lower_bound, self.g_upper_bound)

        res = cv2.bitwise_and(cv_image, cv_image, mask=g_mask)

        obs = np.squeeze(res)
        new_img = cv2.resize(obs, TRAINING_DIMS, interpolation=cv2.INTER_AREA)

        np_img = new_img[np.newaxis, :, :, :]
        print(np_img.shape)
        np_img = np_img / 255.0

        feed_dict = {"visual_observation_0:0": np_img, "batch_size:0": 1, "sequence_length:0": 1}
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        action_probs = self.sess.run("action_probs:0", feed_dict=feed_dict)
        action = np.argmax(action_probs)
        cv2.imwrite("C:/Users/drewb/PycharmProjects/SADrone/output.png", new_img)


ca = choose_action()
ca.callback()