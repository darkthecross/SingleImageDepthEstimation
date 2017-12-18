from rawDataLoader import rawDataLoader
import tensorflow as tf
import cv2
import pickle
#import matplotlib.pyplot as plt
import math
import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(s):
    rd = rawDataLoader()
    # loading database is required before any further operation
    # this step takes ~30 sec on my mbp
    rd.loadImageNames()
    # placeholders for data and target
    x = tf.placeholder(tf.float32, shape=[None, 240, 320, 3])
    #y_ = tf.placeholder(tf.float32, shape=[None, 60, 80, 1])
    #m_ = tf.placeholder(tf.float32, shape=[None, 60, 80, 1])
    #y_est = tf.image.resize_images(y_, (58, 78))
    #m_est = tf.image.resize_images(m_, (58, 78), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # res blocks

    upper_conv1 = tf.layers.conv2d(x, 96, 11, (4,4), activation=tf.nn.relu, padding='same')
    upper_nm1 = tf.nn.l2_normalize(upper_conv1, [1,2])
    upper_mp1 = tf.layers.max_pooling2d(upper_nm1, strides=(2,2), pool_size=(3,3), padding='same')
    upper_conv2 = tf.layers.conv2d(upper_mp1, 256, 5, activation=tf.nn.relu, padding='same')
    upper_nm2 = tf.nn.l2_normalize(upper_conv2, [1,2])
    upper_mp2 = tf.layers.max_pooling2d(upper_nm2, strides=(2,2), pool_size=(2,2), padding='same')
    upper_conv3 = tf.layers.conv2d(upper_mp2, 256, 3, activation=tf.nn.relu, padding='same')
    upper_conv4 = tf.layers.conv2d(upper_conv3, 256, 3, activation=tf.nn.relu, padding='same')
    upper_conv5 = tf.layers.conv2d(upper_conv4, 256, 3, activation=tf.nn.relu, padding='same')
    upper_conv6 = tf.layers.conv2d(upper_conv5, 128, 3, activation=tf.nn.relu, padding='same')
    total_ft1 = tf.contrib.layers.flatten(upper_conv6)
    fc1 = tf.contrib.layers.fully_connected(total_ft1, 4800)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.contrib.layers.fully_connected(h_fc1_drop, 80*60)
    res = tf.reshape(fc2, [-1, 60, 80, 1])
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('testing')
        saver.restore(sess, "saved_Alexnet/model.ckpt")
        img_path = 'result_images/Alexnet/'
        [imt, dpt, mst] = rd.getNextBatchTesting(20)
        for i in range(0,20):
            tmpRes = sess.run(res, feed_dict={x: [imt[i,:,:,:]], keep_prob: 1})
            dp1 = cv2.resize(dpt[i,:,:,:], None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)*255
            re1 = cv2.resize(tmpRes[0,:,:,:], None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)*255
            org = imt[i, :, :, :]*255
            cv2.imwrite(img_path + str(i) + '_origin_dps.jpg', dp1)
            cv2.imwrite(img_path + str(i) + '_predicted_dps.jpg', re1)
            cv2.imwrite(img_path + str(i) + '_origin_rgb.jpg', org)


if __name__ == '__main__':
    main(sys.argv)
