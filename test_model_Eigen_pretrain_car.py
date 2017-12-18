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

    coarse1 = tf.layers.conv2d(x, 96, 11, (4,4), activation=tf.nn.relu, padding='valid')
    coarse_pool1 = tf.layers.max_pooling2d(coarse1, strides=(2,2), pool_size=(3,3), padding='valid')
    coarse2 = tf.layers.conv2d(coarse_pool1, 256, 5, activation=tf.nn.relu, padding='valid')
    coarse_pool2 = tf.layers.max_pooling2d(coarse2, strides=(2,2), pool_size=(3,3), padding='same')
    coarse3 = tf.layers.conv2d(coarse_pool2, 384, 3, activation=tf.nn.relu, padding='valid')
    coarse4 = tf.layers.conv2d(coarse3, 384, 3, activation=tf.nn.relu, padding='valid')
    #coarse_pool3 = tf.layers.max_pooling2d(coarse4, strides=(2,2), pool_size=(2,2), padding='same')
    coarse5 = tf.layers.conv2d(coarse4, 256, 3, activation=tf.nn.relu, padding='valid')
    coarse5_flatten = tf.contrib.layers.flatten(coarse5)
    coarse6 = tf.contrib.layers.fully_connected(coarse5_flatten, 4800)
    coarse7 = tf.contrib.layers.fully_connected(coarse6, 4800)
    coarse7_output = tf.reshape(coarse7, [-1, 60, 80, 1])

    keep_prob = tf.placeholder(tf.float32)

    fine1 = tf.layers.conv2d(x, 63, 9, (2,2), activation=tf.nn.relu, padding='same')
    fine_pool1 = tf.layers.max_pooling2d(fine1, strides=(2,2), pool_size=(3,3), padding='same')
    fine_pool1_dp = tf.nn.dropout(fine_pool1, keep_prob)
    fine2 = tf.concat([fine_pool1_dp, coarse7_output], 3)
    fine3 = tf.layers.conv2d(fine2, 64, 5, activation=tf.nn.relu, padding='same')
    fine3_dp = tf.nn.dropout(fine3, keep_prob)
    fine4 = tf.layers.conv2d(fine3_dp, 1, 5, activation=tf.nn.relu, padding='same')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('testing')
        saver.restore(sess, "saved_Eigen_pretrain_2_car/model.ckpt")
        img_path = 'result_images/Eigen_pretrain_car/'
        [imt, dpt, mst] = rd.getNextBatchTesting(20)
        for i in range(0,20):
            tmpRes = sess.run(fine4, feed_dict={x: [imt[i,:,:,:]], keep_prob: 1})
            dp1 = cv2.resize(dpt[i,:,:,:], None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)*255
            re1 = cv2.resize(tmpRes[0,:,:,:], None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)*255
            cr = sess.run(coarse7_output, feed_dict={x: [imt[i,:,:,:]]})
            cr1 = cv2.resize(cr[0,:,:,:], None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
            amincr = np.amin(cr1)
            amaxcr = np.amax(cr1)
            cr1 = (cr1-amincr) / (amaxcr - amincr) * 255
            org = imt[i, :, :, :]*255
            cv2.imwrite(img_path + str(i) + '_origin_dps.jpg', dp1)
            cv2.imwrite(img_path + str(i) + '_predicted_dps.jpg', re1)
            cv2.imwrite(img_path + str(i) + '_origin_rgb.jpg', org)
            cv2.imwrite(img_path + str(i) + 'coarse_output.jpg', cr1)


if __name__ == '__main__':
    main(sys.argv)
