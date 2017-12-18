from dbReader_simplified import dbReader
import tensorflow as tf
import cv2
import numpy as np

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(s):
    if len(s) == 0:
        print('Error: please specify model.')
    else:
        rd = dbReader()
        # loading database is required before any further operation
        # this step takes ~30 sec on my mbp
        rd.loadData('rgb', 'origin')
        # placeholders for data and target
        x = tf.placeholder(tf.float32, shape=[None, 320, 240, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 80, 60, 1])
        if s[1] == 'Alex_new':
            upper_conv1 = tf.layers.conv2d(x, 96, 11, (4,4), activation=tf.nn.relu, padding='same')
            upper_nm1 = tf.nn.l2_normalize(upper_conv1, [1,2])
            upper_mp1 = tf.layers.max_pooling2d(upper_nm1, strides=(2,2), pool_size=(3,3), padding='same')
            upper_conv2 = tf.layers.conv2d(upper_mp1, 256, 5, activation=tf.nn.relu, padding='same')
            upper_nm2 = tf.nn.l2_normalize(upper_conv2, [1,2])
            upper_mp2 = tf.layers.max_pooling2d(upper_nm2, strides=(2,2), pool_size=(2,2), padding='same')
            upper_conv3 = tf.layers.conv2d(upper_mp2, 384, 3, activation=tf.nn.relu, padding='same')
            upper_conv4 = tf.layers.conv2d(upper_conv3, 384, 3, activation=tf.nn.relu, padding='same')
            upper_conv5 = tf.layers.conv2d(upper_conv4, 256, 3, activation=tf.nn.relu, padding='same')
            upper_conv6 = tf.layers.conv2d(upper_conv5, 32, 3, activation=tf.nn.relu, padding='same')
            total_ft1 = tf.contrib.layers.flatten(upper_conv6)
            fc1 = tf.contrib.layers.fully_connected(total_ft1, 4800)
            res = tf.reshape(fc1, [-1, 80, 60, 1])
            # loss definition, MSE for raw test
            loss = tf.losses.huber_loss(y_, res, delta=0.5)
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print('testing')
                saver.restore(sess, "saved_Alex_new/model.ckpt")
                [im, dp] = rd.getTest(4)
                for i in range(0,50):
                    tmpRes = sess.run(res, feed_dict={x: [im[i,:,:,:]]})
                    dp1 = cv2.resize(dp[i,:,:,:], None,fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
                    mdp1 = np.amax(dp1)
                    dp1 = dp1/mdp1
                    re1 = cv2.resize(tmpRes[0,:,:,:], None,fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
                    mre1 = np.amax(re1)
                    re1 = re1/mre1
                    cv2.imshow('origin', dp1)
                    cv2.imshow('predicted', re1)
                    cv2.waitKey(0)
        elif s[1] == 'GLN_dp':
                inception1_conv1 = tf.layers.conv2d(x, 64, 9, (4,4), activation=tf.nn.relu, padding='same')
                inception1_mp1 = tf.layers.max_pooling2d(x, strides=(4,4), pool_size=(4,4), padding='same')
                inception1_conv2 = tf.layers.conv2d(inception1_mp1, 128, 5, activation=tf.nn.relu, padding='same')
                inception1_mp2 = tf.layers.max_pooling2d(inception1_conv2, strides=(2,2), pool_size=(2,2), padding='same')
                inception1_conv3 = tf.layers.conv2d(inception1_mp2, 128, 5, activation=tf.nn.relu, padding='same')
                inception1_mp3 = tf.layers.max_pooling2d(inception1_conv3, strides=(2,2), pool_size=(2,2), padding='same')
                inception1_flatten = tf.contrib.layers.flatten(inception1_mp3)
                fc1 = tf.contrib.layers.fully_connected(inception1_flatten, 1024)
                fc2 = tf.contrib.layers.fully_connected(fc1, 80*60)
                conc = tf.reshape(fc2, [-1, 80, 60, 1])
                inception2_conv1 = tf.layers.conv2d(x, 16, 1, activation=tf.nn.relu, padding='same')
                inception2_conv2 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu, padding='same')
                inception2_conv3 = tf.layers.conv2d(x, 16, 5, activation=tf.nn.relu, padding='same')
                inception2_mp1 = tf.layers.max_pooling2d(x, strides=(1,1), pool_size=(3,3), padding='same')
                inception2 = tf.concat( [inception2_conv1, inception2_conv2, inception2_conv3, inception2_mp1], 3 ) # 195
                #conv3 = tf.layers.conv2d(inception2, 128, 3, activation=tf.nn.relu, padding='same')
                inception_pool2 = tf.layers.max_pooling2d(inception2, strides=(2,2), pool_size=(2,2), padding='same')
                conv4 = tf.layers.conv2d(inception_pool2, 64, 3, activation=tf.nn.relu, padding='same')
                inception_pool3 = tf.layers.max_pooling2d(conv4, strides=(2,2), pool_size=(2,2), padding='same')
                conv5 = tf.layers.conv2d(inception_pool3, 63, 3, activation=tf.nn.relu, padding='same')
                join = tf.concat( [conc, conv5], 3 ) # 16
                conv6 = tf.layers.conv2d(join, 128, 3, activation=tf.nn.relu, padding='same')
                conv7 = tf.layers.conv2d(conv6, 64, 3, activation=tf.nn.relu, padding='same')
                keep_prob = tf.placeholder(tf.float32)
                conv7_dp = tf.nn.dropout(conv7, keep_prob)
                conv8 = tf.layers.conv2d(conv7_dp, 1, 3, activation=tf.nn.relu, padding='same')
                # loss definition, MSE for raw test
                saver = tf.train.Saver()
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    print('testing')
                    saver.restore(sess, "saved_GLN_dp/model.ckpt")
                    [im, dp] = rd.getTest(4)
                    for i in range(0,50):
                        tmpRes = sess.run(conv8, feed_dict={x: [im[i,:,:,:]], keep_prob: 1})
                        dp1 = cv2.resize(dp[i,:,:,:], None,fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
                        mdp1 = np.amax(dp1)
                        dp1 = dp1/mdp1
                        re1 = cv2.resize(tmpRes[0,:,:,:], None,fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
                        mre1 = np.amax(re1)
                        re1 = re1/mre1
                        cv2.imshow('origin', dp1)
                        cv2.imshow('predicted', re1)
                        cv2.waitKey(0)
        elif s[1] == 'Alex_new_dp':
            upper_conv1 = tf.layers.conv2d(x, 96, 11, (4,4), activation=tf.nn.relu, padding='same')
            upper_nm1 = tf.nn.l2_normalize(upper_conv1, [1,2])
            upper_mp1 = tf.layers.max_pooling2d(upper_nm1, strides=(2,2), pool_size=(3,3), padding='same')
            upper_conv2 = tf.layers.conv2d(upper_mp1, 256, 5, activation=tf.nn.relu, padding='same')
            upper_nm2 = tf.nn.l2_normalize(upper_conv2, [1,2])
            upper_mp2 = tf.layers.max_pooling2d(upper_nm2, strides=(2,2), pool_size=(2,2), padding='same')
            upper_conv3 = tf.layers.conv2d(upper_mp2, 384, 3, activation=tf.nn.relu, padding='same')
            upper_conv4 = tf.layers.conv2d(upper_conv3, 384, 3, activation=tf.nn.relu, padding='same')
            upper_conv5 = tf.layers.conv2d(upper_conv4, 256, 3, activation=tf.nn.relu, padding='same')
            upper_conv6 = tf.layers.conv2d(upper_conv5, 128, 3, activation=tf.nn.relu, padding='same')
            total_ft1 = tf.contrib.layers.flatten(upper_conv6)
            #fc1 = tf.contrib.layers.fully_connected(total_ft1, 4800)
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(total_ft1, keep_prob)
            fc2 = tf.contrib.layers.fully_connected(h_fc1_drop, 80*60)
            res = tf.reshape(fc2, [-1, 80, 60, 1])
            # loss definition, MSE for raw test
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print('testing')
                saver.restore(sess, "saved_GLN_dp/model.ckpt")
                [im, dp] = rd.getTest(4)
                for i in range(0,50):
                    tmpRes = sess.run(res, feed_dict={x: [im[i,:,:,:]], keep_prob: 1})
                    dp1 = cv2.resize(dp[i,:,:,:], None,fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
                    mdp1 = np.amax(dp1)
                    dp1 = dp1/mdp1
                    re1 = cv2.resize(tmpRes[0,:,:,:], None,fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
                    mre1 = np.amax(re1)
                    re1 = re1/mre1
                    cv2.imshow('origin', dp1)
                    cv2.imshow('predicted', re1)
                    cv2.waitKey(0)
        elif s[1] == 'Alex':
            rd.loadData('rgb', 'normalized')
            upper_conv1 = tf.layers.conv2d(x, 48, 11, (4,4), activation=tf.nn.relu, padding='same')
            lower_conv1 = tf.layers.conv2d(x, 48, 11, (4,4), activation=tf.nn.relu, padding='same')
            upper_nm1 = tf.nn.l2_normalize(upper_conv1, [1,2])
            upper_mp1 = tf.layers.max_pooling2d(upper_nm1, strides=(2,2), pool_size=(2,2), padding='same')
            lower_nm1 = tf.nn.l2_normalize(lower_conv1, [1,2])
            lower_mp1 = tf.layers.max_pooling2d(lower_nm1, strides=(2,2), pool_size=(2,2), padding='same')
            upper_conv2 = tf.layers.conv2d(upper_mp1, 128, 5, activation=tf.nn.relu, padding='same')
            lower_conv2 = tf.layers.conv2d(lower_mp1, 128, 5, activation=tf.nn.relu, padding='same')
            total_conc1 = tf.concat([upper_conv2, lower_conv2], 3)
            total_nm1 = tf.nn.l2_normalize(total_conc1, [1,2])
            total_mp1 = lower_mp1 = tf.layers.max_pooling2d(total_nm1, strides=(2,2), pool_size=(2,2), padding='same')
            upper_conv3 = tf.layers.conv2d(total_mp1, 192, 3, activation=tf.nn.relu, padding='same')
            lower_conv3 = tf.layers.conv2d(total_mp1, 192, 3, activation=tf.nn.relu, padding='same')
            upper_conv4 = tf.layers.conv2d(upper_conv3, 192, 3, activation=tf.nn.relu, padding='same')
            lower_conv4 = tf.layers.conv2d(lower_conv3, 192, 3, activation=tf.nn.relu, padding='same')
            upper_conv5 = tf.layers.conv2d(upper_conv4, 128, 3, activation=tf.nn.relu, padding='same')
            lower_conv5 = tf.layers.conv2d(lower_conv4, 128, 3, activation=tf.nn.relu, padding='same')
            total_conc2 = tf.concat([upper_conv5, lower_conv5], 3)
            total_conv1 = tf.layers.conv2d(total_conc2, 32, 3, activation=tf.nn.relu, padding='same')
            total_ft1 = tf.contrib.layers.flatten(total_conv1)
            fc1 = tf.contrib.layers.fully_connected(total_ft1, 4800)
            #fc2 = tf.contrib.layers.fully_connected(fc1, 80*60)
            res = tf.reshape(fc1, [-1, 80, 60, 1])
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print('testing')
                saver.restore(sess, "saved_Alex/model.ckpt")
                [im, dp] = rd.getTest(4)
                for i in range(0,50):
                    tmpRes = sess.run(res, feed_dict={x: [im[i,:,:,:]]})
                    dp1 = cv2.resize(dp[i,:,:,:], None,fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
                    mdp1 = np.amax(dp1)
                    dp1 = dp1/mdp1
                    re1 = cv2.resize(tmpRes[0,:,:,:], None,fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
                    mre1 = np.amax(re1)
                    re1 = re1/mre1
                    cv2.imshow('origin', dp1)
                    cv2.imshow('predicted', re1)
                    cv2.waitKey(0)
        else:
            print('Invalid model.')

if __name__ == '__main__':
    main(sys.argv)
