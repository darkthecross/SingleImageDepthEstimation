from rawDataLoader import rawDataLoader
import tensorflow as tf
import cv2
import pickle
#import matplotlib.pyplot as plt
import math
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(s):
    rd = rawDataLoader()
    # loading database is required before any further operation
    # this step takes ~30 sec on my mbp
    rd.loadImageNames()
    # placeholders for data and target
    x = tf.placeholder(tf.float32, shape=[None, 240, 320, 3])
    # res blocks

    conv0 = tf.layers.conv2d(x, 64, 7, (2,2), activation=None, padding='same')
    bn0 = tf.layers.batch_normalization(conv0)
    first_conv = tf.nn.relu(bn0)

    # res_block_1
    mp1_1 = tf.layers.max_pooling2d(first_conv, strides=(2,2), pool_size=(3,3), padding='same')
    conv1_1 = tf.layers.conv2d(mp1_1, 64, 3, activation=None, padding='same')
    bn1_1 = tf.layers.batch_normalization(conv1_1)
    relu1_1 = tf.nn.relu(bn1_1)
    conv1_2 = tf.layers.conv2d(relu1_1, 64, 3, activation=None, padding='same')
    bn1_2 = tf.layers.batch_normalization(conv1_2)
    first_conv_resize = tf.image.resize_images(first_conv, [60,80])
    add1 = tf.add(first_conv_resize, bn1_2)
    res1 = tf.nn.relu(add1)

    # res_block_2
    conv2_1 = tf.layers.conv2d(res1, 64, 3, activation=None, padding='same')
    bn2_1 = tf.layers.batch_normalization(conv2_1)
    relu2_1 = tf.nn.relu(bn2_1)
    conv2_2 = tf.layers.conv2d(relu2_1, 64, 3, activation=None, padding='same')
    bn2_2 = tf.layers.batch_normalization(conv2_2)
    add2 = tf.add(res1, bn2_2)
    res2 = tf.nn.relu(add2)

    # res_block_3
    conv3_1 = tf.layers.conv2d(res2, 128, 3, (2,2), activation=None, padding='same')
    bn3_1 = tf.layers.batch_normalization(res2)
    relu3_1 = tf.nn.relu(bn3_1)
    conv3_2 = tf.layers.conv2d(relu3_1, 128, 3, activation=None, padding='same')
    bn3_2 = tf.layers.batch_normalization(conv3_2)
    res2_resize = tf.layers.conv2d(res2, 128, 3, activation=None, padding='same')
    add3 = tf.add(res2_resize, bn3_2)
    res3 = tf.nn.relu(add3)

    # res_block_4
    conv4_1 = tf.layers.conv2d(res3, 128, 3, activation=None, padding='same')
    bn4_1 = tf.layers.batch_normalization(conv4_1)
    relu4_1 = tf.nn.relu(bn4_1)
    conv4_2 = tf.layers.conv2d(relu4_1, 128, 3, activation=None, padding='same')
    bn4_2 = tf.layers.batch_normalization(conv4_2)
    add4 = tf.add(res3, bn4_2)
    res4 = tf.nn.relu(add4)

    # res_block_5
    conv5_1 = tf.layers.conv2d(res4, 256, 3, (2,2), activation=None, padding='same')
    bn5_1 = tf.layers.batch_normalization(conv5_1)
    relu5_1 = tf.nn.relu(bn5_1)
    conv5_2 = tf.layers.conv2d(relu5_1, 256, 3, activation=None, padding='same')
    bn5_2 = tf.layers.batch_normalization(conv5_2)
    res4_resize = tf.layers.conv2d(res4, 256, 3, (2,2), activation=None, padding='same')
    add5 = tf.add(res4_resize, bn5_2)
    res5 = tf.nn.relu(add5)

    # res_block_6
    conv6_1 = tf.layers.conv2d(res5, 256, 3, activation=None, padding='same')
    bn6_1 = tf.layers.batch_normalization(conv6_1)
    relu6_1 = tf.nn.relu(bn6_1)
    conv6_2 = tf.layers.conv2d(relu6_1, 256, 3, activation=None, padding='same')
    bn6_2 = tf.layers.batch_normalization(conv6_2)
    add6 = tf.add(res5, bn6_2)
    res6 = tf.nn.relu(add6)

    # res_block_7
    conv7_1 = tf.layers.conv2d(res6, 512, 3, (2,2), activation=None, padding='same')
    bn7_1 = tf.layers.batch_normalization(conv7_1)
    relu7_1 = tf.nn.relu(bn7_1)
    conv7_2 = tf.layers.conv2d(relu7_1, 512, 3, activation=None, padding='same')
    bn7_2 = tf.layers.batch_normalization(conv7_2)
    res6_resize = tf.layers.conv2d(res6, 512, 3, (2,2), activation=None, padding='same')
    add7 = tf.add(res6_resize, bn7_2)
    res7 = tf.nn.relu(add7)

    # res_block_8
    conv8_1 = tf.layers.conv2d(res7, 512, 3, activation=None, padding='same')
    bn8_1 = tf.layers.batch_normalization(conv8_1)
    relu8_1 = tf.nn.relu(bn8_1)
    conv8_2 = tf.layers.conv2d(relu8_1, 512, 3, activation=None, padding='same')
    bn8_2 = tf.layers.batch_normalization(conv8_2)
    add8 = tf.add(res7, bn8_2)
    res8 = tf.nn.relu(add8)

    conv_elim = tf.layers.conv2d(res8, 64, 3, activation=None, padding='same')
    res_flatten = tf.contrib.layers.flatten(conv_elim)
    fc1 = tf.contrib.layers.fully_connected(res_flatten, 80*60)
    fc2 = tf.contrib.layers.fully_connected(fc1, 80*60)
    y = tf.reshape(fc2, [-1, 60, 80, 1])

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('testing')
        saver.restore(sess, "saved_resnet/model.ckpt")
        img_path = 'result_images/resnet/'
        [imt, dpt, mst] = rd.getNextBatchTesting(20)
        for i in range(0,20):
            tmpRes = sess.run(y, feed_dict={x: [imt[i,:,:,:]]})
            dp1 = cv2.resize(dpt[i,:,:,:], None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)*255
            re1 = cv2.resize(tmpRes[0,:,:,:], None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)*255
            org = imt[i, :, :, :]*255
            cv2.imwrite(img_path + str(i) + '_origin_dps.jpg', dp1)
            cv2.imwrite(img_path + str(i) + '_predicted_dps.jpg', re1)
            cv2.imwrite(img_path + str(i) + '_origin_rgb.jpg', org)

if __name__ == '__main__':
    main(sys.argv)
