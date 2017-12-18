from rawDataLoader import rawDataLoader
import tensorflow as tf
import cv2
import pickle
#import matplotlib.pyplot as plt
import math
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = "resnet"

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 50):
    """
    This is only a function to indicate progress and does nothing else
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '|' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def lossFunc(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, 60*80])
    depths_flat = tf.reshape(depths, [-1, 60*80])
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, 60*80])

    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat)
    d = tf.subtract(predict, target)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(sum_square_d / 60.0*80.0 - 0.5*sqare_sum_d / math.pow(60*80, 2))
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def main():

    # placeholders for data and target
    x = tf.placeholder(tf.float32, shape=[None, 240, 320, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 60, 80, 1])
    m_ = tf.placeholder(tf.float32, shape=[None, 60, 80, 1])
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


    # loss definition, MSE for raw test
    #loss = tf.losses.huber_loss(y_est, fine4, delta=0.5)
    loss = lossFunc(y, y_, m_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver = tf.train.Saver()
    trainingLoss = []
    testingLoss = []

    batch_size = 15
    epoch = 200

    print('Loading start..')
    rd = rawDataLoader()
    # loading database is required before any further operation
    # this step takes ~30 sec on my mbp
    rd.loadImageNames()
    print('Loading end.')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('training ')
        it_count = int(epoch*2200/batch_size)
        for i in range(it_count):
            [im, dp, ms] = rd.getNextBatchTraining(batch_size)
            train_step.run(feed_dict={x: im, y_: dp, m_:ms})
            printProgress(i+1, it_count)
            if i % 50 == 0:
                [imt, dpt, mst] = rd.getNextBatchTesting(10)
                tmpTL = loss.eval(feed_dict={x: imt, y_: dpt, m_:mst})
                tmpLoss = loss.eval(feed_dict={x: im, y_: dp, m_:ms})
                print('loss = ' + str(tmpTL) + ' ' + str(tmpLoss) )
            if int(i*batch_size / 2200) > int((i-1)*batch_size/2200) :
                tmpLoss = loss.eval(feed_dict={x: im, y_: dp, m_:ms})
                trainingLoss.append( tmpLoss )
                [imt, dpt, mst] = rd.getNextBatchTesting(10)
                tmpTL = loss.eval(feed_dict={x: imt, y_: dpt, m_:mst})
                testingLoss.append( tmpTL )
        saver.save(sess, 'saved_' + MODEL_NAME + '/model.ckpt')
        print('trained model saved. ')
    f = open('analyst_new/loss_' + MODEL_NAME + '.pkl', 'wb')
    pickle.dump([trainingLoss, testingLoss], f)
    #plt.plot(recLoss)
    #plt.ylabel('Loss over epoch')
    #plt.show()

if __name__ == "__main__":
    main()
