from rawDataLoader import rawDataLoader
import tensorflow as tf
import cv2
import pickle
#import matplotlib.pyplot as plt
import math
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = "Eigen_modified_2"

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
    #y_est = tf.image.resize_images(y_, (58, 78))
    #m_est = tf.image.resize_images(m_, (58, 78), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # res blocks

    coarse1 = tf.layers.conv2d(x, 96, 11, (4,4), activation=tf.nn.relu, padding='valid', trainable=False)
    coarse_pool1 = tf.layers.max_pooling2d(coarse1, strides=(2,2), pool_size=(3,3), padding='valid')
    coarse2 = tf.layers.conv2d(coarse_pool1, 256, 5, activation=tf.nn.relu, padding='valid', trainable=False)
    coarse_pool2 = tf.layers.max_pooling2d(coarse2, strides=(2,2), pool_size=(3,3), padding='same')
    coarse3 = tf.layers.conv2d(coarse_pool2, 384, 3, activation=tf.nn.relu, padding='valid', trainable=False)
    #coarse4 = tf.layers.conv2d(coarse3, 384, 3, activation=tf.nn.relu, padding='valid', trainable=False)
    #coarse_pool3 = tf.layers.max_pooling2d(coarse4, strides=(2,2), pool_size=(2,2), padding='same')
    coarse5 = tf.layers.conv2d(coarse3, 256, 3, activation=tf.nn.relu, padding='valid', trainable=False)
    coarse5_flatten = tf.contrib.layers.flatten(coarse5)
    coarse6 = tf.contrib.layers.fully_connected(coarse5_flatten, 4800, trainable=False)
    coarse7 = tf.contrib.layers.fully_connected(coarse6, 4800, trainable=False)
    coarse7_output = tf.reshape(coarse7, [-1, 60, 80, 1])

    keep_prob = tf.placeholder(tf.float32)

    fine1 = tf.layers.conv2d(x, 1, 9, (2,2), activation=tf.nn.relu, padding='same')
    fine_pool1 = tf.layers.max_pooling2d(fine1, strides=(2,2), pool_size=(3,3), padding='same')
    fine_pool1_dp = tf.nn.dropout(fine_pool1, keep_prob)
    fine2 = tf.concat([fine_pool1_dp, coarse7_output], 3)
    fine3 = tf.layers.conv2d(fine2, 64, 5, activation=tf.nn.relu, padding='same')
    fine3_dp = tf.nn.dropout(fine3, keep_prob)
    fine5 = tf.layers.conv2d(fine3_dp, 1, 5, activation=tf.nn.relu, padding='same')
    # loss definition, MSE for raw test
    #loss = tf.losses.huber_loss(y_est, fine4, delta=0.5)

    loss2 = lossFunc(fine5, y_, m_)
    train_step2 = tf.train.AdamOptimizer(1e-4).minimize(loss2)

    saver = tf.train.Saver()
    trainingLoss = []
    testingLoss = []

    batch_size = 20
    epoch = 100

    print('Loading start..')
    rd = rawDataLoader()
    # loading database is required before any further operation
    # this step takes ~30 sec on my mbp
    rd.loadImageNames()
    print('Loading end.')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "saved_Eigen_modified_1/model.ckpt")
        print('training ')
        it_count = int(epoch*2200/batch_size)
        for i in range(it_count):
            [im, dp, ms] = rd.getNextBatchTraining(batch_size)
            train_step2.run(feed_dict={x: im, y_: dp, m_:ms, keep_prob: 0.5})
            printProgress(i+1, it_count)
            if i % 50 == 0:
                [imt, dpt, mst] = rd.getNextBatchTesting(10)
                tmpTL = loss2.eval(feed_dict={x: imt, y_: dpt, m_:mst, keep_prob: 1})
                tmpLoss = loss2.eval(feed_dict={x: im, y_: dp, m_:ms, keep_prob: 1})
                print('loss = ' + str(tmpTL) + ' ' + str(tmpLoss) )
            if int(i*batch_size / 2200) > int((i-1)*batch_size/2200) :
                tmpLoss = loss2.eval(feed_dict={x: im, y_: dp, m_:ms, keep_prob: 1})
                trainingLoss.append( tmpLoss )
                [imt, dpt, mst] = rd.getNextBatchTesting(10)
                tmpTL = loss2.eval(feed_dict={x: imt, y_: dpt, m_:mst, keep_prob: 1})
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
