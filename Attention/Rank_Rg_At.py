from skimage import io, transform
import os
import sys
import glob
import math
import numpy as np
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.setrecursionlimit(100000)

# decision  parameter
N_LABEL = 1  # Number of classes
N_BATCH = 32  # Number of data points per mini-batch

# decision picture parameter
w = 224
h = 224
c = 3


path = '/home/s2/data/Pain/leave_one/tfrecord/train.tfrecords'
# path = '/home/s2/data/CK_new/tfrecord/train.tfrecords'

tr_path = '/home/s2/data/CK_new/surprise/train/'
te_path = '/home/s2/data/CK_new/surprise/test/'

log_path = '/home/s2/PycharmProjects/Lab/log/'
model_path = '/home/s2/PycharmProjects/Lab/model/'


##################################################
# Cal PCC ICC funtion
##################################################
def calcICC(dat):

    k = np.size(dat, 1)
    n = np.size(dat, 0)
    mpt = np.mean(dat, 1)
    mpr = np.mean(dat, 0)
    tm = np.mean(mpt)

    ws = sum(np.square(dat-mpt))
    WSS = np.sum(ws)

    rs = np.square(mpr - tm)
    RSS = np.sum(rs) * n

    bs = np.square(mpt - tm)
    BSS = np.sum(bs) * k

    BMS = BSS / (n - 1)
    ESS = WSS - RSS
    EMS = ESS / ((k - 1) * (n - 1))
    icc = (BMS - EMS) / (BMS + (k - 1) * EMS)

    return icc


def calcMean(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x + 0.0) / n
    y_mean = float(sum_y + 0.0) / n
    return x_mean, y_mean


def calcPearson(x, y):
    x_mean, y_mean = calcMean(x, y)
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0

    for i in range(n):
        sumTop += (x[i] - x_mean) * (y[i]-y_mean)

    for i in range(n):
        x_pow += math.pow(x[i] - x_mean, 2)

    for i in range(n):
        y_pow += math.pow(y[i]-y_mean, 2)

    sumBottom = math.sqrt(x_pow * y_pow)
    p = sumTop/sumBottom
    return p


def calMAE(x, y):
    x = np.array(x)
    y = np.array(y)
    z = abs(x-y)
    out = z/len(z)
    return np.sum(out)


##################################################
# Load data
##################################################
def read_imgs(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()

    Pimgs = []
    Plabels = []

    Nomlabels = []

    for idx, folder in enumerate(cate):

        L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*png')
        temp.sort()

        temp_label = list(range(L))  # a sequence truth
        temp_label = (1 * np.array(temp_label) / (L - 1)).tolist()

        for im in range(0, L):
            print('reading the images:%s' % (temp[im]))
            img = io.imread(temp[im])
            img = transform.resize(img, (w, h))
            Pimgs.append(img)
            Plabels.append([temp_label[im]])
            Nomlabels.append(temp_label[im])

    return np.asarray(Pimgs, np.float32), np.asarray(Plabels, np.float32), Nomlabels


teX, teY, test_label = read_imgs(te_path)

teX = teX.reshape(-1, 224, 224, 3)

###################################
# Input X, output Y
###################################
X0 = tf.placeholder("float", [None, 224, 224, 3], name="input_X0")
X1 = tf.placeholder("float", [None, 224, 224, 3], name="input_X1")
Y = tf.placeholder("float", [None, N_LABEL], name="input_Y")
keep_prob = tf.placeholder("float", name="keep_prob")

M = tf.placeholder("float", [None], name="input_m")
P = tf.placeholder("float", [None], name="input_p")
F = tf.placeholder("float", [None], name="input_f")


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='conv_w',
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(), trainable=True)
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name='conv')
        p += [kernel, biases]
        return activation


def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='f_w',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        # activation = tf.nn.relu_layer(input_op, kernel, biases, name='fc')
        activation = tf.matmul(input_op, kernel) + biases
        p += [kernel, biases]
        return activation, kernel


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


def inference_op(input_op, keep_prob):
    p = []
    # assume input_op shape is 224x224x3

    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    # conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_1, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=5, kw=5, n_out=128, dh=1, dw=1, p=p)
    # conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_1, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # # block 3 -- outputs 28x28x256
    # conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    # conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    # conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    # pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)
    #
    # # block 4 -- outputs 14x14x512
    # conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)
    #
    # # block 5 -- outputs 7x7x512
    # conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # flatten
    shp = pool2.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool2, [-1, flattened_shape], name="resh1")

    # fully connected
    fc, kernel = fc_op(resh1, name="fc6", n_out=N_LABEL, p=p)
    # fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    # fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    # fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
    #
    # fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    # softmax = tf.nn.softmax(fc8)
    # predictions = tf.argmax(softmax, 1)
    return fc, kernel


with tf.variable_scope("inference_op") as scope:
    fc_0, kernel_0 = inference_op(X0, keep_prob)
    scope.reuse_variables()
    fc_1, kernel_1 = inference_op(X1, keep_prob)


fc = fc_1 - fc_0

s_loss = tf.reduce_mean(tf.multiply(tf.square(tf.maximum((fc_1 - Y - 0.1), 0)) + tf.square(tf.maximum((Y-fc_1 - 0.1), 0)), F))

regularization_loss = tf.reduce_mean(tf.square(kernel_1))

hinge_loss = tf.reduce_mean(tf.exp(-fc))
# hinge_loss = tf.reduce_mean(tf.multiply(tf.maximum(tf.zeros([N_BATCH, N_LABEL]), 1 - fc), M)) \
#              + tf.reduce_mean(tf.maximum(0.0, fc_1-1))+tf.reduce_mean(tf.maximum(0.0, -fc_1))

cost = hinge_loss + regularization_loss

train_step = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
predict = fc_1


###################################################
# Train and Test
###################################################
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

filename_queue = tf.train.string_input_producer([path], num_epochs=10000)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                    features={
                                        'label': tf.FixedLenFeature([], tf.int64),
                                        'X0': tf.FixedLenFeature([], tf.string),
                                        'X1': tf.FixedLenFeature([], tf.string),
                                        'M': tf.FixedLenFeature([], tf.int64),
                                        'F': tf.FixedLenFeature([], tf.int64),
                                    })
img0 = tf.reshape(tf.decode_raw(features['X0'], tf.float64), [224, 224, 3])
img1 = tf.reshape(tf.decode_raw(features['X1'], tf.float64), [224, 224, 3])
label = tf.cast(features['label'], tf.int64)
margin = tf.cast(features['M'], tf.int64)
flag = tf.cast(features['F'], tf.int64)


img0_batch, img1_batch, label_batch, margin_batch, flag_batch = tf.train.shuffle_batch([img0, img1, [label], margin, flag], batch_size=32, num_threads=4, capacity=1280, min_after_dequeue=10, allow_smaller_final_batch=True)

with tf.Session() as sess:

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(10000):
        # One epoch
        costs = []
        predicts = []

        trX0, trX1, trY, Margin, Flag = sess.run([img0_batch, img1_batch, label_batch, margin_batch, flag_batch])
        # trX_s, trY_s = sess.run([img_batch_s, label_batch_s])

        _, c = sess.run([train_step, cost],
                        feed_dict={X0: trX0, X1: trX1, Y: trY, M: Margin, F: Flag, keep_prob: 1.0})
        costs.append(c)

        # Result on the test set
        results = []
        list_dat = []
        for start, end in zip(range(0, len(teX), N_BATCH), range(N_BATCH, len(teX) + N_BATCH, N_BATCH)):
            p = sess.run(predict,
                         feed_dict={X1: teX[start:end], Y: teY[start:end], keep_prob: 1.0})
            results.extend(p)

        test_y = []
        truth_y = test_label

        for k in range(len(results)):
            test_y.append(results[k][0])

        # test_y = ((np.array(test_y) - min(test_y)) / max(test_y)).tolist()

        list_dat.append(test_y)
        list_dat.append(truth_y)
        dat = np.matrix(list_dat)
        dat = np.transpose(dat)
        pcc1 = calcPearson(test_y, truth_y)
        icc = calcICC(dat)
        mae = calMAE(test_y, truth_y)

        print('Epoch: %d, PCC: %f, ICC: %f, MAE: %f, cost: %f' % (step + 1, pcc1, icc, mae, np.mean(costs)))

        # if np.mean(costs) < 1e-6:
        #     print("convergent")
        #     break

    saver.save(sess, model_path + 'model.ckpt')
    writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
    writer.close()
    coord.request_stop()
    coord.join(threads)
    sess.close()


