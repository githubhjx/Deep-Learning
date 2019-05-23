from skimage import io, transform
import os
import sys
import glob
import math
import numpy as np
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.setrecursionlimit(100000)

# decision  parameter
N_LABEL = 1  # Number of classes
N_BATCH = 32  # Number of data points per mini-batch

dim_intensity = 2


# decision picture parameter
w = 224
h = 224
c = 3


# path = '/home/s2/data/Pain/leave_one/tfrecord/train.tfrecords'

tr_path = '/home/s2/data/CK_new/sadness/train/'
te_path = '/home/s2/data/CK_new/sadness/test/'

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
def read_img1(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()
    imgs0 = []
    imgs1 = []
    imgs2 = []

    for idx, folder in enumerate(cate):

        L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        temp.sort()

        for b in range(0, 3):
            data = []

            data0 = []
            data1 = []
            data2 = []

            for im in range(b, L, 3):
                print('reading the images:%s' % (temp[im]))
                img = io.imread(temp[im])
                img = transform.resize(img, (w, h))
                data.append(img)

            for i in range(len(data)):
                for j in range(len(data)):
                    for k in range(len(data)):
                        if k > j > i:
                            data0.append(data[j])
                            data1.append(data[k])
                            data2.append(data[i])
            imgs0.extend(data0)
            imgs1.extend(data1)
            imgs2.extend(data2)

    return np.asarray(imgs0, np.float32), np.asarray(imgs1, np.float32), np.asarray(imgs2, np.float32)


def read_img2(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()

    Pimgs = []
    Plabels = []

    for idx, folder in enumerate(cate):

        L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        temp.sort()

        temp_label = list(range(L))  # a sequence truth
        temp_label = ((dim_intensity - 1) * np.array(temp_label) / (L - 1)).tolist()
        # temp_label = getsoft_label(temp_label)

        for im in range(0, L):
            print('reading the images:%s' % (temp[im]))
            img = io.imread(temp[im])
            img = transform.resize(img, (w, h))
            Pimgs.append(img)
            Plabels.append(temp_label[im])

    return np.asarray(Pimgs, np.float32), Plabels


trX0, trX1, trX2 = read_img1(tr_path)
teX, teY = read_img2(te_path)

trX0 = trX0.reshape(-1, 224, 224, 3)
trX1 = trX1.reshape(-1, 224, 224, 3)
trX2 = trX2.reshape(-1, 224, 224, 3)

teX = teX.reshape(-1, 224, 224, 3)

num_example = trX0.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
trX0 = trX0[arr]
trX1 = trX1[arr]
trX2 = trX2[arr]


###################################
# Input X, output Y
###################################
X0 = tf.placeholder("float", [None, 224, 224, 3], name="input_X0")
X1 = tf.placeholder("float", [None, 224, 224, 3], name="input_X1")
X2 = tf.placeholder("float", [None, 224, 224, 3], name="input_X2")

keep_prob = tf.placeholder("float", name="keep_prob")


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
        activation = tf.nn.sigmoid(activation)
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
    scope.reuse_variables()
    fc_2, kernel_2 = inference_op(X2, keep_prob)


# fc = fc_1 - fc_0

regularization_loss = tf.reduce_mean(tf.square(kernel_1))

cost = tf.reduce_mean(tf.exp(fc_0 - fc_1) - tf.exp(fc_0 - fc_2))
# cost = tf.reduce_mean(tf.exp(-fc))

train_step = tf.train.AdamOptimizer(learning_rate=1e-7).minimize(cost)
predict = fc_0


###################################################
# Train and Test
###################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for step in range(500):
    # One epoch
    costs = []
    predicts = []

    for start, end in zip(range(0, len(trX0), N_BATCH), range(N_BATCH, len(trX0) + N_BATCH, N_BATCH)):
        _, c = sess.run([train_step, cost],
                        feed_dict={X0: trX0[start:end], X1: trX1[start:end], X2: trX2[start:end], keep_prob: 1.0})
        costs.append(c)

    # Result on the test set
    results = []
    list_dat = []
    for start, end in zip(range(0, len(teX), N_BATCH), range(N_BATCH, len(teX) + N_BATCH, N_BATCH)):
        p = sess.run(predict,
                     feed_dict={X0: teX[start:end], keep_prob: 1.0})
        results.extend(p)

    test_y = []
    for i in results:
        test_y.append(i[0])
    truth_y = teY

    list_dat.append(test_y)
    list_dat.append(truth_y)
    dat = np.matrix(list_dat)
    dat = np.transpose(dat)
    pcc1 = calcPearson(test_y, truth_y)
    icc = calcICC(dat)
    mae = calMAE(test_y, truth_y)

    print('Epoch: %d, PCC: %f, ICC: %f, MAE: %f, cost: %f' % (step + 1, pcc1, icc, mae, np.mean(costs)))

saver.save(sess, model_path + 'model.ckpt')
writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
writer.close()
