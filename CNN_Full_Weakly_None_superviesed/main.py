
import scipy.stats.stats as stats
from scipy.special import comb
from skimage import io, transform
from scipy.io import loadmat
import glob
import os
import sys
import shutil
import math
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.setrecursionlimit(100000)

# decision  parameter
N_LABEL = 1  # Number of classes
N_BATCH = 32  # Number of data points per mini-batch

# decision picture parameter
w = 224
h = 224
c = 3


path = '.../train.tfrecords'
# path_s = '.../train.tfrecords'

tr_path = '.../train/'
te_path = '.../test/'

trlb_path = '.../train/label.txt'
telb_path = '.../test/label.txt'


log_path = '...log/'
model_path = '.../model/'


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


# calculate PCC,ICC
def calculate(raw, seq, frame, x0, x1, y, yt):
    y_truth = []
    y_predict = []

    for i in range(len(seq)):
        last_truth = []
        truth = list(range(raw[i]))  # a sequence truth

        temp = (10 * np.array(truth)/(raw[i]-1)).tolist()
        for t in range(0, len(temp), 3):
            last_truth.append(temp[t])

        predict = [0] * frame[i]  # a sequence predict
        for j in range(seq[i]):
            if y[j + sum(seq[:i])] < 0:
                predict[x0[j + sum(seq[:i])]] += 1
            else:
                predict[x1[j + sum(seq[:i])]] += 1

            if yt[j + sum(seq[:i])] < 0:
                predict[x1[j + sum(seq[:i])]] += 1
            else:
                predict[x0[j + sum(seq[:i])]] += 1

        predict = (np.array(predict) / 2).tolist()

        y_truth.extend(last_truth)
        y_predict.extend(predict)

    return y_truth, y_predict


##################################################
# Load data
##################################################
def read_label(path):
    r = open(path)
    file = r.readlines()
    records = []
    for i in range(len(file)):
        f = file[i].split()
        ft = list(map(int, f))
        records.append(ft)
    return records


def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()
    imgs0 = []
    imgs1 = []
    labels = []

    # primary_label = []

    record = read_label(trlb_path)

    for idx, folder in enumerate(cate):

        # L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        temp.sort()

        id_temp = []
        # data = []
        # label = []

        for st, ed in zip(range(0, len(record[idx]), 5), range(5, len(record[idx])+5, 5)):
            if len(record[idx][st:ed]) >= 5:
                if record[idx][st:ed] == [record[idx][st]] * 5:
                    id_temp.append(st)
                else:
                    id_temp.extend(list(range(st, ed)))
            else:
                id_temp.extend(list(range(st, len(record[idx]))))

        L = len(id_temp)

        # for im in range(0, L):
        #     print('reading the images:%s' % (temp[im]))
        #     img = io.imread(temp[im])
        #     img = transform.resize(img, (w, h))
        #     data.append(img)
        #     label.append([record[idx][im]])

        data = []
        data0_0 = []
        data1_1 = []
        label = []
        for im in id_temp:
            print('reading the images:%s' % (temp[im]))
            img = io.imread(temp[im])
            img = transform.resize(img, (w, h))
            data.append(img)
            # primary_label.append([record[idx][im]])

        # rank data
        label0 = [0]  # negative label
        label1 = [1]  # positive label
        for i in range(len(data)):
            for j in range(len(data)):
                if j > i:
                    data0_0.append(data[i])  # negative sample
                    data1_1.append(data[j])
                    # data0_0.append(data[j])
                    # data1_1.append(data[i])  # positive sample
                    label.append(label1)
                    # label.append(label0)

        # push data to img and label
        imgs0.extend(data0_0)
        imgs1.extend(data1_1)
        labels.extend(label)
        # label[idx] = 1
        # labels.append(label)

    return np.asarray(imgs0, np.float32), np.asarray(imgs1, np.float32), np.asarray(labels, np.float32)


def read_pimg(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()

    Pimgs = []
    Plabels = []

    for idx, folder in enumerate(cate):

        L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        temp.sort()
        temp_label = [[-1]] * L
        temp_label[0] = [0]
        temp_label[-1] = [1]

        for im in range(0, L):
            print('reading the images:%s' % (temp[im]))
            img = io.imread(temp[im])
            img = transform.resize(img, (w, h))
            Pimgs.append(img)
            Plabels.append(temp_label[im])

    return np.asarray(Pimgs, np.float32), np.asarray(Plabels, np.float32)


def read_imgs(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()

    imgs = []
    primary_label = []

    norm_label = []

    record = read_label(telb_path)

    for idx, folder in enumerate(cate):

        # L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        temp.sort()

        id_temp = []
        temp_labels = []

        for st, ed in zip(range(0, len(record[idx]), 5), range(5, len(record[idx]) + 5, 5)):
            if len(record[idx][st:ed]) >= 5:
                if record[idx][st:ed] == [record[idx][st]] * 5:
                    id_temp.append(st)
                else:
                    id_temp.extend(list(range(st, ed)))
            else:
                id_temp.extend(list(range(st, len(record[idx]))))

        L = len(id_temp)

        data = []

        for im in id_temp:
            print('reading the images:%s' % (temp[im]))
            img = io.imread(temp[im])
            img = transform.resize(img, (w, h))
            data.append(img)
            temp_labels.append([record[idx][im]])
            norm_label.append(record[idx][im])

        imgs.extend(data)
        primary_label.extend(temp_labels)
    return np.asarray(imgs, np.float32), np.asarray(primary_label, np.float32), norm_label


teX, teY, test_label = read_imgs(te_path)

teX = teX.reshape(-1, 224, 224, 3)

# trXs, trYs = read_pimg(tr_path)

# trX0, trX1, trY = read_img(tr_path)  # train set
# teX0_0, teX1_0, teY_0 = read_img1(te_path)   # (+)test set
# teX0_1, teX1_1, teY_1 = read_img2(te_path)   # (-)test set
# raw_seq_num, frame_num, frame, x0_idx, x1_idx = read_img3(te_path)     # cal pcc set


# trX0 = trX0.reshape(-1, 224, 224, 3)
# trX1 = trX1.reshape(-1, 224, 224, 3)

# trXs = trXs.reshape(-1, 224, 224, 3)

# teX0_0 = teX0_0.reshape(-1, 224, 224, 3)
# teX1_0 = teX1_0.reshape(-1, 224, 224, 3)
# teX0_1 = teX0_1.reshape(-1, 224, 224, 3)
# teX1_1 = teX1_1.reshape(-1, 224, 224, 3)

# teX0 = teX0.reshape(-1, 224, 224, 3)
# teX1 = teX1.reshape(-1, 224, 224, 3)

# zz = [0.1] * len(Margin)
# Threshold = np.asarray(zz, np.float32)

# num_example = trX0.shape[0]
# arr = np.arange(num_example)
# np.random.shuffle(arr)
# trX0 = trX0[arr]
# trX1 = trX1[arr]
# trY = trY[arr]

###################################
# Input X, output Y
###################################
X0 = tf.placeholder("float", [None, 224, 224, 3], name="input_X0")
X1 = tf.placeholder("float", [None, 224, 224, 3], name="input_X1")
X2 = tf.placeholder("float", [None, 224, 224, 3], name="input_X2")


Ys = tf.placeholder("float", [None, N_LABEL], name="input_Ys")
Y = tf.placeholder("float", [None, N_LABEL], name="input_Y")
M = tf.placeholder("float", [None], name="input_m")
P = tf.placeholder("float", [None], name="input_p")
F = tf.placeholder("float", [None], name="input_f")

keep_prob = tf.placeholder("float", name="keep_prob")


#######################################
# init weights function
#######################################
def extract_weights(path):
    kernels = []
    data = loadmat(path)
    layers = data['layers']
    for layer in layers[0]:
        layer_type = layer[0]['type'][0][0]
        if layer_type == 'conv':
            kernel, bias = layer[0]['weights'][0][0]
            kernels.append(kernel)
    return kernels


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_prob_weights(shape, minval=-5, maxval=5):
    return tf.Variable(tf.random_uniform(shape, minval, maxval))


global kernels
kernels = extract_weights('vgg-face.mat')


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p, wp):
    with tf.name_scope(name) as scope:
        kernel = kernels.copy()
        kernel = tf.Variable(kernel[wp], trainable=False, name="conv_kernel")
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=False, name="conv_b")
        # biases =bias[wp]
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def fc_op6(input_op, name, n_out, p, wp):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel_1 = kernels.copy()
        kernel = tf.Variable(kernel_1[wp], trainable=False)
        kernel = tf.reshape(kernel, shape=[n_in, n_out], name="fc_kernel")
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), trainable=False, name="fc_b")
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


def fc_op7(input_op, name, n_out, p, wp):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel_1 = kernels.copy()
        kernel = tf.Variable(kernel_1[wp], trainable=True)
        kernel = tf.reshape(kernel, shape=[n_in, n_out], name="fc_kernel")
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), trainable=True, name="fc_b")
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        # activation = tf.matmul(input_op, kernel) + biases
        p += [kernel, biases]
        return activation


def fc_op8(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='fc8_w',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 trainable=True)
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), trainable=True, name="b")
        # activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
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
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p, wp=0)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p, wp=1)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p, wp=2)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p, wp=3)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p, wp=4)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p, wp=5)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p, wp=6)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=7)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=8)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=9)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=10)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=11)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=12)
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # flatten
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op6(resh1, name="fc6", n_out=4096, p=p, wp=13)
    # fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op7(fc6, name="fc7", n_out=4096, p=p, wp=14)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    fc8, kernel = fc_op8(fc7_drop, name="fc8", n_out=N_LABEL, p=p)

    softmax = tf.nn.softmax(fc8, name="softmax")

    # predictions = tf.argmax(softmax, 1, name="predictions")

    predictions = tf.sign(fc8, name="predictions")
    # fc8 = tf.maximum(fc8, 0)

    return predictions, softmax, fc8, fc7, p, kernel


with tf.variable_scope("inference_op") as scope:
    predictions, softmax, fc8, fc7, p, kernel = inference_op(X0, keep_prob)
    scope.reuse_variables()
    predictions_s, softmax_s, fc8_s, fc7_s, p_s, kernel_s = inference_op(X1, keep_prob)

fc8_l = fc8_s - fc8
# output_s = tf.to_float(tf.greater_equal(Ys, 0)) * fc8_s
s_loss = tf.reduce_mean(tf.multiply(tf.maximum((fc8_s-Y - 0.1), 0) + tf.maximum((Y-fc8_s - 0.1), 0), F))

regularization_loss = tf.reduce_mean(tf.square(kernel))

## M is a weakly_superviesed flag
hinge_loss = tf.reduce_mean(tf.square(tf.losses.hinge_loss(Y, fc8_l))*M)  

# hinge_loss = tf.reduce_mean(tf.square(tf.maximum(tf.zeros([N_BATCH, N_LABEL]), 1 - Y * fc8)))

cost = hinge_loss + regularization_loss + s_loss*100
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=fc8))
# cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(Y, fc8, M))

# acc = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
# acc = tf.equal(tf.greater(predictions, 0), tf.greater_equal(Y, 1))

train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
predict = fc8_s

tf.add_to_collection("fc7", fc7)
tf.add_to_collection("fc8", fc8)
tf.add_to_collection("train_step", train_step)
tf.add_to_collection("predict", predict)
tf.add_to_collection("X0", X0)
tf.add_to_collection("X1", X1)
tf.add_to_collection("X2", X2)
tf.add_to_collection("Y", Y)
tf.add_to_collection("keep_prob", keep_prob)

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
# img = tf.reshape(img, [224, 224, 3])
# img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
label = tf.cast(features['label'], tf.int64)
margin = tf.cast(features['M'], tf.int64)
flag = tf.cast(features['F'], tf.int64)


# filename_queue_s = tf.train.string_input_producer([path_s], num_epochs=10000)
# reader_s = tf.TFRecordReader()
# _, serialized_example_s = reader_s.read(filename_queue_s)
# feature_s = tf.parse_single_example(serialized_example_s,
#                                     features={
#                                         'label': tf.FixedLenFeature([], tf.int64),
#                                         'X': tf.FixedLenFeature([], tf.string),
#                                     })
# img_s = tf.reshape(tf.decode_raw(feature_s['X'], tf.float64), [224, 224, 3])
#
# label_s = tf.cast(features['label'], tf.int64)


# sess = tf.Session()
# sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
# # 启动多线程处理输入数据。
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess,coord=coord)

# for i in range(1):
#     image0, image1, label = sess.run([img0, img1, label])
#     print(image0.shape)
#     print(label.shape)
#     # print(pixel)

img0_batch, img1_batch, label_batch, margin_batch, flag_batch = tf.train.shuffle_batch([img0, img1, [label], margin, flag], batch_size=32, num_threads=4, capacity=1280, min_after_dequeue=10, allow_smaller_final_batch=True)
# img_batch_s, label_batch_s = tf.train.shuffle_batch([img_s, [label_s]], batch_size=32, num_threads=4, capacity=1280, min_after_dequeue=10, allow_smaller_final_batch=True)

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

        # for start, end in zip(range(0, len(trX0), N_BATCH), range(N_BATCH, len(trX0) + N_BATCH, N_BATCH)):
            # _, c = sess.run([train_step, cost], feed_dict={X0: trX0[start:end], X1: trX1[start:end], Y: trY[start:end], M: Margin[start:end], P: Threshold[start:end], keep_prob: 1.0})
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

        list_dat.append(test_y)
        list_dat.append(truth_y)
        dat = np.matrix(list_dat)
        dat = np.transpose(dat)
        pcc1 = calcPearson(test_y, truth_y)
        # pcc2 = stats.pearsonr(test_y, truth_y)[0]
        icc = calcICC(dat)
        mae = calMAE(test_y, truth_y)

        print('Epoch: %d, PCC: %f, ICC: %f, MAE: %f, cost: %f' % (step + 1, pcc1, icc, mae, np.mean(costs)))

        # if step%200 == 0:
        #     print(test_y)
        #     print(truth_y)

        # print('Epoch: %d, (+)&(-)Test Accuracy: %f, cost: %f' % (step + 1, np.mean(result), np.mean(costs)))

        # print('Epoch: %d, (+)&(-)Test Accuracy: %f, cost: %f' % (step + 1, np.mean(result), np.mean(costs)))

        if np.mean(costs) < 1e-6:
            print("convergent")
            break

    saver.save(sess, model_path + 'model.ckpt')
    writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
    writer.close()
    coord.request_stop()
    coord.join(threads)
    sess.close()
    #     results0 = []
    #     results1 = []
    #     Y_t = []
    #     Y_t_f = []
    #     list_dat = []
    #     for start, end in zip(range(0, len(teX0_0), N_BATCH), range(N_BATCH, len(teX0_0) + N_BATCH, N_BATCH)):
    #         p = sess.run(predict,
    #                      feed_dict={X0: teX0_0[start:end], X1: teX1_0[start:end], Y: teY_0[start:end], keep_prob: 1.0})
    #         results0.extend(p)
    #
    #         p_ = sess.run(predict,
    #                       feed_dict={X0: teX0_1[start:end], X1: teX1_1[start:end], Y: teY_1[start:end], keep_prob: 1.0})
    #         results1.extend(p_)
    #
    #         Y_ = sess.run(predict,
    #                       feed_dict={X0: teX0_0[start:end], X1: teX1_0[start:end], Y: teY_0[start:end], keep_prob: 1.0})
    #         Y_t.extend(Y_)
    #
    #         Y_f = sess.run(predict, feed_dict={X0: teX0_1[start:end], X1: teX1_1[start:end], Y: teY_1[start:end],
    #                                            keep_prob: 1.0})
    #         Y_t_f.extend(Y_f)
    #
    #     truth_y, test_y = calculate(raw_seq_num, frame_num, frame, x0_idx, x1_idx, Y_t, Y_t_f)
    #
    #     list_dat.append(test_y)
    #     list_dat.append(truth_y)
    #     dat = np.matrix(list_dat)
    #     dat = np.transpose(dat)
    #     pcc1 = calcPearson(test_y, truth_y)
    #     pcc2 = stats.pearsonr(test_y, truth_y)[0]
    #     icc = calcICC(dat)
    #
    #     result = results0.copy()
    #     for i in range(len(results0)):
    #         if results0[i] == 1 and results1[i] == -1:
    #             result[i] = True
    #         else:
    #             result[i] = False
    #
    #     print('Epoch: %d, (+)&(-)Test Accuracy: %f, PCC: %f, ICC: %f, cost: %f' % (
    #     step + 1, np.mean(result), pcc1, icc, np.mean(costs)))
    #
    #     # print('Epoch: %d, (+)&(-)Test Accuracy: %f, cost: %f' % (step + 1, np.mean(result), np.mean(costs)))
    #
    #     # print('Epoch: %d, (+)&(-)Test Accuracy: %f, cost: %f' % (step + 1, np.mean(result), np.mean(costs)))
    #
    #     if np.mean(costs) < 1e-6:
    #         print("convergent")
    #         break
    #
    # saver.save(sess, model_path + 'model.ckpt')
    # writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
    # writer.close()
    # coord.request_stop()
    # coord.join(threads)
    # sess.close()
