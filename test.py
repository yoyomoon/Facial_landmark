# -*- coding: utf-8 -*-

# from PrepareData import get_next_batch
import tensorflow as tf
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')

def loadtxt(train_path):
    f = open(train_path, 'r')

    img_names = []
    face_loc = []
    face_landmark = []
    lines = f.readlines()

    for i in lines:
        annotation = i.strip().split(' ')
        a = annotation
        img_names.append((a[0]))
        face_loc.append(a[1:5])
        face_landmark.append(a[5:])

    return img_names, face_loc, face_landmark


def select_ran_data(howlong):
    data_comb = range(howlong)
    random.shuffle(data_comb)
    train_quantity = int(0.9*howlong)

    train_select = data_comb[:train_quantity]
    test_select = data_comb[train_quantity:]
    return train_select, test_select


def gen_img(img_names):
    cnt = 0

    for names in img_names:
        if cnt == 0:
            I = cv2.imread(names)
            I = I[:, :, 0]
            I = cv2.resize(I, (40, 40))
            img = I
            cnt += 1
        else:
            I = cv2.imread(names)
            I = I[:, :, 0]
            I = cv2.resize(I, (40, 40))

            img = np.dstack((img, I))
    return img


def gen_landmark_size(landmark_loc):
    heights = 480.0
    widths = 640.0
    resized_heght = 400.0
    resized_width = 400.0

    after_process = []
    for i in landmark_loc:
        i = list(int(j) for j in i)
        after_process.append([round((resized_width / widths) * i[0], 2), round((resized_heght / heights) * i[1], 2),
                              round((resized_width / widths) * i[2], 2), round((resized_heght / heights) * i[3], 2),
                              round((resized_width / widths) * i[4], 2), round((resized_heght / heights) * i[5], 2)])
    return after_process

def get_next_batch(imgs, landmarks, batch_size):
    howlong = range(np.shape(imgs)[0])
    shffle_list = random.sample(howlong, batch_size)
    training_imgs = []
    training_landmarks = []
    for i in shffle_list:
        training_imgs.append(imgs[i, :])
        training_landmarks.append(landmarks[i,:])
    training_imgs = np.asarray(training_imgs)
    training_landmarks = np.asarray(training_landmarks)
    return training_imgs, training_landmarks


# def conv_net(image, weights, biases, dropout):
#     # Reshape input picture
#     x_image = tf.reshape(image, [-1, 40, 40, 1])
#     # Layer1
#     h_conv1 = tf.abs(tf.nn.tanh(conv2d(x_image, weights['wc1']) + biases['bc1']))
#     h_pool1 = max_pool_2x2(h_conv1)
#
#     # Layer2
#     h_conv2 = tf.abs(tf.nn.tanh(conv2d(h_pool1, weights['wc2']) + biases['bc2']))
#     h_pool2 = max_pool_2x2(h_conv2)
#
#     # Layer3
#     h_conv3 = tf.abs(tf.nn.tanh(conv2d(h_pool2, weights['wc3']) + biases['bc3']))
#     h_pool3 = max_pool_2x2(h_conv3)
#
#     # Layer4
#     h_conv4 = tf.abs(tf.nn.tanh(conv2d(h_pool3, weights['wc4']) + biases['bc4']))
#     h_pool4 = h_conv4
#
#     # Fully connected layer
#     # Reshape conv4 output to fit fully connected layer input
#     h_pool4_flat = tf.reshape(h_pool4, [-1, 2 * 2 * 64])
#     h_fc1 = tf.abs(tf.nn.tanh(tf.matmul(h_pool4_flat, weights['wd1']) + biases['bd1']))
#     h_fc1_drop = tf.nn.dropout(h_fc1, dropout)
#
#     # Output, landmark regression
#     y_landmark = tf.matmul(h_fc1_drop, weights['out']) + biases['out']
#     W_fc_landmark = weights['out']
#     return y_landmark, W_fc_landmark


# Store layers weight & bias

weights = {
    # 5x5 conv, 1 input, 16 outputs
    'wc1': weight_variable([5, 5, 1, 16]),
    # 3x3 conv, 16 inputs, 48 outputs
    'wc2': weight_variable([3, 3, 16, 48]),
	# 3x3 conv, 48 inputs, 64 outputs
    'wc3': weight_variable([3, 3, 48, 64]),
    # 2x2 conv, 64 inputs, 64 outputs
    'wc4': weight_variable([2, 2, 64, 64]),

    # fully connected, 2*2*64 inputs, 100 outputs
    'wd1': weight_variable([2 * 2 * 64, 100]),
    # 100 inputs, 68*2 outputs (class prediction)
    'out': weight_variable([100, 6])
}

biases = {
    'bc1': bias_variable([16]),
    'bc2': bias_variable([48]),
    'bc3': bias_variable([64]),
    'bc4': bias_variable([64]),

    'bd1': bias_variable([100]),
    'out': bias_variable([6])
}

batch_size = 1

img_names, face_loc, landmarks = loadtxt('train_landmark_list.txt')
train_selects, test_selects = select_ran_data(len(img_names))

train_img_name = list(img_names[i] for i in train_selects)
test_img_name = list(img_names[i] for i in test_selects)
train_landmarks = list(landmarks[i] for i in train_selects)
test_landmarks = list(landmarks[i] for i in test_selects)

train_imgs = gen_img(train_img_name)
test_imgs = gen_img(test_img_name)
train_landmarks = gen_landmark_size(train_landmarks)
test_landmarks = gen_landmark_size(test_landmarks)

train_imgs = np.transpose(train_imgs, (2, 0, 1))
test_imgs = np.transpose(test_imgs, (2, 0, 1))
train_landmarks = np.asarray(train_landmarks)
test_landmarks = np.asarray(test_landmarks)

image = tf.placeholder(tf.float32, shape=[None, 40, 40])
landmark = tf.placeholder(tf.float32, shape=[None, 6])
keep_prob = tf.placeholder(tf.float32)



# Reshape input picture
x_image = tf.reshape(image, [-1, 40, 40, 1])
# Layer1
h_conv1 = tf.abs(tf.nn.tanh(conv2d(x_image, weights['wc1']) + biases['bc1']))
h_pool1 = max_pool_2x2(h_conv1)

# Layer2
h_conv2 = tf.abs(tf.nn.tanh(conv2d(h_pool1, weights['wc2']) + biases['bc2']))
h_pool2 = max_pool_2x2(h_conv2)

# Layer3
h_conv3 = tf.abs(tf.nn.tanh(conv2d(h_pool2, weights['wc3']) + biases['bc3']))
h_pool3 = max_pool_2x2(h_conv3)

# Layer4
h_conv4 = tf.abs(tf.nn.tanh(conv2d(h_pool3, weights['wc4']) + biases['bc4']))
h_pool4 = h_conv4

# Fully connected layer
# Reshape conv4 output to fit fully connected layer input
h_pool4_flat = tf.reshape(h_pool4, [-1, 2 * 2 * 64])
h_fc1 = tf.abs(tf.nn.tanh(tf.matmul(h_pool4_flat, weights['wd1']) + biases['bd1']))
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output, landmark regression
y_landmark = tf.matmul(h_fc1_drop, weights['out']) + biases['out']
W_fc_landmark = weights['out']

# loss = tf.reduce_mean(tf.reduce_sum(tf.square(landmark - y_landmark), 1))

# y_landmark, W_fc_landmark = conv_net(image, weights, biases, keep_prob)
cost = tf.reduce_sum(tf.square(landmark - y_landmark)) / 2 + 2*tf.nn.l2_loss(W_fc_landmark)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.001, global_step, 800, 0.9, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

landmark_error = tf.nn.l2_loss(landmark - y_landmark) / 2
fc_landmark_error = 2*tf.nn.l2_loss(W_fc_landmark)

add_global = global_step.assign_add(1)

train_Flag = True

my_totoal_loss = []
plt.ion()
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if train_Flag:

        for x in range(1000000):
                training_imgs, training_landmarks = get_next_batch(train_imgs, train_landmarks, batch_size)
                # training_imgs = cv2.imread('/home/yoyomoon/Data/T_train/face12.png')
                # I = training_imgs
                # training_imgs = cv2.resize(training_imgs, (40,40))
                # training_imgs = training_imgs[:, :, 0]
                # training_imgs = training_imgs[np.newaxis, :, :]
                # plt.imshow(training_imgs)
                _, c1, c2, c3 = sess.run([optimizer, h_conv1, h_conv2, h_conv3], feed_dict={image: training_imgs, landmark: training_landmarks, keep_prob: 1})
                # test_er, _, test_mse, test_regular, test_loss_func = sess.run([error, optimizer, rmse, regular, loss_func],
                #                                 feed_dict={image: test_imgs, landmark: test_landmarks, keep_prob: 1})
                # plt.figure(0)
                # plt.imshow(training_imgs)
                # plt.figure(1)
                # for j in range(48):
                #     plt.subplot(6, 8, j+1)
                #     plt.imshow(c2[0, :, :, j])
                #
                # plt.show()
                if x % 5 == 0:
                    loss, landerror, fc_err, lr = sess.run([cost, landmark_error, fc_landmark_error, learning_rate],\
                                             feed_dict={image: training_imgs, landmark: training_landmarks, keep_prob: 1})

                    print('Num of step : ' + str(x) + '/1000000')
                    print('train_loss_func : ' + str(loss) + ', reg_er: ' + str(fc_err))
                    print ('LandMark er: : ' + str(landerror))
                # print('test_cost : '+ str(test_er)+', test_rmse : '+ str(test_mse)+ ', test_regular: '+ str(test_regular))
                # print ('test_loss_func : '+ str(test_loss_func))

                if x % 500 == 0 and x is not 0:
                    my_totoal_loss.append(loss)
                    plt.plot(my_totoal_loss)
                    plt.pause(0.01)
                if x % 5000 == 0:
                    saver.save(sess, './mymodel/landmark_model.ckpt')
                if x % 10000 == 0 and x is not 0:
                    plt.savefig('./train_fig/my_total_loss_'+str(x)+'.png')
    else:
        # saver = tf.train.import_meta_graph('./mymodel/landmark_model-400.meta')
        saver = tf.train.import_meta_graph('./mymodel/landmark_model.ckpt.meta')
        saver.restore(sess, './mymodel/landmark_model.ckpt')
        test_data = cv2.imread('/home/yoyomoon/Data/T_train/face12.png')
        # test_data = cv2.imread('/home/yoyomoon/圖片/T_face/face1050.jpg')
        img = test_data
        test_data = test_data[:, :, 0]
        test_data = cv2.resize(test_data, (40, 40))
        test_data =test_data[np.newaxis, :, :]
        yloc = sess.run(y_landmark, feed_dict={image: test_data, keep_prob: 1})
        # yloc = np.asarray([[20.25, 24.17, 18.31, 20.92, 22.81, 20.75]])
        yloc = [int((640/40)*yloc[0,0]), int((480/40)*yloc[0,1]), int((640/40)*yloc[0,2]), int((480/40)*yloc[0,3]),
                    int((640 / 40) * yloc[0,4]), int((480/40)*yloc[0,5])]
        img = cv2.circle(img, (yloc[0], yloc[1]), 3, (255, 255, 0))
        img = cv2.circle(img, (yloc[2], yloc[3]), 3, (255, 0, 255))
        img = cv2.circle(img, (yloc[4], yloc[5]), 3, (0, 0, 255))
        # print((yloc_T-yloc))
        cv2.imshow('Moon_T_face', img)
        cv2.waitKey(-1)