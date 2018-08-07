# #快速验证tensorflow　api接口的含义及参数
# import tensorflow as tf
#
# raw = tf.Variable(tf.random_normal(shape=(1, 3, 2, 1)))
# squeezed = tf.squeeze(raw,squeeze_dims=[2,3])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(raw.shape)
#     print('-----------------------------')
#     print(sess.run(squeezed).shape)


import numpy as np
import cv2

image = cv2.imread(filename)
cv2.flip(image, 1, image)