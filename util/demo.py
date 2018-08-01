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


area = 1
box_area= np.array([7,8,9,10])
inter = np.array([1,2,3,4])
ovr = inter * 1.0 / (box_area + area - inter)
print(ovr)