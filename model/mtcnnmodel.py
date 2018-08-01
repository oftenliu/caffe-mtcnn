"""
mtcnn模型
"""
from model import netlayer
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import slim
pnet_params = [[10, 3, 1, 'valid', 'conv1', 'relu'],  # pool
                [16, 3, 1, 'valid', 'conv2', 'relu'],
                [32, 3, 1, 'valid', 'conv3', 'relu'],
                [2, 1, 1, 'valid', 'class_pred', 'softmax'],
                [4, 1, 1, 'valid', 'bbox_pred', 'none'],
                [10, 1, 1, 'valid', 'landmark_pred', 'none']
                ]



KEEP_RATE = 0.7


#封装netlayer的卷积层　　调用时参数过多　　使用数组传参
def mtcnnconv(input, params, training):
    output = netlayer.conv_layer(input, params[0], params[1], params[2], params[3], params[4], params[5], training)  # 30,30
    return output


"""
#　计算训练过程中的分类损失 使用正例和负例　labels = 1 labels =0 
param class_pred: 网络输出的类别预测值　
param labels: 标注值

output:　难例损失值
"""
def class_ohem(class_pred,labels):
    #class_pred [batch,class_num=2]
    prednum = tf.size(class_pred) #所有预测值数量　batch*class_num
    class_pred_reshape = tf.reshape(class_pred,[prednum,-1])  #reshape成１维的tensor


    #交叉熵损失　　只使用标注值对应的预测值计算损失　标注值有四种　　非正例都当成负例处理
    zeros = tf.zeros_like(labels)
    labels_filter = tf.where(tf.less(labels,0),zeros,labels) #小于０的label赋值为０
    label_int = tf.cast(labels_filter, tf.int32)             #类型转换


    num_row = tf.to_int32(class_pred.get_shape()[0])   #sample数量
    row = tf.range(num_row)*2    #每个sample两个预测值　　row为该sample预测值起始位置
    indices_ = row + label_int   #根据标注值　获取正确预测值得位置
    label_prob = tf.squeeze(tf.gather(class_pred_reshape, indices_))  #筛选用于计算loss的值


    #计算loss
    loss = -tf.log(label_prob + 1e-10)  #1e-10 防止label_prob = 0

    #只使用正例和负例进行loss的最终计算
    ones = tf.ones_like(labels)
    valid_indexs = tf.where(tf.less(labels,0),zeros,ones)
    pos_neg_loss = loss * valid_indexs

    #难例在线挖掘
    num_valids = tf.reduce_sum(valid_indexs)
    sess = tf.Session()
    print(sess.run(num_valids))
    print("12343462535254352352364464")
    keep_num = tf.cast(num_valids*KEEP_RATE,tf.int32)  #使用损失值位于前７０％loss进行计算

    hard_loss, _ = tf.nn.top_k(pos_neg_loss, k=keep_num)  #难例损失
    return tf.reduce_mean(hard_loss)




"""
#　计算训练过程中的目标框回归损失 使用正例和部分人脸样例例　labels = 1 labels = -1
param bbox_pred: 网络输出的目标框预测值　
param bbox_truth: 真实目标框坐标
param labels: 类别标注值

output:　难例损失值
"""
def bbox_ohem(bbox_pred,bbox_truth,labels):
    zeros = tf.zeros_like(labels)
    ones = tf.ones_like(labels)

    valid_indexs = tf.where(tf.equal(tf.abs(labels),1),ones,zeros)  #使用pos 和　part计算目标框回归损失

    bbox_square_loss = tf.square(bbox_pred-bbox_truth) #每个sample的box有四个值　二维
    bbox_square_loss = tf.reduce_sum(bbox_square_loss,axis=1)  #将每个sample的目标框左上角坐标值　宽　高的loss累加起来　
    bbox_square_loss = bbox_square_loss * valid_indexs      #使用pos 和　part计算目标框回归损失

    num_valids = tf.reduce_sum(valid_indexs)
    keep_num = tf.cast(num_valids * KEEP_RATE,tf.int32)

    hard_loss, _ = tf.nn.top_k(bbox_square_loss, k=keep_num)  #难例损失
    return tf.reduce_mean(hard_loss)




"""
#　计算训练过程中的目标框回归损失 使用正例和部分人脸样例例　labels = 1 labels = -1
param bbox_pred: 网络输出的目标框预测值　
param bbox_truth: 真实目标框坐标
param labels: 类别标注值

output:　难例损失值
"""
def landmark_ohem(landmark_pred,landmark_truth,labels):
    #keep label =-2  then do landmark detection

    ones = tf.ones_like(labels,dtype=tf.float32)
    zeros = tf.zeros_like(labels,dtype=tf.float32)

    valid_indexs = tf.where(tf.equal(labels,-2),ones,zeros)  #使用landmark  计算人脸关键点坐标损失

    landmark_square_loss = tf.square(landmark_pred-landmark_truth)
    landmark_square_loss = tf.reduce_sum(landmark_square_loss,axis=1)
    landmark_square_loss = landmark_square_loss * valid_indexs  #使用landmark  计算人脸关键点坐标损失

    num_valid = tf.reduce_sum(valid_indexs)
    keep_num = tf.cast(num_valid*KEEP_RATE,dtype=tf.int32)
    hard_loss, _ = tf.nn.top_k(landmark_square_loss, k=keep_num)  #难例损失
    return tf.reduce_mean(hard_loss)



def cal_accuracy(cls_prob,label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op



"""
#mtcnn p-net网络
param input: 输入层数据　[batch, in_height, in_width, in_channels]
param labels: 类别标签　
param bboxs:　真实目标框
param landmarks:人脸关键点坐标标注
param mode: learn.ModeKeys.TRAIN　　

output:　卷积层输出　[batch, out_height, out_width, filternum]
"""
def mtcnn_pnet(inputs, labels=None,bboxs_truth=None,landmarks_truth=None, training=False):
    """Build convolutional network layers attached to the given input tensor"""


    with tf.variable_scope("mtcnn_pnet"):
        conv1 = mtcnnconv(inputs, pnet_params[0], training)
        pool1 = netlayer.maxpool_layer(conv1, [2, 2], 2, 'same', 'pool1')
        conv2 = mtcnnconv(pool1, pnet_params[1], training)
        conv3 = mtcnnconv(conv2, pnet_params[2], training)
        class_pred = mtcnnconv(conv3, pnet_params[3], training)

        bbox_pred = mtcnnconv(conv3, pnet_params[4], training)  # 7,14
        landmark_pred = mtcnnconv(conv3, pnet_params[5], training)  # 7,14
        net = slim.conv2d(landmark_pred, num_outputs=16, kernel_size=[3, 3], stride=1, scope='conv2')
        if training:
            #batch*2
            cls_prob = tf.squeeze(class_pred, [1,2], name='cls_prob') #训练时输出w,h为１＊１
            cls_loss = class_ohem(cls_prob, labels)
            #batch
            bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred, bboxs_truth, labels)
            #batch*10
            landmark_pred = tf.squeeze(landmark_pred, [1, 2], name="landmark_pred")
            landmark_loss = landmark_ohem(landmark_pred, landmarks_truth, labels)

            l2_loss = tf.add_n( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            accuracy = cal_accuracy(cls_prob, labels)
            #l2_loss = tf.losses.get_regularization_loss()

            return cls_loss, bbox_loss, landmark_loss, l2_loss,accuracy
        else: # testing
            #when test, batch_size = 1
            cls_pro_test = tf.squeeze(class_pred, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred, axis=0)
            landmark_pred_test = tf.squeeze(landmark_pred, axis=0)
            return cls_pro_test, bbox_pred_test, landmark_pred_test




rnet_params = [[28, 3, 1, 'valid', 'conv1', 'relu'],  # pool
                [48, 3, 1, 'valid', 'conv2', 'relu'],
                [64, 2, 1, 'valid', 'conv3', 'relu']
                ]



"""
#mtcnn r-net网络
param input: 输入层数据　[batch, in_height, in_width, in_channels]
param labels: 类别标签　
param bboxs:　真实目标框
param landmarks:人脸关键点坐标标注
param mode: learn.ModeKeys.TRAIN　　

output:　卷积层输出　[batch, out_height, out_width, filternum]
"""
def mtcnn_rnet(inputs, labels=None,bboxs_truth=None,landmarks_truth=None, training=False):


    with tf.variable_scope("mtcnn_rnet"):
        conv1 = mtcnnconv(inputs, rnet_params[0], training)

        pool1 = netlayer.maxpool_layer(conv1, [2, 2], 2, 'same', 'pool1')

        conv2 = mtcnnconv(pool1, rnet_params[1], training)

        pool2 = netlayer.maxpool_layer(conv2, [3, 3], 2, 'same', 'pool2')

        conv3 = mtcnnconv(pool2, rnet_params[2], training)

        fc_flatten = netlayer.flatten_layer(conv3,scope='flatten')
        fc1 = netlayer.dense_layer(fc_flatten, outputnum=128, scope="fc1", activation='relu', training=training)
        #batch*2
        cls_prob = netlayer.dense_layer(fc1, outputnum=2, scope="cls_fc", activation='softmax', training=training)
        #batch*4
        bbox_pred = netlayer.dense_layer(fc1, outputnum=4, scope="bbox_fc", activation='none', training=training)
        #batch*10
        landmark_pred = netlayer.dense_layer(fc1, outputnum=10, scope="landmark_fc", activation='none', training=training)
        #train
        if training:
            cls_loss = class_ohem(cls_prob,labels)
            bbox_loss = bbox_ohem(bbox_pred,bboxs_truth,labels)
            landmark_loss = landmark_ohem(landmark_pred,landmarks_truth,labels)
            accuracy = cal_accuracy(cls_prob,labels)
            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return cls_loss,bbox_loss,landmark_loss,l2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred





onet_params = [[32, 3, 1, 'valid', 'conv1', 'relu'],  # pool
                [64, 3, 1, 'valid', 'conv2', 'relu'],
                [64, 3, 1, 'valid', 'conv3', 'relu'],
                [128, 2, 1, 'valid', 'conv4', 'relu'],
                ]

"""
#mtcnn o-net网络
param input: 输入层数据　[batch, in_height, in_width, in_channels]
param labels: 类别标签　
param bboxs:　真实目标框
param landmarks:人脸关键点坐标标注
param mode: learn.ModeKeys.TRAIN　　

output:　卷积层输出　[batch, out_height, out_width, filternum]
"""
def mtcnn_onet(inputs, labels=None,bboxs_truth=None,landmarks_truth=None, training=False):



    with tf.variable_scope("mtcnn_onet"):
        conv1 = mtcnnconv(inputs, onet_params[0], training)

        pool1 = netlayer.maxpool_layer(conv1, [2, 2], 2, 'same', 'pool1')

        conv2 = mtcnnconv(pool1, onet_params[1], training)

        pool2 = netlayer.maxpool_layer(conv2, [2, 2], 2, 'same', 'pool2')

        conv3 = mtcnnconv(pool2, onet_params[2], training)

        pool3 = netlayer.maxpool_layer(conv3, [2, 2], 2, 'same', 'pool3')

        conv4 = mtcnnconv(pool3, onet_params[3], training)

        fc_flatten = netlayer.flatten_layer(conv4,scope='flatten')

        fc1 = netlayer.dense_layer(fc_flatten, outputnum=256, scope="fc1", activation='relu', training=training)
        #batch*2
        cls_prob = netlayer.dense_layer(fc1, outputnum=2, scope="cls_fc", activation='softmax', training=training)
        #batch*4
        bbox_pred = netlayer.dense_layer(fc1, outputnum=4, scope="bbox_fc", activation='none', training=training)
        #batch*10
        landmark_pred = netlayer.dense_layer(fc1, outputnum=10, scope="landmark_fc", activation='none', training=training)
        #train
        if training:
            cls_loss = class_ohem(cls_prob,labels)
            bbox_loss = bbox_ohem(bbox_pred,bboxs_truth,labels)
            landmark_loss = landmark_ohem(landmark_pred,landmarks_truth,labels)
            accuracy = cal_accuracy(cls_prob,labels)
            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return cls_loss,bbox_loss,landmark_loss,l2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred
