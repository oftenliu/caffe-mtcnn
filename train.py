import os,sys
import tensorflow as tf
from tensorflow.contrib import learn
import cv2
import numpy as np

from model import mtcnnmodel
from util import tfrecord_read
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./"))
sys.path.insert(0, rootPath)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output', '../bin/model',
                           """Directory for event logs and checkpoints""")
tf.app.flags.DEFINE_string('tune_from', '',
                           """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('tune_scope', '',
                           """Variable scope for training""")

tf.app.flags.DEFINE_integer('batch_size', 2 ** 5,
                            """Mini-batch size""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                          """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """Optimizer gradient first-order momentum""")
tf.app.flags.DEFINE_float('decay_rate', 0.9,
                          """Learning rate decay base""")
tf.app.flags.DEFINE_float('decay_steps', 2 ** 16,
                          """Learning rate decay exponent scale""")
tf.app.flags.DEFINE_boolean('decay_staircase', False,
                            """Staircase learning rate decay by integer division""")

tf.app.flags.DEFINE_integer('max_num_steps', 2 ** 21,
                            """Number of optimization steps to run""")

tf.app.flags.DEFINE_string('train_device', '/gpu:1',
                           """Device for training graph placement""")
tf.app.flags.DEFINE_string('input_device', '/gpu:0',
                           """Device for preprocess/batching graph placement""")

tf.app.flags.DEFINE_string('train_path', '../data/train/',
                           """Base directory for training data""")
tf.app.flags.DEFINE_string('filename_pattern', 'words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads', 4,
                            """Number of readers for input data""")
tf.app.flags.DEFINE_integer('width_threshold', None,
                            """Limit of input image width""")
tf.app.flags.DEFINE_integer('length_threshold', None,
                            """Limit of input string length width""")
LR_EPOCH = [6, 14, 20]

tf.logging.set_verbosity(tf.logging.INFO)

# Non-configurable parameters
optimizer = 'Adam'
mode = learn.ModeKeys.TRAIN  # 'Configure' training mode for dropout layers


# all mini-batch mirror
def random_flip_images(image_batch, label_batch, landmark_batch):
    # mirror
    if np.random.choice([0, 1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch == -2)[0]
        flipposindexes = np.where(label_batch == 1)[0]
        # only flip
        flipindexes = np.concatenate((fliplandmarkindexes, flipposindexes))
        # random flip
        for i in flipindexes:
            cv2.flip(image_batch[i], 1, image_batch[i])

            # pay attention: flip landmark
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1, 2))
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
            landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth
            landmark_batch[i] = landmark_.ravel()

    return image_batch, landmark_batch






def get_training(rnn_logits, label, sequence_length):
    """Set up training ops"""
    with tf.name_scope("train"):

        if FLAGS.tune_scope:
            scope = FLAGS.tune_scope
        else:
            scope = "convnet|rnn"

        rnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope)

        loss = rnn_logits#model.ctc_loss_layer(rnn_logits, label, sequence_length)

        # Update batch norm stats [http://stackoverflow.com/questions/43234667]
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(extra_update_ops):

            learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate,
                tf.train.get_global_step(),
                FLAGS.decay_steps,
                FLAGS.decay_rate,
                staircase=FLAGS.decay_staircase,
                name='learning_rate')

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=FLAGS.momentum)

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=learning_rate,
                optimizer=optimizer,
                variables=rnn_vars)

            tf.summary.scalar('learning_rate', learning_rate)

    return train_op


def _get_session_config():
    """Setup session config to soften device placement"""

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    return config


def _get_init_pretrained():
    """Return lambda for reading pretrained initial model"""

    if not FLAGS.tune_from:
        return None

    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    ckpt_path = FLAGS.tune_from

    init_fn = lambda sess: saver_reader.restore(sess, ckpt_path)

    return init_fn


def _get_training(baseLr, loss, data_num):
    """
    train model
    :param baseLr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op
    """
    lr_factor = 0.1

    #LR_EPOCH [8,14]
    #boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / FLAGS.batch_size) for epoch in LR_EPOCH]
    #lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [baseLr * (lr_factor ** x) for x in range(0, len(LR_EPOCH) + 1)]  #学习率衰减
    #control learning rate
    lr_op = tf.train.piecewise_constant(tf.train.get_global_step(), boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    tf.summary.scalar('learning_rate', lr_op)
    return train_op, lr_op



def main(argv=None):

    net = 'pnet'

    dataPath = os.path.join(rootPath, "tmp/data/%s" % ('pnet'))
    gpus = '1'
    # set GPU
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    if net == 'pnet': # PNet use this method to get data
        dataset_dir = os.path.join(dataPath, 'all.tfrecord')
        total_num = sum(1 for _ in tf.python_io.tf_record_iterator(dataset_dir))
        print(total_num)
    elif net in ['rnet', 'onet']: # RNet and ONet use 4 tfrecords to get data
        pos_dir = os.path.join(dataPath, 'pos.tfrecord')
        part_dir = os.path.join(dataPath, 'part.tfrecord')
        neg_dir = os.path.join(dataPath, 'neg.tfrecord')
        landmark_dir = os.path.join(dataPath, 'landmark.tfrecord')
        dataset_dirs = [pos_dir, part_dir, neg_dir, landmark_dir]
        pos_ratio, part_ratio, landmark_ratio, neg_ratio = 1.0/6, 1.0/6, 1.0/6, 3.0/6
        pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_ratio))
        part_batch_size = int(np.ceil(config.BATCH_SIZE*part_ratio))
        neg_batch_size = int(np.ceil(config.BATCH_SIZE*neg_ratio))
        landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_ratio))
        batch_sizes = [pos_batch_size, part_batch_size, neg_batch_size, landmark_batch_size]
        image_batch, label_batch, bbox_batch, landmark_batch = read_multi_tfrecords(dataset_dirs, batch_sizes, net)
        total_num = 0
        for d in dataset_dirs:
            total_num += sum(1 for _ in tf.python_io.tf_record_iterator(d))
    #ratio
    if net == 'pnet':
        image_size = 12
        ratio_cls_loss, ratio_bbox_loss, ratio_landmark_loss = 1.0, 0.5, 0.5
    elif net == 'rnet':
        image_size = 24
        ratio_cls_loss, ratio_bbox_loss, ratio_landmark_loss = 1.0, 0.5, 1.0
    elif net == 'onet':
        ratio_cls_loss, ratio_bbox_loss, ratio_landmark_loss = 1.0, 0.5, 1.0
        image_size = 48
    else:
        raise Exception("incorrect net type.")


    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        image_batch, label_batch, bbox_batch, landmark_batch = tfrecord_read.read_single_tfrecord(dataset_dir,
                                                                                                  FLAGS.batch_size, net)
        input_image = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 12, 12, 3],
                                     name='input_image')
        label = tf.placeholder(tf.float32, shape=[FLAGS.batch_size], name='label')
        bboxs_truth = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 4], name='bboxs_truth')
        landmarks_truth = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 10], name='landmarks_truth')

        with tf.device(FLAGS.train_device):
            cls_loss, bbox_loss, landmark_loss, l2_loss,accuracy_op = mtcnnmodel.mtcnn_pnet(input_image, label,bboxs_truth,landmarks_truth, mode=learn.ModeKeys.TRAIN)

            train_op,lr_op = _get_training(0.1, ratio_cls_loss*cls_loss + ratio_bbox_loss*bbox_loss + ratio_landmark_loss*landmark_loss + l2_loss, total_num)

        session_config = _get_session_config()

        tf.summary.scalar("cls_loss", cls_loss)  # cls_loss
        tf.summary.scalar("bbox_loss", bbox_loss)  # bbox_loss
        tf.summary.scalar("landmark_loss", landmark_loss)  # landmark_loss
        tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc


        summary_op = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sv = tf.train.Supervisor(
            logdir=FLAGS.output,
            init_op=init_op,
            summary_op=summary_op,
            save_summaries_secs=30,
            init_fn=_get_init_pretrained(),
            save_model_secs=150)

        with sv.managed_session(config=session_config) as sess:
            step = sess.run(global_step)
            while step < FLAGS.max_num_steps:
                if sv.should_stop():
                    break

                image_batch_array, label_batch_array, bbox_batch_array, landmark_batch_array = sess.run(
                    [image_batch, label_batch, bbox_batch, landmark_batch])
                # random flip
                image_batch_array, landmark_batch_array = random_flip_images(image_batch_array, label_batch_array,
                                                                             landmark_batch_array)
                _, _, summary,step = sess.run([train_op, lr_op, summary_op,global_step],
                                         feed_dict={input_image: image_batch_array, label: label_batch_array,
                                                    bboxs_truth: bbox_batch_array,
                                                    landmarks_truth: landmark_batch_array})
                #[step_loss, step] = sess.run([train_op, global_step])
            sv.saver.save(sess, os.path.join(FLAGS.output, 'model.ckpt'),
                          global_step=global_step)


if __name__ == '__main__':
    tf.app.run()