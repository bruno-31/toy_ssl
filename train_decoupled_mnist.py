import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from cifar_model import generator,discriminator
from data import cifar10_input


from tensorflow.examples.tutorials.mnist import input_data
from utils import *
slim = tf.contrib.slim
from tqdm import tqdm_notebook as tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from mnist_model import generator,discriminator
import utils
l = tf.layers


flags = tf.app.flags
flags.DEFINE_integer("batch_cnn", 50, "batch size [250]")
flags.DEFINE_integer("batch_gan", 50, "batch size [250]")
flags.DEFINE_integer('seed', 10, 'seed numpy')
flags.DEFINE_float('lr_gan', 1e-4, 'learning_rate[0.003]')
flags.DEFINE_float('lr_cnn', 1e-4, 'learning_rate[0.003]')

flags.DEFINE_integer('labeled', 400, 'labeled data per class')
flags.DEFINE_integer('zdim', 100, 'seed numpy')
flags.DEFINE_integer('epoch', 800, 'labeled data per class')

flags.DEFINE_float('epsilon', 5., 'learning_rate[0.003]')
flags.DEFINE_float('eta', 1., 'learning_rate[0.003]')
flags.DEFINE_float('gamma',5, 'learning_rate[0.003]')

flags.DEFINE_float('ema', 0.995, 'exp moving average for inference [0.9999]')
flags.DEFINE_string('logdir', './log/cifar', 'log directory')
flags.DEFINE_string('data_dir', '/tmp/data/cifar-10-python', 'data directory')

flags.DEFINE_boolean('large', False, 'enable manifold reg')
flags.DEFINE_boolean('vanilla', False, 'enable manifold reg')

FLAGS = flags.FLAGS



def main(_):
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.lower(), value))
    print("")
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    rng = np.random.RandomState(FLAGS.seed)  # seed labels

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    trainx = np.vstack([mnist.train.images, mnist.validation.images])
    trainy = np.hstack([mnist.train.labels, mnist.validation.labels])
    testx = mnist.test.images
    testy = mnist.test.labels
    trainx = np.reshape(trainx, [-1, 28, 28, 1])
    testx = np.reshape(testx, [-1, 28, 28, 1])
    trainx_unl = trainx.copy()

    validation = False
    if validation:
        split = int(0.1 * trainx.shape[0])
        print(split)
        testx = trainx[:split]
        testy = trainy[:split]
        trainx = trainx[split:]
        trainy = trainy[split:]

    trainx_unl = trainx.copy()
    inds = rng.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]
    txs = []
    tys = []
    for j in range(10):
        txs.append(trainx[trainy == j][:FLAGS.labeled])
        tys.append(trainy[trainy == j][:FLAGS.labeled])
    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)
    trainx = txs
    trainy = tys

    nr_batch_train = trainx.shape[0] // FLAGS.batch_gan
    nr_batch_test = testx.shape[0] // FLAGS.batch_gan
    print(trainx.shape, testx.shape)

    unl_dataset = tf.data.Dataset.from_tensor_slices(trainx_unl)
    unl_dataset = unl_dataset.shuffle(10000).repeat().batch(FLAGS.batch_gan)
    iterator_unl = unl_dataset.make_one_shot_iterator()
    next_unl = iterator_unl.get_next()

    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    data = tf.cast(next_unl, tf.float32)
    noise = tf.random_normal(shape=[FLAGS.batch_gan, 100])

    samples = generator(noise, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
    real_score = discriminator(data, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
    fake_score = discriminator(samples, is_training=is_training_pl, reuse=tf.AUTO_REUSE)

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

    loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.ones_like(fake_score)))
    loss_d = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, labels=tf.ones_like(fake_score))) + \
             tf.reduce_mean(
                 tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.zeros_like(fake_score)))

    optimizer_dis = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=0.5, name='dis_optimizer')
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=0.5, name='gen_optimizer')

    update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    update_ops_dis = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

    with tf.control_dependencies(update_ops_gen):  # attached op for moving average batch norm
        traing = optimizer_gen.minimize(loss_g, var_list=gen_vars)
    with tf.control_dependencies(update_ops_dis):
        traind = optimizer_dis.minimize(loss_d, var_list=disc_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    path = './gan_cifar_saved/'
    if tf.train.latest_checkpoint(os.path.join('./gan_mnist/')) is not None:
        path = saver.restore(sess, tf.train.latest_checkpoint(os.path.join('./gan_mnist/')))
    else:
        print('no model found')
    ###################### CNN #########################
    def lenet(x, training_pl, getter=None):
        with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE, custom_getter=getter):
            x = tf.reshape(x, [-1, 28, 28, 1])
            x = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu, padding='same')
            x = tf.layers.max_pooling2d(x, 2, 2)
            x = tf.layers.conv2d(x, 64, 5, activation=tf.nn.relu, padding='same')
            x = tf.layers.max_pooling2d(x, 2, 2)
            x = tf.reshape(x, [-1, 7 * 7 * 64])
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu)
            x = tf.layers.dropout(x, rate=0.4, training=training_pl)
            x = tf.layers.dense(x, 10)
            return x

    soft = lambda x: 1 / (1 + tf.exp(-x))

    inp = tf.placeholder(tf.float32, [FLAGS.batch_cnn, 28, 28, 1])
    lbl = tf.placeholder(tf.int64, [FLAGS.batch_cnn['batch']])
    training_cnn = tf.placeholder(tf.bool, [])
    logits = lenet(inp, training_cnn)
    xloss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=lbl)

    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')
    manifold_loss_pl = tf.placeholder(tf.float32, [])
    xloss_pl = tf.placeholder(tf.float32, [])

    ################ AMBIANT GAN ##############
    samp_logits1 = lenet(samples, training_cnn)
    p1 = tf.random_normal([FLAGS.batch_gan, 28, 28, 1], stddev=0.2)
    samp_logits2 = lenet(samples + p1, training_cnn)
    p3 = tf.random_normal([FLAGS.batch_gan, 28, 28, 1], stddev=0.2)
    samp_logits3 = lenet(samples + p3, training_cnn)

    consistency_gan = tf.reduce_sum(tf.sqrt(tf.square(samp_logits1 - samp_logits2) + 1e-8), axis=1)
    consistency_gan_loss = tf.reduce_mean(consistency_gan)

    ambient_gan = tf.reduce_sum(tf.sqrt(tf.square(samp_logits3 - samp_logits2) + 1e-8), axis=1)
    ambient_gan_loss = tf.reduce_mean(ambient_gan)

    # ################ AMBIANT UNL #################

    unl = tf.placeholder(tf.float32, [FLAGS.batch_gan, 28, 28, 1])
    unl_logits1 = lenet(unl, training_cnn)
    p2 = tf.random_normal([FLAGS.batch_gan, 28, 28, 1], stddev=0.2)
    unl_logits2 = lenet(unl + p2, training_cnn)

    consistency_unl = tf.reduce_sum(tf.sqrt(tf.square(unl_logits1 - unl_logits2) + 1e-8), axis=1)
    consistency_unl_loss = tf.reduce_mean(consistency_unl)

    kl_gan = kl_divergence_with_logit(samp_logits1, samp_logits2)
    kl_unl = kl_divergence_with_logit(unl_logits1, unl_logits2)

    # ############ MANIFOLD ##########
    z1 = tf.random_normal(shape=[FLAGS.batch_gan, 100])
    pert_n = tf.nn.l2_normalize(tf.random_normal(shape=[FLAGS.batch_gan, 100]), dim=[1])
    z1_pert = z1 + 1. * pert_n
    pz = tf.random_normal([FLAGS.batch_gan, 28, 28, 1], stddev=0.2)

    samp_z1 = generator(z1, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
    samp_z2 = generator(z1_pert, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
    samp_adv = samp_z1 + 10. * tf.nn.l2_normalize(samp_z2 - samp_z1, dim=[1, 2, 3])

    logits_z1 = lenet(samp_z1, training_cnn)
    logits_z2 = lenet(samp_z2 + pz, training_cnn)
    logits_adv = lenet(samp_adv, training_cnn)

    manifold = tf.reduce_sum(tf.sqrt(tf.square(logits_z1 - logits_adv) + 1e-8), axis=1)
    manifold_loss = tf.reduce_mean(manifold)


    if FLAGS.vanilla:
        loss = xloss
        print('vanille')
    else:
        print('mani reg')
        loss = xloss + FLAGS.gamma * manifold_loss

    with tf.variable_scope("adam", reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
        train_op = optimizer.minimize(loss, var_list=tf.trainable_variables(scope='classifier'))

        correct_prediction = tf.equal(tf.argmax(logits, 1), lbl)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
    maintain_averages_op = ema.apply(dvars)

    with tf.control_dependencies([train_op]):
        train_op = tf.group(maintain_averages_op)

    logits_ema = lenet(inp, training_cnn, getter=ema_getter)
    correct_prediction_ema = tf.equal(tf.argmax(logits_ema, 1), lbl)
    accuracy_ema = tf.reduce_mean(tf.cast(correct_prediction_ema, tf.float32))

    # summaries
    with tf.name_scope('epoch'):
        tf.summary.scalar('accuracy_train', acc_train_pl, ['epoch'])
        tf.summary.scalar('accuracy_test_moving_average', acc_test_pl_ema, ['epoch'])
        tf.summary.scalar('accuracy_test_raw', acc_test_pl, ['epoch'])
        tf.summary.scalar('xloss', xloss_pl, ['epoch'])
        tf.summary.scalar('mani loss', manifold_loss_pl, ['epoch'])
    sum_op_epoch = tf.summary.merge_all('epoch')

    # init
    var = tf.global_variables(scope='classifier') + tf.global_variables(scope='ema') + tf.global_variables(scope='adam')
    init_op = tf.variables_initializer(var_list=var)

    sess.run(init_op)
    writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

    con_loss = []
    test_array = [];
    test_ema = []

    length_epoch = 1000
    nr_batch_train = length_epoch // FLAGS.batch_cnn
    print(nr_batch_train)
    print("epoch length:", length_epoch, ", batch size:", FLAGS.batch_cnn,
          ", nr_batch_train:", nr_batch_train, ", mc size:", params['batch_size'], ",total grad iter:",
          length_epoch * FLAGS.epoch)

    for epoch in tqdm(range(FLAGS.epoch+1)):
        train_acc = train_loss = test_acc = train_manifold = test_acc_ema = 0
        trainx = [];
        trainy = []
        for t in range(int(np.ceil(length_epoch / float(txs.shape[0])))):  # same size lbl and unlb
            inds = np.random.permutation(txs.shape[0])
            trainx.append(txs[inds])
            trainy.append(tys[inds])
        trainx = np.concatenate(trainx, axis=0)
        trainy = np.concatenate(trainy, axis=0)
        trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]  # shuffling unl dataset
        #     print(trainx.shape)
        for t in range(nr_batch_train):
            #         print(t)
            ran_from = t * FLAGS.batch_cnn
            ran_to = (t + 1) * FLAGS.batch_cnn
            ran_from_mc = t * FLAGS.batch_gan
            ran_to_mc = (t + 1) * FLAGS.batch_gan
            #         print(ran_from,ran_to)

            xl, _, acc, ml = sess.run([xloss, train_op, accuracy, manifold_loss],
                                       feed_dict={inp: trainx[ran_from:ran_to],
                                                  lbl: trainy[ran_from:ran_to],
                                                  unl: trainx_unl[ran_from_mc:ran_to_mc],
                                                  training_cnn: True,
                                                  is_training_pl: False})
            train_acc += acc;
            train_loss += xl;
            train_manifold += ml

        train_manifold /= nr_batch_train
        train_acc /= nr_batch_train;
        train_loss /= nr_batch_train

        for t in range(nr_batch_test):
            ran_from = t *  FLAGS.batch_cnn
            ran_to = (t + 1) *  FLAGS.batch_cnn
            xl, acc, acc_ema = sess.run([xloss, accuracy, accuracy_ema], feed_dict={inp: testx[ran_from:ran_to],
                                                                                    lbl: testy[ran_from:ran_to],
                                                                                    training_cnn: False})
            test_acc += acc;
            test_acc_ema += acc_ema

        test_acc /= nr_batch_test;
        test_acc_ema /= nr_batch_test;
        test_ema.append(test_acc_ema)
        print("Epoch: {}, xloss: {:.5f}, training acc: {:.2f}%, test acc: {:.2f}%, test acc ema: {:.2f}%".format(
            epoch, train_loss, train_acc * 100, test_acc * 100, test_acc_ema * 100))


        sum = sess.run(sum_op_epoch, feed_dict={acc_train_pl: train_acc,
                                                acc_test_pl: test_acc,
                                                acc_test_pl_ema: test_acc_ema,
                                                manifold_loss_pl: train_manifold,
                                                xloss_pl: train_loss})
        writer.add_summary(sum, epoch)


if __name__ == '__main__':
    tf.app.run()