import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from cifar_model import generator,discriminator
from cifar_model import convnet as cnn
import utils
from data import cifar10_input


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

flags.DEFINE_float('ema', 0.999, 'exp moving average for inference [0.9999]')
flags.DEFINE_string('logdir', './log/cifar', 'log directory')
flags.DEFINE_string('data_dir', '/tmp/data/cifar-10-python', 'data directory')
# flags.DEFINE_integer('seed', 10, 'seed numpy')
# flags.DEFINE_integer('seed_data', 10, 'seed data')
# flags.DEFINE_integer('labeled', 400, 'labeled data per class')
# flags.DEFINE_float('learning_rate', 0.0003, 'learning_rate[0.003]')
# flags.DEFINE_float('unl_weight', 1.0, 'unlabeled weight [1.]')
# flags.DEFINE_float('lbl_weight', 1.0, 'unlabeled weight [1.]')
# flags.DEFINE_float('ema', 0.9999, 'exp moving average for inference [0.9999]')
#
# flags.DEFINE_float('scale', 1e-5, 'scale perturbation')
# flags.DEFINE_float('nabla_w', 1e-3, 'weight regularization')
# flags.DEFINE_integer('decay_start', 1200, 'start of learning rate decay')
# flags.DEFINE_integer('epoch', 1400, 'labeled data per class')
# flags.DEFINE_boolean('nabla', True, 'enable manifold reg')
#
# flags.DEFINE_float('gamma', 0.01, 'weight regularization')
# flags.DEFINE_float('epsilon', 20., 'displacement along data manifold')
# flags.DEFINE_float('eta', 1., 'perturbation latent code')
#
# flags.DEFINE_integer('freq_print', 10000, 'frequency image print tensorboard [10000]')
# flags.DEFINE_integer('step_print', 50, 'frequency scalar print tensorboard [50]')
# flags.DEFINE_integer('freq_test', 1, 'frequency test [500]')
# flags.DEFINE_integer('freq_save', 50, 'frequency saver epoch[50]')

FLAGS = flags.FLAGS



def main(_):
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.lower(), value))
    print("")
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)
    filename = "gamma"+str(FLAGS.gamma)+"_epsilon"+str(FLAGS.epsilon)+"_eta"+str(FLAGS.eta)
    print('filename:   '+filename)
    rng = np.random.RandomState(FLAGS.seed)  # seed labels

    (trainx, trainy), (testx, testy) = tf.keras.datasets.cifar10.load_data()
    # trainx, trainy = cifar10_input._get_dataset(FLAGS.data_dir, 'train')  # float [-1 1] images
    # testx, testy = cifar10_input._get_dataset(FLAGS.data_dir, 'test')
    def rescale(mat):
        return ((-127.5 + mat) / 127.5)
    #
    trainx = rescale(trainx)
    testx = rescale(testx)
    trainy = np.squeeze(trainy)
    testy = np.squeeze(testy)
    trainx_unl = trainx.copy()
    nr_batch_unl = trainx_unl.shape[0] // FLAGS.batch_gan

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
    nr_batch_train = trainx.shape[0] // FLAGS.batch_cnn
    nr_batch_test = testx.shape[0] // FLAGS.batch_cnn
    print('train:', trainx.shape, 'test:', testx.shape)

    unl = tf.placeholder(tf.float32,[FLAGS.batch_gan, 32, 32, 3])
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    noise = tf.random_normal(shape=[FLAGS.batch_gan, FLAGS.zdim])

    samples = generator(noise, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
    real_score = discriminator(unl, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
    fake_score = discriminator(samples, is_training=is_training_pl, reuse=tf.AUTO_REUSE)

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

    loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.ones_like(fake_score)))
    loss_d = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, labels=tf.ones_like(fake_score))) + \
             tf.reduce_mean(
                 tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.zeros_like(fake_score)))

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_gan, beta1=0.5)

    update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    update_ops_dis = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

    with tf.control_dependencies(update_ops_gen):  # attached op for moving average batch norm
        traing = optimizer.minimize(loss_g, var_list=gen_vars)
    with tf.control_dependencies(update_ops_dis):
        traind = optimizer.minimize(loss_d, var_list=disc_vars)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    path = './gan_cifar_saved/'
    if tf.train.latest_checkpoint(path) is not None:
        path = saver.restore(sess, tf.train.latest_checkpoint(path))
        print('model restored')
    else:
        print('no model found')

    # ############ MANIFOLD ##########
    z1 = tf.random_normal(shape=[FLAGS.batch_gan, FLAGS.zdim])
    pert_n = tf.nn.l2_normalize(tf.random_normal(shape=[FLAGS.batch_gan, FLAGS.zdim]), dim=[1])
    z1_pert = z1 + FLAGS.eta * pert_n
    pz = tf.random_normal([FLAGS.batch_gan, 32, 32, 1], stddev=0.2)

    samp_z1 = generator(z1, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
    samp_z2 = generator(z1_pert, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
    samp_adv = samp_z1 + FLAGS.epsilon * tf.nn.l2_normalize(samp_z2 - samp_z1, dim=[1, 2, 3])

    ################## CNN #########
    inp = tf.placeholder(tf.float32, [FLAGS.batch_cnn, 32, 32, 3])
    lbl = tf.placeholder(tf.int64, [FLAGS.batch_cnn])
    training_cnn = tf.placeholder(tf.bool, [])
    logits = cnn(inp, training_cnn)
    xloss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=lbl)

    logits_z1 = cnn(samp_z1, training_cnn)
    logits_z2 = cnn(samp_z2 + pz, training_cnn)
    logits_adv = cnn(samp_adv, training_cnn)  # ADD PZ

    manifold = tf.reduce_sum(tf.sqrt(tf.square(logits_z1 - logits_adv) + 1e-8), axis=1)
    manifold_loss = tf.reduce_mean(manifold)

    loss = xloss + FLAGS.gamma * manifold_loss

    with tf.variable_scope("adam", reuse=tf.AUTO_REUSE):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_cnn)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                       scope='classifier')  # control dependencies for batch norm ops
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, var_list=tf.trainable_variables(scope='classifier'),
                                          global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(logits, 1), lbl)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ema)
    dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
    maintain_averages_op = ema.apply(dvars)

    with tf.control_dependencies([train_op]):
        train_op = tf.group(maintain_averages_op)

    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    logits_ema = cnn(inp, training_cnn, getter=ema_getter)
    correct_prediction_ema = tf.equal(tf.argmax(logits_ema, 1), lbl)
    accuracy_ema = tf.reduce_mean(tf.cast(correct_prediction_ema, tf.float32))

    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')
    manifold_loss_pl = tf.placeholder(tf.float32, [])
    xloss_pl = tf.placeholder(tf.float32, [])

    # summaries
    with tf.name_scope('epoch'):
        tf.summary.scalar('accuracy_train', acc_train_pl, ['epoch'])
        tf.summary.scalar('accuracy_test_moving_average', acc_test_pl_ema, ['epoch'])
        tf.summary.scalar('accuracy_test_raw', acc_test_pl, ['epoch'])
        tf.summary.scalar('xloss', xloss_pl, ['epoch'])
        tf.summary.scalar('mani loss', manifold_loss_pl, ['epoch'])
    sum_op_epoch = tf.summary.merge_all('epoch')

    # init
    var = tf.global_variables(scope='classifier') + tf.global_variables(scope='adam')
    init_op = tf.variables_initializer(var_list=var)

    sess.run(init_op)
    writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir,filename), sess.graph)

    for epoch in tqdm(range(FLAGS.epoch)):
        inds = rng.permutation(trainx.shape[0])
        trainx = trainx[inds]
        trainy = trainy[inds]
        train_acc = test_acc = train_loss = train_manifold = test_acc_ema = 0

        for step in range(nr_batch_train):
            ran_from = step * FLAGS.batch_cnn
            ran_to = (step + 1) * FLAGS.batch_cnn
            xl, _, acc, ml = sess.run([xloss, train_op, accuracy, manifold_loss], {inp: trainx[ran_from:ran_to],
                                                                                   lbl: trainy[ran_from:ran_to],
                                                                                   is_training_pl: False,
                                                                                   training_cnn: True})
            train_acc += acc;
            train_loss += xl;
            train_manifold += ml
        train_acc /= nr_batch_train;
        train_loss /= nr_batch_train;
        train_manifold /= nr_batch_train

        for step in range(nr_batch_test):
            ran_from = step * FLAGS.batch_cnn
            ran_to = (step + 1) * FLAGS.batch_cnn
            acc, acc_ema = sess.run([accuracy, accuracy_ema], {inp: testx[ran_from:ran_to],
                                                               lbl: testy[ran_from:ran_to],
                                                               training_cnn: False})
            test_acc += acc;
            test_acc_ema += acc_ema
        test_acc /= nr_batch_test;
        test_acc_ema /= nr_batch_test

        print(
            "Epoch: {},global_stp: {}, xloss: {:.5f}, maniloss: {:.5f}, train: {:.2f}%, test: {:.2f},test_ma: {:.2f}%".format(
                epoch, sess.run(global_step), train_loss, train_manifold, train_acc * 100, test_acc * 100,
                                                                          test_acc_ema * 100))
        sum = sess.run(sum_op_epoch, feed_dict={acc_train_pl: train_acc,
                                                acc_test_pl: test_acc,
                                                acc_test_pl_ema: test_acc_ema,
                                                manifold_loss_pl: train_manifold,
                                                xloss_pl: train_loss})
        writer.add_summary(sum, epoch)

if __name__ == '__main__':
    tf.app.run()