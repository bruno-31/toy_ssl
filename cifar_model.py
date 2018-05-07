import tensorflow as tf
l = tf.layers

def leaky_relu(features, alpha=0.2, name=None):
  return tf.maximum(alpha * features, features, name)

def generator(inputs, is_training=False, reuse=False, name='generator'):
    with tf.variable_scope(name, reuse=reuse):
        normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        
        # Expand input to [batch, 1, 1, channels]
        inputs = tf.expand_dims(tf.expand_dims(inputs, 1), 1)
        
        # Transposed convolution outputs [batch, 4, 4, 1024]
        net = tf.layers.conv2d_transpose(inputs, 1024, 4, padding='valid',
                                         kernel_initializer=normal_initializer,
                                        name='tconv1')
        net = tf.layers.batch_normalization(net, training=is_training,
                                            name='tconv1/batch_normalization')
        net = tf.nn.relu(net, name='tconv1/relu')
        
        # Transposed convolution outputs [batch, 8, 8, 256]
        net = tf.layers.conv2d_transpose(net, 256, 4, 2, padding='same',
                                         kernel_initializer=normal_initializer,
                                         name='tconv2')
        net = tf.layers.batch_normalization(net, training=is_training,
                                            name='tconv2/batch_normalization')
        net = tf.nn.relu(net, name='tconv2/relu')
        
        # Transposed convolution outputs [batch, 16, 16, 64]
        net = tf.layers.conv2d_transpose(net, 64, 4, 2, padding='same',
                                         kernel_initializer=normal_initializer,
                                         name='tconv3')
        net = tf.layers.batch_normalization(net, training=is_training,
                                            name='tconv3/batch_normalization')
        net = tf.nn.relu(net, name='tconv3/relu')
        
        # Transposed convolution outputs [batch, 32, 32, 3]
        net = tf.layers.conv2d_transpose(net, 3, 4, 2, padding='same',
                                         kernel_initializer=normal_initializer,
                                         name='tconv4')
        net = tf.tanh(net, name='tconv4/tanh')
        
        return net

def discriminator(inputs, is_training=False, reuse=False, name='discriminator'):
    with tf.variable_scope(name, reuse=reuse):
        normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        
        # Convolution outputs [batch, 16, 16, 64]
        net = tf.layers.conv2d(inputs, 64, 4, 2, padding='same',
                               kernel_initializer=normal_initializer,
                               name='conv1')
        net = leaky_relu(net, 0.2, name='conv1/leaky_relu')
        
        # Convolution outputs [batch, 8, 8, 256]
        net = tf.layers.conv2d(net, 256, 4, 2, padding='same',
                               kernel_initializer=normal_initializer,
                               name='conv2')
        net = tf.layers.batch_normalization(net, training=is_training,
                                            name='conv2/batch_normalization')
        net = leaky_relu(net, 0.2, name='conv2/leaky_relu')
        
        # Convolution outputs [batch, 4, 4, 1024]
        net = tf.layers.conv2d(net, 1024, 4, 2, padding='same',
                               kernel_initializer=normal_initializer,
                               name='conv3')
        net = tf.layers.batch_normalization(net, training=is_training,
                                            name='conv3/batch_normalization')
        net = leaky_relu(net, 0.2, name='conv3/leaky_relu')
        
        # Convolution outputs [batch, 1, 1, 1]
        net = tf.layers.conv2d(net, 1, 4, padding='valid',
                               kernel_initializer=normal_initializer,
                               name='conv4')
        
        # Squeeze height and width dimensions
        net = tf.squeeze(net, [1, 2, 3])
        
        return net
    
def lrelu(x,alpha=0.1):
    return tf.nn.leaky_relu(x,alpha=alpha)

def cnn(x, is_training):
    with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
        x = lrelu(l.batch_normalization(l.conv2d(x,96,3,padding='same'),training=is_training))
        x = lrelu(l.batch_normalization(l.conv2d(x,96,3,padding='same'),training=is_training))
        x = lrelu(l.batch_normalization(l.conv2d(x,96,3,padding='same'),training=is_training))
        x = l.max_pooling2d(x,2,2)
        x = l.dropout(x,training=is_training)
        x = lrelu(l.batch_normalization(l.conv2d(x,128,3,padding='same'),training=is_training))
        x = lrelu(l.batch_normalization(l.conv2d(x,128,3,padding='same'),training=is_training))
        x = lrelu(l.batch_normalization(l.conv2d(x,128,3,padding='same'),training=is_training))
        x = l.max_pooling2d(x,2,2)
        x = l.dropout(x,training=is_training)
        x = lrelu(l.batch_normalization(l.conv2d(x,256,3,padding='valid'),training=is_training))
        x = lrelu(l.batch_normalization(l.conv2d(x,128,1,padding='same'),training=is_training))
        x = lrelu(l.batch_normalization(l.conv2d(x,128,1,padding='same'),training=is_training))
        x = tf.squeeze(l.average_pooling2d(x,6,6))# global avg pool
        x = l.dense(x,10)
        return x