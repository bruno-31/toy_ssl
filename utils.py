import tensorflow as tf
import matplotlib.pyplot as plt

def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm

def accuracy(logit, y):
    pred = tf.to_float(tf.greater(logit,0.))
    return tf.reduce_mean(tf.to_float(tf.equal(pred,tf.to_float(y))))

def prediction(logit):
    pred = tf.to_float(tf.greater(logit,0.))
    return pred

def scatter_2_class(testx,pred,**kwargs):
    plt.scatter(testx[:,0][pred==0],testx[:,1][pred==0],**kwargs)
    plt.scatter(testx[:,0][pred==1],testx[:,1][pred==1], **kwargs)
    
def get_jacobian(y,x):
    with tf.name_scope("jacob"):
        grads = tf.stack([tf.gradients(yi, x)[0] for yi in tf.unstack(y, axis=1)], axis=2)
    return grads    

