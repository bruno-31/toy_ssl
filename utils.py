import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

def plot_boundaries():
    x_min =-1.5; x_max = 2.5; y_min = -1; y_max = 1.5; h = 0.01;
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.vstack([xx.ravel(),yy.ravel()]).T

    zz = sess.run(logits_pl,{x_pl:grid})
    softmax = lambda x: 1/(1+np.exp(-x))
    zz=softmax(zz)
    _xx, mm = sess.run([samp,tan_consistency])

    n_test = len(pred)
    fig = plt.figure(figsize=(15,5))
    plt.subplot(131)
    cnt = plt.contourf(xx, yy, zz.reshape(xx.shape),200,cmap=plt.cm.coolwarm, alpha=0.4)
    # plt.contour(xx, yy, zz.reshape(xx.shape),1,map=plt.cm.binary, alpha=0.5,linewidths=2,linestyles='solid')
    utils.scatter_2_class(testx[:n_test],pred,s=8)
    utils.scatter_2_class(trainx,trainy, marker='D',s=100)
    plt.xlim([-1.5,2.5]);plt.ylim([-1,1.5])

    # This is the fix for the white lines between contour levels
    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(1e-9)

    plt.subplot(132)
    plt.scatter(_xx[:,0],_xx[:,1], s=50, alpha=1.,c=mm,cmap='Reds',norm=None,linewidths=0.1,edgecolors='b')

    plt.subplot(133)
    plt.plot([j*params_dnn['gamma'] for j in jls])
    plt.plot(xls)

    
def grid_x(X):
    # [-1, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = ((X + 1.) * 127.5).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples // rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw))

    for n, x in enumerate(X):
        j = n // nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x

    return img

def show_digits(digits_flatten):
    X = grid_x(digits_flatten.reshape([-1,28,28]))
    plt.imshow(X,cmap='gray')
    plt.axis('off')
    
    
  