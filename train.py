import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model.pix2pix import Pix2pix

#tf.keras.backend.clear_session()

iters = 200*400
batch_size = 1

A = np.load('dataset_Y.npy') 
B = np.load('dataset_X.npy') 

with tf.device('/gpu:0'):
    model = Pix2pix(256, 256, ichan=3, ochan=3)

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(iters):
        a = np.expand_dims(A[step % A.shape[0]], axis=0)
        b = 2. * np.expand_dims(B[step % B.shape[0]], axis=0) - 1. 

        gloss_curr, dloss_curr = model.train_step(sess, a, a, b)
        print('Step %d: G loss: %f | D loss: %f' % (step, gloss_curr, dloss_curr))

        if step % 500 == 0:
            fig = plt.figure()
            fig.set_size_inches(10, 10)
            fig.subplots_adjust(left=0, bottom=0,
                                   right=1, top=1, wspace=0, hspace=0.1)
            p = np.random.permutation(B.shape[0])
            for i in range(0, 81, 3):

                fig.add_subplot(9, 9, i + 1)
                plt.imshow(A[p[i // 3]])
                plt.axis('off')
                fig.add_subplot(9, 9, i + 2)
                plt.imshow((model.sample_generator(sess, np.expand_dims(A[p[i // 3]], axis=0), is_training=True)[0] + 1.) / 2.)
                plt.axis('off')
                fig.add_subplot(9, 9, i +3)
                plt.imshow(B[p[i // 3]])
                plt.axis('off')
            plt.savefig('/content/images/iter_%d.jpg' % step)
            plt.close()
        if step % 3000 == 0:

            save_path = saver.save(sess, "/content/models/model.ckpt")
            print("Model saved in file: %s" % save_path)
