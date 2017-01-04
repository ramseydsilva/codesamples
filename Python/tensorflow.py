from django.db import models, IntegrityError
from django.core.files.base import ContentFile
from django.utils.functional import cached_property
from server.core.models import NamedModelBase, ModelBase
import numpy as np
import tensorflow as tf


class NeuralNetwork(NamedModelBase):
    pass

class ConvolutionalNeuralNetwork(NeuralNetwork):

    @cached_property
    def x(self):
        return tf.placeholder(tf.float32, [None, self.obj.num_features*self.obj.num_channels])

    @cached_property
    def y(self):
        return tf.placeholder(tf.float32, [None, self.obj.num_labels])

    @cached_property
    def out_weight(self):
        input_size = self.layers.filter(active=True).last().output_size
        out_size = [input_size, self.obj.num_labels]
        return tf.Variable(tf.random_normal(out_size))

    @cached_property
    def out_bias(self):
        return tf.Variable(tf.random_normal([self.obj.num_labels]))

    # Create some wrappers for simplicity
    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    # Create model
    @cached_property
    def model(self):
        # Reshape input picture
        print('orig x', self.x)
        rx = tf.reshape(self.x, shape=[-1, self.obj.num_features, self.obj.num_channels, 1]) # 4D Tensor
        print('rx', rx)
        first = True
        for layer in self.layers.filter(active=True):
            if first:
                prev_layer = rx
                first = False
            _type =  layer._type.name.lower()

            print('before', _type, prev_layer)

            if 'conv2d' in _type:
                convx = tf.nn.conv2d(prev_layer, layer.weight, strides=[1, 1, 1, 1], padding='SAME')
                prev_layer = tf.nn.bias_add(convx, layer.bias)
            elif 'relu' in _type:
                prev_layer = tf.nn.relu(prev_layer)
            elif 'maxpool' in _type:
                prev_layer = self.maxpool2d(prev_layer, k=2) # Max Pooling (down-sampling)
            elif 'full' in _type:
                # Fully connected layer
                # Reshape conv2 output to fit fully connected layer input
                fc = tf.reshape(prev_layer, [-1, layer.input_size])
                prev_layer= tf.add(tf.matmul(fc, layer.weight), layer.bias)
            elif 'dropout' in _type:
                # Apply Dropout
                prev_layer = tf.nn.dropout(prev_layer, self.dropout)

            print('after', _type, prev_layer, layer.weight, layer.bias)

        # Output, class prediction
        print('mat mul weight add bias', prev_layer, self.out_weight, self.out_bias)
        out = tf.add(tf.matmul(prev_layer, self.out_weight), self.out_bias)
        print('out', out)
        return out

    @property
    def dropout(self):
        return 0.75

    def train(self, display_step=1):
        data = self.obj.data

        keep_prob = tf.placeholder(tf.float32)
        x = self.x
        y = self.y

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=float(self.obj.learning_rate)).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * self.obj.batch_size < self.obj.training_iterations:
                batch_x, batch_y = data.train.next_batch(self.obj.batch_size)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: self.dropout})
                if step % display_step == 0:
                    # Calculate batch loss and accuracy

                    mb_loss, mb_acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                    tr_loss, tr_acc = sess.run([cost, accuracy], feed_dict={x: data.train.images, y: data.train.labels, keep_prob: 1.})
                    v_loss, v_acc = sess.run([cost, accuracy], feed_dict={x: data.validation.images, y: data.validation.labels, keep_prob: 1.})

                    print("Iter %s \n M Accuracy %.6f Loss %.6f \n T Accuracy %.6f Loss %.6f \n V Accuracy %.6f Loss %.6f" %(step*self.obj.batch_size, 
                                            mb_acc, mb_loss, tr_acc, tr_loss, v_acc, v_loss))

                step += 1

            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: data.test.images[:256], y: data.test.labels[:256], keep_prob: 1.}))

            if not self.obj.checkpoint_file:
                self.obj.checkpoint_file.save('checkpoint', ContentFile(''))

            saver.save(sess, self.obj.checkpoint_file.path)

            # Construct model
            pred = tf.argmax(self.model, 1)
            prediction = sess.run(pred, feed_dict={x: data.train.images, y:data.train.labels, keep_prob: 1.0})
            print(prediction)
            prediction = sess.run(pred, feed_dict={x: data.validation.images, y:data.validation.labels, keep_prob: 1.0})
            print(prediction)
            prediction = sess.run(pred, feed_dict={x: data.test.images, y:data.test.labels, keep_prob: 1.0})
            print(prediction)

    def predict(self, to_predict):
        if self.obj.checkpoint_file:
            model = self.model
            saver = tf.train.Saver()
            keep_prob = tf.placeholder(tf.float32)
            x = self.x
            y = self.y
            with tf.Session() as sess:
                saver.restore(sess, self.obj.checkpoint_file.path)

                # Construct model
                pred = tf.argmax(self.model, 1)
                if type(to_predict) == list and type(to_predict[0]) != str:
                    data = np.matrix(to_predict).astype(np.float)
                    return sess.run(pred, feed_dict={x: data, keep_prob: 1.0})
                else:
                    data = self.obj.data
                    prediction = sess.run(pred, feed_dict={x: data.train.images, y:data.train.labels, keep_prob: 1.0})
                    print(prediction)
                    prediction = sess.run(pred, feed_dict={x: data.validation.images, y:data.validation.labels, keep_prob: 1.0})
                    print(prediction)
                    prediction = sess.run(pred, feed_dict={x: data.test.images, y:data.test.labels, keep_prob: 1.0})
                    print(prediction)
                    return prediction
        return []

