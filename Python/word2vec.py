from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.utils.html import strip_tags
import tensorflow as tf
import numpy as np
import IPython, collections, math, os, random, zipfile, pylab, time
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.models.embedding.word2vec_optimized import *
from products.models import Product
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from tensorflow.models.rnn.ptb.reader import ptb_producer, _build_vocab, _file_to_word_ids

logging = tf.logging


def data_type():
    return tf.float32

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = ptb_producer(
        data, batch_size, num_steps, name=name)

class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = [tf.squeeze(input_step, [1])
        #           for input_step in tf.split(1, num_steps, inputs)]
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class Config(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 10
    hidden_size = 20
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 200
    vocab_size = 30000

def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)

def get_config():
    return Config()

class Command(BaseCommand):
    help = "Predict search term"
    save_path = settings.PROJECT_PATH + "/ml/data/word2vec"

    def add_arguments(self, parser):
        #parser.add_argument('--query', type=str, default='', help="String to Predict", nargs="+")
        pass

    def write_words_to_file(self):
        open('text9', 'w').close()
        f = open('text9', 'w')
        for product in Product.objects.all():
            f.write(product.name.lower() + ". ")
            if False and product.description:
                f.write(strip_tags(product.description.lower()))
                f.write(product.name.lower() + ". ")
        f.close()

    def word2vec(self):
        opts = Options()
        opts.train_data = 'text9'
        opts.save_path = self.save_path
        opts.eval_data = 'questions-words.txt'
        opts.epochs_to_train = 15

        self.write_words_to_file()

        with tf.Graph().as_default(), tf.Session() as session:
            with tf.device("/cpu:0"):
                model = Word2Vec(opts, session)
                model.read_analogies() # Read analogy questions
            for _ in xrange(opts.epochs_to_train):
                model.train()  # Process one epoch

    def handle(self, **kwargs):
        self.write_words_to_file()
        word_to_id = _build_vocab('text9')
        train_data = _file_to_word_ids('text9', word_to_id)
        vocabulary = len(word_to_id)
        print("Vocbulary size:", len(word_to_id))
        print(train_data)

        config = get_config()
        eval_config = get_config()
        eval_config.batch_size = 1
        eval_config.num_steps = 1

        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.scalar_summary("Training Loss", m.cost)
            tf.scalar_summary("Learning Rate", m.lr)

        sv = tf.train.Supervisor(logdir='logs')
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                         verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            print("Saving model to %s." % self.save_path)
            sv.saver.save(session, self.save_path, global_step=sv.global_step)

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
            labels = [reverse_dictionary[i] for i in xrange(plot_only)]
            plot_with_labels(low_dim_embs, labels)


            IPython.start_ipython(argv=[], user_ns=locals().update(globals()))
