import tensorflow as tf
import math
import numpy as np
import pandas as pd
import time

logdir = '/Users/jasonchen/tmp/test'
vocabulary_size = 888

batch_size = 5
embedding_size = 10  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a context.

train_context = tf.placeholder(tf.int32, shape=(None,), name='context')
train_inputs = tf.placeholder(tf.int32, shape=(None,), name='input')

# word embeddings
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embedding')

# item vectors that is the combination of word vectors
embed_list = []
for i in range(vocabulary_size):
    indices = map(lambda x:x%vocabulary_size, range(i, i+3))
    new_vec = tf.reduce_sum(tf.nn.embedding_lookup(embeddings, indices), axis=0, name='item_vec')
    embed_list.append(new_vec)
embeddings_items = tf.stack(embed_list, name='all_item_vecs')

# user vector which is the combination of items
embed = tf.nn.embedding_lookup(embeddings_items, train_inputs, name='user_vec')

# Construct the variables for the softmax
weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)), name='sm_weight')
biases = tf.Variable(tf.zeros([vocabulary_size]), name='bias')
hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases

# convert train_context to a one-hot format
train_one_hot = tf.one_hot(train_context, vocabulary_size, name='label_one_hot')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot), name='cross_entropy')
# Construct the SGD optimizer using a learning rate of 1.0.
# optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)


with tf.Session() as sess:
    # Properly initialize all variables.
    tf.global_variables_initializer().run()

    tf.summary.scalar('loss', cross_entropy)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(logdir + '/train' + time.strftime('%x-%X').replace('/', '').replace(':', ''), sess.graph)
    last_w2v = None
    for i in range(10):
        feed_dict = {train_context:[30], train_inputs:np.random.choice(vocabulary_size, batch_size, replace=False)}
        _, loss, w2v = sess.run([optimizer, cross_entropy, embeddings], feed_dict=feed_dict)
        if last_w2v is not None:
            diff = np.sum(w2v, axis=1) - np.sum(last_w2v, axis=1)
            print 'number of word updated', np.sum(diff != 0)
        last_w2v = w2v
        print 'finish example', i