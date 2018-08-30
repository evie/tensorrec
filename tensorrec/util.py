import math
import numpy as np
import random
import scipy.sparse as sp
import six
import tensorflow as tf
from weakref import WeakKeyDictionary
import os
import psutil

from .input_utils import create_tensorrec_dataset_from_sparse_matrix, create_tensorrec_dataset_from_tfrecord


def sample_items_stratified(n_items, n_users, n_sampled_items,
                 tf_interaction_rows, tf_interaction_cols, tf_interaction_values,replace):
    users = {}
    for i, u in enumerate(tf_interaction_rows):
        if tf_interaction_values[i] > 0.:
            interact = users.get(u, [])
            interact.append(tf_interaction_cols[i])
            users[u] = interact

    max_pos_num = int(0.5*n_sampled_items)
    print 'interact:', tf_interaction_values.shape, 'users:', len(set(tf_interaction_rows)), 'items:', len(set(tf_interaction_cols))

    def get_item(uid, users):
        interact = users.get(uid, [])
        if interact and len(interact) > max_pos_num:
            interact = np.random.choice(interact, size=max_pos_num, replace=replace).tolist()
        others = np.random.choice(a=n_items, size=n_sampled_items, replace=replace)
        others = [x for x in others if x not in interact]
        # print 'pos', len(interact)
        interact.extend(others[:n_sampled_items-len(interact)])
        # print 'all', len(interact)
        return np.array(interact)

    items_per_user = [get_item(uid, users) for uid in range(n_users)]
    sample_indices = []
    for user, users_items in enumerate(items_per_user):
        for item in users_items:
            sample_indices.append((user, item))

    return np.array(sample_indices, np.int64)


def sample_items(n_items, n_users, n_sampled_items, replace):
    items_per_user = [np.random.choice(a=n_items, size=n_sampled_items, replace=replace)
                      for _ in range(n_users)]

    sample_indices = []
    for user, users_items in enumerate(items_per_user):
        for item in users_items:
            sample_indices.append((user, item))

    return np.array(sample_indices, np.int64)


def calculate_batched_alpha(num_batches, alpha):
    if num_batches < 1:
        raise ValueError('num_batches must be >=1, num_batches={}'.format(num_batches))
    elif num_batches > 1:
        batched_alpha = alpha / (math.e * math.log(num_batches))
    else:
        batched_alpha = alpha
    return batched_alpha


def datasets_from_raw_input(raw_input, is_coo):

    if isinstance(raw_input, tf.data.Dataset):
        return [raw_input]

    if sp.issparse(raw_input):
        return [create_tensorrec_dataset_from_sparse_matrix(raw_input, is_coo)]

    if isinstance(raw_input, six.string_types):
        return [create_tensorrec_dataset_from_tfrecord(raw_input)]

    if isinstance(raw_input, list):

        if all([isinstance(input_val, tf.data.Dataset) for input_val in raw_input]):
            return raw_input

        if all([sp.issparse(input_val) for input_val in raw_input]):
            return [create_tensorrec_dataset_from_sparse_matrix(input_sparse_matrix, is_coo)
                    for input_sparse_matrix in raw_input]

        if all([isinstance(input_val, six.string_types) for input_val in raw_input]):
            return [create_tensorrec_dataset_from_tfrecord(input_str) for input_str in raw_input]

    raise ValueError('Input must be a scipy sparse matrix, an iterable of scipy sprase matrices, or a TensorFlow '
                     'Dataset')


def generate_dummy_data(num_users=15000, num_items=30000, interaction_density=.00045, num_user_features=200,
                        num_item_features=200, n_features_per_user=20, n_features_per_item=20,  pos_int_ratio=.5,
                        return_datasets=False):

    if pos_int_ratio <= 0.0:
        raise Exception("pos_int_ratio must be > 0")

    print("Generating positive interactions")
    interactions = sp.rand(num_users, num_items, density=interaction_density * pos_int_ratio)
    if pos_int_ratio < 1.0:
        print("Generating negative interactions")
        interactions += -1 * sp.rand(num_users, num_items, density=interaction_density * (1 - pos_int_ratio))

    print("Generating user features")
    user_features = sp.rand(num_users, num_user_features, density=float(n_features_per_user) / num_user_features)

    print("Generating item features")
    item_features = sp.rand(num_items, num_item_features, density=float(n_features_per_item) / num_item_features)

    if return_datasets:
        interactions = create_tensorrec_dataset_from_sparse_matrix(interactions, False)
        user_features = create_tensorrec_dataset_from_sparse_matrix(user_features, True)
        item_features = create_tensorrec_dataset_from_sparse_matrix(item_features, True)

    return interactions, user_features, item_features


def generate_dummy_data_with_indicator(num_users=15000, num_items=30000, interaction_density=.00045, pos_int_ratio=.5):

    n_user_features = int(num_users * 1.2)
    n_user_tags = num_users * 3
    n_item_features = int(num_items * 1.2)
    n_item_tags = num_items * 3
    n_interactions = (num_users * num_items) * interaction_density

    user_features = sp.lil_matrix((num_users, n_user_features))
    for i in range(num_users):
        user_features[i, i] = 1

    for i in range(n_user_tags):
        user_features[random.randrange(num_users), random.randrange(num_users, n_user_features)] = 1

    item_features = sp.lil_matrix((num_items, n_item_features))
    for i in range(num_items):
        item_features[i, i] = 1

    for i in range(n_item_tags):
        item_features[random.randrange(num_items), random.randrange(num_items, n_item_features)] = 1

    interactions = sp.lil_matrix((num_users, num_items))
    for i in range(int(n_interactions * pos_int_ratio)):
        interactions[random.randrange(num_users), random.randrange(num_items)] = 1

    for i in range(int(n_interactions * (1 - pos_int_ratio))):
        interactions[random.randrange(num_users), random.randrange(num_items)] = -1

    return interactions, user_features, item_features


def append_to_string_at_point(string, value, point):
    for _ in range(0, (point - len(string))):
        string += " "
    string += "{}".format(value)
    return string


def simple_tf_print(tensor, places=100):
    return tf.Print(tensor, [tensor, tf.shape(tensor)], summarize=places)


class lazyval(object):
    """
    Decorator that marks that an attribute of an instance should not be
    computed until needed, and that the value should be memoized.

    Lifted from quantopian/trading_calendars
    """
    def __init__(self, get):
        self._get = get
        self._cache = WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            return self._cache[instance]
        except KeyError:
            self._cache[instance] = val = self._get(instance)
            return val

    def __set__(self, instance, value):
        raise AttributeError("Can't set read-only attribute.")

    def __delitem__(self, instance):
        del self._cache[instance]


def variable_summaries(var, cosine=None, N=100):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('vars_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('values_histogram', var/tf.reshape(tf.norm(var, axis=1), shape=(-1, 1)))
        if cosine:
            # n = tf.shape(var)[0]
            # idx1 = tf.random_uniform(shape=(N,),maxval=n, dtype=tf.int32)
            # idx2 = tf.random_uniform(shape=(N,),maxval=n, dtype=tf.int32)
            idx1 = tf.range(0,100,dtype=tf.int64)
            idx2 = idx1
            v1 = tf.nn.embedding_lookup(var,idx1)
            v2 = tf.nn.embedding_lookup(var,idx2)
            ret = cosine(v1,v2)
            tf.summary.histogram('similarity', ret)
            tf.summary.image('similarity', tf.reshape(ret, shape=(1,tf.shape(ret)[0], tf.shape(ret)[1], 1)))


def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss