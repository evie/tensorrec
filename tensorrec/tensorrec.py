from functools import partial
from itertools import cycle
import logging
import numpy as np
import os
import pickle
from scipy import sparse as sp
import tensorflow as tf

from .errors import (
    ModelNotBiasedException, ModelNotFitException, ModelWithoutAttentionException, BatchNonSparseInputException,
    TfVersionException
)
from .input_utils import create_tensorrec_iterator, get_dimensions_from_tensorrec_dataset
from .loss_graphs import AbstractLossGraph, WMRBLossGraph
from .prediction_graphs import AbstractPredictionGraph, CosineSimilarityPredictionGraph
from .recommendation_graphs import (
    project_biases, split_sparse_tensor_indices, bias_prediction_dense, bias_prediction_serial, rank_predictions,
    densify_sampled_item_predictions, collapse_mixture_of_tastes, predict_similar_items
)
from .representation_graphs import AbstractRepresentationGraph, LinearRepresentationGraph
from .session_management import get_session
from .util import (sample_items, calculate_batched_alpha, datasets_from_raw_input, sample_items_stratified,
                   variable_summaries, get_memory)


class TensorRec(object):

    def __init__(self,
                 n_components=100,
                 n_tastes=1,
                 user_repr_graph=LinearRepresentationGraph(),
                 item_repr_graph=LinearRepresentationGraph(),
                 attention_graph=None,
                 prediction_graph=CosineSimilarityPredictionGraph(),
                 loss_graph=WMRBLossGraph(),
                 biased=False,
                 stratified_sample=False,
                 logdir='.',):
        """
        A TensorRec recommendation model.
        :param n_components: Integer
        The dimension of a single output of the representation function. Must be >= 1.
        :param n_tastes: Integer
        The number of tastes/reprs to be calculated for each user. Must be >= 1.
        :param user_repr_graph: AbstractRepresentationGraph
        An object which inherits AbstractRepresentationGraph that contains a method to calculate user representations.
        See tensorrec.representation_graphs for examples.
        :param item_repr_graph: AbstractRepresentationGraph
        An object which inherits AbstractRepresentationGraph that contains a method to calculate item representations.
        See tensorrec.representation_graphs for examples.
        :param attention_graph: AbstractRepresentationGraph or None
        Optional. An object which inherits AbstractRepresentationGraph that contains a method to calculate user
        attention. Any valid repr_graph is also a valid attention graph. If None, no attention process will be applied.
        :param prediction_graph: AbstractPredictionGraph
        An object which inherits AbstractPredictionGraph that contains a method to calculate predictions from a pair of
        user/item reprs.
        See tensorrec.prediction_graphs for examples.
        :param loss_graph: AbstractLossGraph
        An object which inherits AbstractLossGraph that contains a method to calculate the loss function.
        See tensorrec.loss_graphs for examples.
        :param biased: bool
        If True, a bias value will be calculated for every user feature and item feature.
        """

        # Check TensorFlow version
        major, minor, patch = tf.__version__.split(".")
        if int(major) < 1 or int(major) == 1 and int(minor) < 7:
            raise TfVersionException(tf_version=tf.__version__)

        # Arg Check
        if (n_components is None) or (n_tastes is None) or (user_repr_graph is None) or (item_repr_graph is None) \
                or (prediction_graph is None) or (loss_graph is None):
            raise ValueError("All arguments to TensorRec() must be non-None")
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if n_tastes < 1:
            raise ValueError("n_tastes must be >= 1")
        if not isinstance(user_repr_graph, AbstractRepresentationGraph):
            raise ValueError("user_repr_graph must inherit AbstractRepresentationGraph")
        if not isinstance(item_repr_graph, AbstractRepresentationGraph):
            raise ValueError("item_repr_graph must inherit AbstractRepresentationGraph")
        if not isinstance(prediction_graph, AbstractPredictionGraph):
            raise ValueError("prediction_graph must inherit AbstractPredictionGraph")
        if not isinstance(loss_graph, AbstractLossGraph):
            raise ValueError("loss_graph must inherit AbstractLossGraph")
        if attention_graph is not None:
            if not isinstance(attention_graph, AbstractRepresentationGraph):
                raise ValueError("attention_graph must be None or inherit AbstractRepresentationGraph")
            if n_tastes == 1:
                raise ValueError("attention_graph must be None if n_tastes == 1")

        self.n_components = n_components
        self.n_tastes = n_tastes
        self.user_repr_graph_factory = user_repr_graph
        self.item_repr_graph_factory = item_repr_graph
        self.attention_graph_factory = attention_graph
        self.prediction_graph_factory = prediction_graph
        self.loss_graph_factory = loss_graph
        self.biased = biased
        self.stratified_sample = stratified_sample
        self.logdir = logdir
        self.graph_nodes = {}
        self.memory_var = None

    def _build_tf_graph(self, n_user_features, n_item_features, item_features):

        # Build placeholders
        self.tf_learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
        self.tf_alpha = tf.placeholder(tf.float32, None, name='alpha')
        self.margin = tf.placeholder(tf.float32, None, name='margin')
        # self.negSearchLimit = tf.placeholder(tf.int64, None, name='negSearchLimit')

        with tf.name_scope('feed_user_feature'):
            self.tf_user_feature_cols = tf.placeholder(tf.int64, [None], name='tf_user_feature_cols')
        with tf.name_scope('feed_interaction'):
            self.tf_interaction_cols = tf.placeholder(tf.int64, [None], name='tf_interaction_cols')

        # Collect the weights for regularization

        tf_weights = []
        # Build all items representations
        with tf.name_scope('init_item_sparse_tensor'):
            item_features = item_features.tocoo()
            tf_item_feature_indices = tf.stack([tf.constant(item_features.row, dtype=tf.int64),
                                                tf.constant(item_features.col, dtype=tf.int64)], axis=1)
            tf_item_features = tf.SparseTensor(tf_item_feature_indices, tf.constant(item_features.data,
                                                            dtype=tf.float32), item_features.shape)

        # item and user variables
        self.tf_item_representation, item_weights = \
            self.item_repr_graph_factory.connect_representation_graph(tf_features=tf_item_features,
                                                                      n_components=self.n_components,
                                                                      n_features=n_item_features,
                                                                      node_name_ending='item')
        tf_weights.extend(item_weights)
        self.graph_nodes['item_weights'] = item_weights

        tf_user_representation_feature, user_weights = self.user_repr_graph_factory.connect_representation_graph(
            tf_features=self.tf_user_feature_cols, n_components=self.n_components, n_features=n_user_features,
            node_name_ending='user', lookup=True)
        tf_weights.extend(user_weights)
        self.graph_nodes['user_weights'] = user_weights

        with tf.name_scope('get_pos_item'):
            def get_positive_item(tf_interaction_cols):
                indices = tf.random_shuffle(tf_interaction_cols)
                pos_cols = indices[0]
                user_cols = indices[1:]
                return pos_cols, user_cols

            pos_cols, user_cols = get_positive_item(self.tf_interaction_cols)
            self.graph_nodes['pos_cols'] = pos_cols
            self.graph_nodes['user_cols'] = user_cols
        with tf.name_scope('pos_example'):
            pos_item_representation = tf.reshape(tf.gather(self.tf_item_representation, pos_cols, name='get_positive_item'), shape=(1,-1))

        # user representation
        # user interation history representation

        with tf.name_scope('user_repr'):
            user_interaction_items_repr = tf.reduce_sum(tf.nn.embedding_lookup(self.tf_item_representation,
                                                                                user_cols,
                                                                                name='lookup_interaction'), axis=0)

            self.tf_user_representation = tf_user_representation_feature + user_interaction_items_repr

        self.graph_nodes['tf_user_representation'] = self.tf_user_representation
        self.graph_nodes['pos_item_representation'] = pos_item_representation
        self.graph_nodes['tf_user_representation_feature'] = tf_user_representation_feature


        # Compose loss function args
        # This composition is for execution safety: it prevents loss functions that are incorrectly configured from
        # having visibility of certain nodes.
        with tf.name_scope('get_loss'):
            loss_graph_kwargs = {
                'prediction_graph_factory': self.prediction_graph_factory,
                'tf_user_representation': self.tf_user_representation,
                'tf_item_representation': self.tf_item_representation,
                'pos_item_representation': pos_item_representation,
                'tf_interaction_cols': self.tf_interaction_cols,
                'maxNegSamples': 5, # not used
                'negSearchLimit': 100,
                'margin': self.margin,
                'tr': self
            }

            # Build loss graph
            self.tf_basic_loss = self.loss_graph_factory.connect_loss_graph(**loss_graph_kwargs)

            self.tf_weight_reg_loss = sum(tf.nn.l2_loss(weights) for weights in tf_weights)
            self.tf_loss = self.tf_basic_loss + (self.tf_alpha * self.tf_weight_reg_loss)

        self.tf_optimizer = tf.train.AdamOptimizer(learning_rate=self.tf_learning_rate).minimize(self.tf_loss)
        # Record the new node names

    def fit(self, interactions, user_features, item_features, epochs=100, learning_rate=0.1, alpha=0.00001,
            verbose=False, margin=0.2):
        """
        Constructs the TensorRec graph and fits the model.
        :param interactions: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of interactions of shape [n_users, n_items].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param user_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of user features of shape [n_users, n_user_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param item_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of item features of shape [n_items, n_item_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param epochs: Integer
        The number of epochs to fit the model.
        :param learning_rate: Float
        The learning rate of the model.
        :param alpha:
        The weight regularization loss coefficient.
        :param verbose: boolean
        If true, the model will print a number of status statements during fitting.
        :param user_batch_size: int or None
        The maximum number of users per batch, or None for all users.
        :param n_sampled_items: int or None
        The number of items to sample per user for use in loss functions. Must be non-None if
        self.loss_graph_factory.is_sample_based is True.
        """

        # Pass-through to fit_partial
        self.fit_partial(interactions=interactions,
                         user_features=user_features,
                         item_features=item_features,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         alpha=alpha,
                         verbose=verbose, margin=margin)

    def fit_partial(self, interactions, user_features, item_features, epochs=1, learning_rate=0.1,
                    alpha=0.00001, verbose=False, margin=0.2, negSearchLimit=100):
        """
        Constructs the TensorRec graph and fits the model.
        :param interactions: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of interactions of shape [n_users, n_items].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param user_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of user features of shape [n_users, n_user_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param item_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of item features of shape [n_items, n_item_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param epochs: Integer
        The number of epochs to fit the model.
        :param learning_rate: Float
        The learning rate of the model.
        :param alpha:
        The weight regularization loss coefficient.
        :param verbose: boolean
        If true, the model will print a number of status statements during fitting.
        :param user_batch_size: int or None
        The maximum number of users per batch, or None for all users.
        :param n_sampled_items: int or None
        The number of items to sample per user for use in loss functions. Must be non-None if
        self.loss_graph_factory.is_sample_based is True.
        """

        session = get_session()

        if verbose:
            logging.info('Processing interaction and feature data')

        # Check input dimensions
        n_users, n_user_features = user_features.shape
        n_items, n_item_features = item_features.shape

        # Check if the graph has been constructed by checking the dense prediction node
        # If it hasn't been constructed, initialize it
        if self.memory_var is None:
            self.memory_var = tf.Variable(get_memory() / 1000000000, name='memory', trainable=False)

            # Numbers of features are either learned at fit time from the shape of these two matrices or specified at
            # TensorRec construction and cannot be changed.
            self._build_tf_graph(n_user_features=n_user_features, n_item_features=n_item_features, item_features=item_features)
            session.run(tf.global_variables_initializer())

        # Build the shared feed dict
        feed_dict = {self.tf_learning_rate: learning_rate,
                     self.tf_alpha: calculate_batched_alpha(num_batches=n_users, alpha=alpha),
                     self.margin: margin}

        if verbose:
            logging.info('Beginning fitting')

        with tf.name_scope('log_item_weights'):
            variable_summaries(self.graph_nodes['item_weights'])
        with tf.name_scope('log_training'):
            tf.summary.scalar('weight_reg_l2_loss', alpha * self.tf_weight_reg_loss)
            tf.summary.scalar('mean_loss', tf.reduce_mean(self.tf_basic_loss))
            tf.summary.scalar('memory', self.memory_var)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.logdir + '/train', session.graph)

        user_features = user_features.tocsr()
        interactions = interactions.tocsr()
        for epoch in range(epochs):
            for u in xrange(interactions.shape[0]):
                feed_dict[self.tf_user_feature_cols] = user_features.indices[user_features.indptr[u]:user_features.indptr[u+1]]
                feed_dict[self.tf_interaction_cols] = interactions.indices[interactions.indptr[u]:interactions.indptr[u+1]]
                if not verbose:
                    session.run(self.tf_optimizer, feed_dict=feed_dict)

                else:
                    _, _, loss, wr_loss, summary = session.run(
                        [self.memory_var.assign(get_memory() / 1000000000), self.tf_optimizer, self.tf_basic_loss, self.tf_weight_reg_loss,
                         merged],
                        feed_dict=feed_dict
                    )
                    if u%1 == 0:
                        mean_loss = np.mean(loss)
                        weight_reg_l2_loss = alpha * wr_loss
                        train_writer.add_summary(summary, epoch*n_users + u)
                        out_str = 'EPOCH {} user_cnt {} loss = {}, weight_reg_l2_loss = {}'.format(
                            epoch, u, mean_loss, weight_reg_l2_loss)
                        logging.info(out_str)
                        print out_str



    def save_model(self, directory_path):
        """
        Saves the model to files in the given directory.
        :param directory_path: str
        The path to the directory in which to save the model.
        :return:
        """

        # Ensure that the model has been fit
        if False:
            raise ModelNotFitException(method='save_model')

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        saver = tf.train.Saver()
        session_path = os.path.join(directory_path, 'tensorrec_session.cpkt')
        saver.save(sess=get_session(), save_path=session_path)

        # Break connections to the graph before saving the python object
        tensorrec_path = os.path.join(directory_path, 'tensorrec.pkl')
        with open(tensorrec_path, 'wb') as file:
            pickle.dump(file=file, obj=self)


    @classmethod
    def load_model(cls, directory_path):
        """
        Loads the TensorRec model and TensorFlow session saved in the given directory.
        :param directory_path: str
        The path to the directory containing the saved model.
        :return:
        """

        graph_path = os.path.join(directory_path, 'tensorrec_session.cpkt.meta')
        saver = tf.train.import_meta_graph(graph_path)

        session_path = os.path.join(directory_path, 'tensorrec_session.cpkt')
        saver.restore(sess=get_session(), save_path=session_path)

        tensorrec_path = os.path.join(directory_path, 'tensorrec.pkl')
        with open(tensorrec_path, 'rb') as file:
            model = pickle.load(file=file)
        return model
