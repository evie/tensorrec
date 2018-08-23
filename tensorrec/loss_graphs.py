import abc
import tensorflow as tf
import numpy as np


class AbstractLossGraph(object):
    __metaclass__ = abc.ABCMeta

    # If True, dense prediction results will be passed to the loss function
    is_dense = False

    # If True, randomly sampled predictions will be passed to the loss function
    is_sample_based = False
    # If True, and if is_sample_based is True, predictions will be sampled with replacement
    is_sampled_with_replacement = False

    @abc.abstractmethod
    def connect_loss_graph(self, tr, prediction_graph_factory, tf_user_representation, pos_item_representation,
                           tf_item_representation, tf_interaction_cols, maxNegSamples, negSearchLimit, margin):
        """
        This method is responsible for consuming a number of possible nodes from the graph and calculating loss from
        those nodes.

        The following parameters are always passed in
        :param tf_prediction_serial: tf.Tensor
        The recommendation scores as a Tensor of shape [n_samples, 1]
        :param tf_interactions_serial: tf.Tensor
        The sample interactions corresponding to tf_prediction_serial as a Tensor of shape [n_samples, 1]
        :param tf_interactions: tf.SparseTensor
        The sample interactions as a SparseTensor of shape [n_users, n_items]
        :param tf_n_users: tf.placeholder
        The number of users in tf_interactions
        :param tf_n_items: tf.placeholder
        The number of items in tf_interactions

        The following parameters are passed in if is_dense is True
        :param tf_prediction: tf.Tensor
        The recommendation scores as a Tensor of shape [n_users, n_items]
        :param tf_rankings: tf.Tensor
        The item ranks as a Tensor of shape [n_users, n_items]

        The following parameters are passed in if is_sample_based is True
        :param tf_sample_predictions: tf.Tensor
        The recommendation scores of a sample of items of shape [n_users, n_sampled_items]
        :param tf_n_sampled_items: tf.placeholder
        The number of items per user in tf_sample_predictions

        :return: tf.Tensor
        The loss value.
        """
        pass


class WMRBLossGraph(AbstractLossGraph):
    """
    Approximation of http://ceur-ws.org/Vol-1905/recsys2017_poster3.pdf
    Interactions can be any positive values, but magnitude is ignored. Negative interactions are ignored.
    """
    is_sample_based = True

    def connect_loss_graph(self, tr, prediction_graph_factory, tf_user_representation, pos_item_representation, tf_item_representation,
                           tf_interaction_cols, maxNegSamples, negSearchLimit, margin):

        def get_loss():
            with tf.name_scope('pos_score'):
                pos_prediction_score = prediction_graph_factory.connect_dense_prediction_graph(
                    tf_user_representation=tf_user_representation, tf_item_representation=pos_item_representation)
                tr.graph_nodes['pos_prediction_score'] = pos_prediction_score
            with tf.name_scope('neg_repr'):
                tf_n_items = tf.shape(tf_item_representation)[0]
                neg_repr = tf.gather(tf_item_representation, tf.random_uniform(shape=(negSearchLimit,), minval=0, maxval=tf.cast(tf_n_items, dtype=tf.int64), dtype=tf.int64), name='collect_neg_sample')
            with tf.name_scope('neg_score'):
                neg_prediction_score = prediction_graph_factory.connect_dense_prediction_graph(
                    tf_user_representation=tf_user_representation, tf_item_representation=neg_repr)
                tr.graph_nodes['neg_prediction_score'] = neg_prediction_score
            with tf.name_scope('margin_score'):
                margin_score = tf.reshape(neg_prediction_score - pos_prediction_score + margin, shape=tf.shape(neg_prediction_score))
                tf_positive_mask = tf.greater(margin_score, 0.0)
                loss_all = tf.boolean_mask(margin_score, tf_positive_mask)
                tr.graph_nodes['margin_score'] = margin_score
            return tf.reduce_sum(loss_all)

        return tf.cond(tf.equal(tf.size(tf_item_representation), 0), lambda : tf.constant(0., tf.float32), get_loss, name='cond_empty')
