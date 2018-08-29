import abc
import tensorflow as tf
import numpy as np


class AbstractLossGraph(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def connect_loss_graph(self, tr, prediction_graph_factory, tf_user_representation, pos_item_representation, neg_items_representation, margin):
        pass


class WMRBLossGraph(AbstractLossGraph):
    """
    Approximation of http://ceur-ws.org/Vol-1905/recsys2017_poster3.pdf
    Interactions can be any positive values, but magnitude is ignored. Negative interactions are ignored.
    """
    is_sample_based = True

    def connect_loss_graph(self, tr, prediction_graph_factory, tf_user_representation, pos_item_representation, neg_items_representation, margin):

        def get_loss():
            with tf.name_scope('pos_score'):
                pos_prediction_score = prediction_graph_factory.connect_dense_prediction_graph(
                    tf_user_representation=tf_user_representation, tf_item_representation=pos_item_representation)
                tr.graph_nodes['pos_prediction_score'] = pos_prediction_score
            # with tf.name_scope('neg_repr'):
            #     tf_n_items = tf.shape(tf_item_representation)[0]
            #     neg_repr = tf.gather(tf_item_representation, tf.random_uniform(shape=(negSearchLimit,), minval=0, maxval=tf.cast(tf_n_items, dtype=tf.int64), dtype=tf.int64), name='collect_neg_sample')
            with tf.name_scope('neg_score'):
                neg_prediction_score = prediction_graph_factory.connect_dense_prediction_graph(
                    tf_user_representation=tf_user_representation, tf_item_representation=neg_items_representation)
                tr.graph_nodes['neg_prediction_score'] = neg_prediction_score
            with tf.name_scope('margin_score'):
                margin_score = tf.reshape(neg_prediction_score - pos_prediction_score + margin, shape=tf.shape(neg_prediction_score))
                tf_positive_mask = tf.greater(margin_score, 0.0)
                tr.graph_nodes['valid_neg_num'] = tf.reduce_sum(tf.cast(tf_positive_mask, dtype=tf.int32))
                loss_all = tf.boolean_mask(margin_score, tf_positive_mask)
                tr.graph_nodes['margin_score'] = margin_score
            return tf.reduce_sum(loss_all)

        return get_loss()
