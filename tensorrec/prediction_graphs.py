import abc
import tensorflow as tf

from .recommendation_graphs import relative_cosine


class AbstractPredictionGraph(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def connect_dense_prediction_graph(self, tf_user_representation, tf_item_representation):
        """
        This method is responsible for consuming user and item representations and calculating prediction scores for all
        possible user-item pairs based on these representations.
        :param tf_user_representation: tf.Tensor
        The user representations as a Tensor of shape [n_users, n_components]
        :param tf_item_representation: tf.Tensor
        The item representations as a Tensor of shape [n_items, n_components]
        :return: tf.Tensor
        The predictions as a Tensor of shape [n_users, n_items]
        """
        pass


class DotProductPredictionGraph(AbstractPredictionGraph):
    """
    This prediction function calculates the prediction as the dot product between the user and item representations.
    Prediction = user_repr * item_repr
    """

    def connect_dense_prediction_graph(self, tf_user_representation, tf_item_representation):
        return tf.matmul(tf_user_representation, tf_item_representation, transpose_b=True, name='predict_score')


class CosineSimilarityPredictionGraph(AbstractPredictionGraph):
    """
    This prediction function calculates the prediction as the cosine between the user and item representations.
    Prediction = cos(user_repr, item_repr)
    """

    def connect_dense_prediction_graph(self, tf_user_representation, tf_item_representation):
        return relative_cosine(tf_tensor_1=tf_user_representation, tf_tensor_2=tf_item_representation)
