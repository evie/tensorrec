import abc
import tensorflow as tf


class AbstractRepresentationGraph(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def connect_representation_graph(self, tf_linear_weights, tf_features, n_components, n_features, node_name_ending):
        """
        This method is responsible for connecting the user/item features to their respective latent representations.
        :param tf_features: tf.SparseTensor
        The user/item features as a SparseTensor of shape [ n_users, n_features ]
        :param n_components: int
        The size of the latent representation per user/item.
        :param n_features: int
        The size of the input features per user/item.
        :param node_name_ending: str
        A string, either 'user' or 'item', which can be added to TensorFlow node names for clarity.
        :return: tf.Tensor
        The user/item representation as a Tensor of shape [ n_users, n_components ]
        """
        pass


class LinearRepresentationGraph(AbstractRepresentationGraph):
    """
    Calculates the representation by passing the features through a linear embedding.
    Rough approximation of http://ceur-ws.org/Vol-1448/paper4.pdf
    """

    def connect_representation_graph(self, feature_weights, tf_features, n_components, n_features, node_name_ending, lookup=False):

        # Weights are normalized before building the variable
        if feature_weights[0] is None:
            raw_weights = tf.random_normal([n_features, n_components], stddev=1.0, name='raw_weights')
            normalized_weights = tf.nn.l2_normalize(raw_weights, 1)
            # Create variable nodes
            # tf_linear_weights = tf.get_variable(name='weights_{}'.format(node_name_ending), initializer=normalized_weights)
            tf_linear_weights = tf.Variable(normalized_weights, name='weights_{}'.format(node_name_ending))
        else:
            tf_linear_weights = feature_weights[0]
        with tf.name_scope('repr_{}'.format(node_name_ending)):
            if lookup:
                tf_repr = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(tf_linear_weights, tf_features),axis=0), shape=(1,-1))
            else:
                tf_repr = tf.sparse_tensor_dense_matmul(tf_features, tf_linear_weights)

        # Return repr layer and variables
        return tf_repr, [tf_linear_weights]


class NormalizedLinearRepresentationGraph(LinearRepresentationGraph):
    """
    Calculates the representation by passing the features through a linear embedding. Embeddings are L2 normalized,
    meaning all embeddings have equal magnitude. This can be useful as a user representation in mixture-of-tastes
    models, preventing one taste from having a much larger magnitude than others and dominating the recommendations.
    """

    def connect_representation_graph(self, tf_features, n_components, n_features, node_name_ending):
        tf_repr, weights_list = super(NormalizedLinearRepresentationGraph, self).connect_representation_graph(
            tf_features=tf_features, n_components=n_components, n_features=n_features, node_name_ending=node_name_ending
        )
        normalized_repr = tf.nn.l2_normalize(tf_repr, 1)
        return normalized_repr, weights_list


class ReLURepresentationGraph(AbstractRepresentationGraph):
    """
    Calculates the representations by passing the features through a single-layer ReLU neural network.
    :param relu_size: int or None
    The number of nodes in the ReLU layer. If None, the layer will be of size 4*n_components.
    """

    def __init__(self, relu_size=None):
        self.relu_size = relu_size

    def connect_representation_graph(self, tf_features, n_components, n_features, node_name_ending):

        # Infer ReLU layer size if necessary
        if self.relu_size is None:
            relu_size = 4 * n_components
        else:
            relu_size = self.relu_size

        # Create variable nodes
        tf_relu_weights = tf.Variable(tf.random_normal([n_features, relu_size], stddev=.5),
                                      name='relu_weights_{}'.format(node_name_ending))
        tf_relu_biases = tf.Variable(tf.zeros([1, relu_size]),
                                     name='relu_biases_{}'.format(node_name_ending))
        tf_linear_weights = tf.Variable(tf.random_normal([relu_size, n_components], stddev=.5),
                                        name='linear_weights_{}'.format(node_name_ending))

        # Create ReLU layer
        tf_relu = tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(tf_features, tf_relu_weights),
                                    tf_relu_biases))
        tf_repr = tf.matmul(tf_relu, tf_linear_weights)

        # Return repr layer and variables
        return tf_repr, [tf_relu_weights, tf_linear_weights, tf_relu_biases]


class AbstractKerasRepresentationGraph(AbstractRepresentationGraph):
    """
    This abstract RepresentationGraph allows you to use Keras layers as a representation function by overriding the
    create_layers() method.
    """
    __metaclass__ = abc.ABCMeta

    def connect_representation_graph(self, tf_features, n_components, n_features, node_name_ending):
        layers = self.create_layers(n_features=n_features, n_components=n_components)

        weights = []
        last_layer = tf_features

        # Iterate through layers, connecting each one and extracting weights/biases
        for layer in layers:
            last_layer = layer(last_layer)
            if hasattr(layer, 'weights'):
                weights.extend(layer.weights)

        return last_layer, weights

    @abc.abstractmethod
    def create_layers(self, n_features, n_components):
        """
        Returns a list of Keras layers.
        :param n_features: int
        The input size of the first Keras layer.
        :param n_components: int
        The output size of the final Keras layer.
        :return: list
        A list of Keras layers.
        """
        pass
