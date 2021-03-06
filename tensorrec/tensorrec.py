import logging
import numpy as np
import os,sys
import pickle
import time
import tensorflow as tf
import threading, thread
import multiprocessing
import itertools

from .errors import (
    TfVersionException
)
from .input_utils import create_tensorrec_iterator, get_dimensions_from_tensorrec_dataset
from .loss_graphs import AbstractLossGraph, WMRBLossGraph
from .prediction_graphs import AbstractPredictionGraph, CosineSimilarityPredictionGraph
from .representation_graphs import AbstractRepresentationGraph, LinearRepresentationGraph
from .session_management import get_session
from .util import (variable_summaries, get_memory, multiple_workers, multiple_workers_thread, stop_subprocess, get_queue_size)


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
                 log_interval=100,
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
        self.log_interval = log_interval
        self.logdir = logdir
        self.graph_nodes = {}
        self.memory_var = None
        self.active_train_thread = None
        self.neg_sample_limit = 100
        self.truncate_interval=1000
        self.coord = None
        self.input_step_size = 1000
        self.sub_process = []
        self.queue_input = None
        self.queue_output = None
        self.queue_input_cap = 100
        self.queue_output_cap = 100
        self.gen_ex_n_worker = 1
        self.tf_enqueue_n_worker = 1
        self.tf_queue_cap = 1000

    def _build_tf_graph(self, n_user_features, n_item_features, tf_learning_rate=0.01, tf_alpha=0.00001, margin=0.2, use_reg=False):
        # Build placeholders
        self.tf_learning_rate = tf.constant(tf_learning_rate, name='lr_rate') #tf.placeholder(tf.float32, None, name='learning_rate')
        self.tf_alpha = tf.constant(tf_alpha, name='alpha') #tf.placeholder(tf.float32, None, name='alpha')
        self.margin = tf.constant(margin, name='margin') #tf.placeholder(tf.float32, None, name='margin')
        # self.negSearchLimit = tf.placeholder(tf.int32, None, name='negSearchLimit')

        with tf.name_scope('feed_user_feature'):
            self.tf_user_feature_cols = tf.placeholder(tf.int32, [None], name='tf_user_feature_cols')
        with tf.name_scope('feed_interaction'):
            self.tf_interaction_words_user = tf.placeholder(tf.int32, [None], name='tf_interaction_words_user')
            self.tf_interaction_words_pos = tf.placeholder(tf.int32, [None], name='tf_interaction_words_pos')
        with tf.name_scope('neg_example'):
            self.tf_interaction_words_neg = tf.placeholder(tf.int32, [None], name='tf_interaction_words_neg')
            self.tf_interaction_words_neg_len = tf.placeholder(tf.int32, [None], name='tf_interaction_words_neg_len')

        # self.graph_nodes['test'] = tf.reduce_sum(self.tf_user_feature_cols) + tf.reduce_sum(self.tf_interaction_cols)

        q = tf.PaddingFIFOQueue(capacity=self.tf_queue_cap, dtypes=[tf.int32, tf.int32, tf.int32, tf.int32, tf.int32],
                                shapes=[[None], [None], [None], [None], [self.neg_sample_limit]], name='padding_queue')
        self.queue = q
        enqueue_op = q.enqueue([self.tf_user_feature_cols, self.tf_interaction_words_user,
                                self.tf_interaction_words_pos, self.tf_interaction_words_neg,
                                self.tf_interaction_words_neg_len], name='enqueue')
        self.graph_nodes['enqueue'] = enqueue_op

        input_batch = q.dequeue(name='dequeue')  # It replaces our input placeholder
        user_features, words_user, words_pos, words_neg, words_neg_len = input_batch
        self.graph_nodes['dequeue'] = input_batch
        self.graph_nodes['q_size'] = q.size(name='q_size')

        # Collect the vars for computing regulation
        # item feature variable
        _, item_weights = \
            self.item_repr_graph_factory.connect_representation_graph(feature_weights=[self.graph_nodes.get('item_weights')],
                                                                      tf_features=[0],
                                                                      n_components=self.n_components,
                                                                      n_features=n_item_features,
                                                                      node_name_ending='item',
                                                                      lookup=True)
        #reg_vars.extend(item_weights)
        self.graph_nodes['item_weights'] = item_weights[0]
        self.tf_word_representation = item_weights[0]
        print 'tf_word_representation', self.tf_word_representation
        print 'item_weights', item_weights[0]

        # user feature variable
        user_weights = self.graph_nodes.get('user_weights')
        tf_user_representation_feature, user_weights = self.user_repr_graph_factory.connect_representation_graph(
            feature_weights=[user_weights],
            tf_features=user_features, n_components=self.n_components, n_features=n_user_features,
            node_name_ending='user', lookup=True)
        self.graph_nodes['tf_user_representation_feature'] = tf_user_representation_feature
        self.graph_nodes['user_weights'] = user_weights[0]
        print 'tf_user_representation_feature', tf_user_representation_feature
        print 'user_weights', user_weights[0]

        with tf.name_scope('pos_example'):
            pos_item_representation = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(self.tf_word_representation, words_pos, name='pos_repr'), axis=0), shape=(1,-1))
            self.graph_nodes['pos_item_representation'] = pos_item_representation
            print 'pos_item_representation', pos_item_representation

        # user representation
        # user interation history representation
        with tf.name_scope('final_user_repr'):
            user_interaction_items_repr = tf.reduce_sum(tf.nn.embedding_lookup(self.tf_word_representation,
                                                                                words_user,
                                                                                name='lookup_interaction'), axis=0)
            self.tf_user_representation = tf_user_representation_feature + user_interaction_items_repr
            print 'user_interaction_items_repr', user_interaction_items_repr
            print 'self.tf_user_representation', self.tf_user_representation
            self.graph_nodes['tf_user_representation'] = self.tf_user_representation
            self.graph_nodes['tf_user_representation_feature'] = tf_user_representation_feature

        # negative examples
        with tf.name_scope('neg_examples'):
            neg_items = tf.split(words_neg, words_neg_len, name='neg_split')
            neg_items_representation = tf.stack([tf.reduce_sum(tf.nn.embedding_lookup(self.tf_word_representation,words,name='lookup_neg'),
                                            axis=0) for words in neg_items])
            print 'neg_items_representation', neg_items_representation
            self.graph_nodes['neg_item_representation'] = neg_items_representation

        with tf.name_scope('words_cnt'):
            self.graph_nodes['words_cnt'] = tf.shape(words_user)[0] + tf.shape(words_pos)[0] + tf.shape(words_neg)[0]

        # Compose loss function args
        # This composition is for execution safety: it prevents loss functions that are incorrectly configured from
        # having visibility of certain nodes.
        loss_graph_kwargs = {
            'prediction_graph_factory': self.prediction_graph_factory,
            'tf_user_representation': self.tf_user_representation,
            'neg_items_representation': neg_items_representation,
            'pos_item_representation': pos_item_representation,
            'margin': self.margin,
            'tr': self
        }

        # Build loss graph
        with tf.name_scope('basic_loss'):
            self.tf_basic_loss = self.loss_graph_factory.connect_loss_graph(**loss_graph_kwargs) / self.neg_sample_limit
        with tf.name_scope('reg_loss'):
            # reg_vars = [self.tf_user_representation, neg_items_representation, pos_item_representation]
            reg_vars = [self.graph_nodes['item_weights'], self.graph_nodes['user_weights']]
            self.tf_weight_reg_loss = self.tf_alpha * sum(tf.nn.l2_loss(weights) for weights in reg_vars)
            self.graph_nodes['tf_weight_reg_loss'] = self.tf_weight_reg_loss
        with tf.name_scope('loss'):
            if use_reg:
                print "using l2_reg of weights for part of loss"
                self.tf_loss = self.tf_basic_loss + self.tf_weight_reg_loss
            else:
                self.tf_loss = self.tf_basic_loss
        with tf.name_scope('optimizer'):
            self.tf_optimizer = tf.train.AdamOptimizer(learning_rate=self.tf_learning_rate).minimize(self.tf_loss)

        def trunc_norm(var, name_ending='var'):
            with tf.name_scope('truc_'+name_ending):
                norm = tf.norm(var, axis=1)
                norm_truc = tf.maximum(norm, 1.0)
                assign = tf.assign(var, var/tf.reshape(norm_truc, shape=(-1,1)))
                return tf.reduce_sum(assign)

        # do truncate norm like Starspace
        with tf.name_scope('trunc_embeds'):
            self.graph_nodes['truncat'] = tf.add(trunc_norm(self.graph_nodes['item_weights'], 'item_weights'),
                                                trunc_norm(self.graph_nodes['user_weights'], 'user_weights'),
                                                name='truncate_weights')

    def fit(self, interactions, user_features, item_features, epochs=100, learning_rate=0.1, alpha=0.00001,
            verbose=False, margin=0.2, negSearchLimit=100, train_threads=None, use_reg=False):
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
        start_time = time.time()
        session = get_session()
        self.neg_sample_limit = negSearchLimit
        self.epochs = epochs

        if verbose:
            logging.info('Processing interaction and feature data')

        # Check input dimensions
        n_users, n_user_features = user_features.shape
        n_items, n_item_features = item_features.shape
        print 'n_users, n_user_features, n_items, n_item_features', n_users, n_user_features, n_items, n_item_features
        self.n_users = n_users

        user_features = user_features.tocsr()
        interactions = interactions.tocsr()
        item_features = item_features.tocsr()
        item_dict = {i: item_features.indices[item_features.indptr[i]:item_features.indptr[i + 1]] for i in
                     range(item_features.shape[0])}

        # Check if the graph has been constructed by checking the dense prediction node
        # If it hasn't been constructed, initialize it
        if self.memory_var is None:
            print 'building tensorflow graph'
            self.memory_var = tf.Variable(get_memory() / 1000000000, name='memory', trainable=False)

            # Numbers of features are either learned at fit time from the shape of these two matrices or specified at
            # TensorRec construction and cannot be changed.
            self._build_tf_graph(n_user_features=n_user_features, n_item_features=n_item_features, use_reg=use_reg,
                                 margin=margin, tf_learning_rate=learning_rate, tf_alpha=alpha)
            print 'end building tensorflow graph'

            queue_input_size = tf.py_func(lambda : get_queue_size(self.queue_input), inp=[], Tout=tf.int64)
            queue_output_size = tf.py_func(lambda : get_queue_size(self.queue_output), inp=[], Tout=tf.int64)

            session.run(tf.global_variables_initializer(), )

        # Build the shared feed dict
        # feed_dict = {self.tf_learning_rate: learning_rate,
        #              self.tf_alpha: calculate_batched_alpha(num_batches=n_users, alpha=alpha),
        #              self.margin: margin}

        with tf.name_scope('log_item_weights'):
            variable_summaries(self.graph_nodes['item_weights'], self.prediction_graph_factory.connect_dense_prediction_graph)
        with tf.name_scope('log_user_weights'):
            variable_summaries(self.graph_nodes['user_weights'])
        with tf.name_scope('log_training'):
            tf.summary.scalar('basic_loss', tf.reduce_mean(self.tf_basic_loss))
            tf.summary.scalar('tf_weight_reg_loss', self.graph_nodes['tf_weight_reg_loss'])
            tf.summary.scalar('loss', self.tf_loss)
            tf.summary.scalar('memory', self.memory_var)
            tf.summary.scalar('valid_neg_num', self.graph_nodes['valid_neg_num'])
            tf.summary.scalar('words_cnt', self.graph_nodes['words_cnt'])
            tf.summary.scalar('tf_queue_size', self.graph_nodes['q_size'])
            tf.summary.scalar('preprocess_input_queue_size', queue_input_size)
            tf.summary.scalar('preprocess_output_queue_size', queue_output_size)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.logdir +'/'+time.strftime('%x-%X').replace('/','').replace(':',''),
                                        session.graph,
                                        max_queue=10, flush_secs=30)

        coord = tf.train.Coordinator()
        self.coord = coord
        threads = []
        self.queue_input, self.queue_output, self.sub_process = multiple_workers_thread(
            lambda tid, q_in, q_out: self.get_train_example(tid, q_in, q_out,
                                                user_features, interactions, item_dict, neg_sample_limit=100),
            queue_capacity_input=self.queue_input_cap, queue_capacity_output=self.queue_output_cap, n_worker=self.gen_ex_n_worker)

        gen_thread = threading.Thread(target=self.gen_user_id, args=(self.queue_input,), name='generate_user_id_thread')
        enqueue_threads = [threading.Thread(target=self.enqueue_func, args=(self.queue_output,), name='enqueue tensorflow') for i in range(3)]
        # self.active_train_thread += 1
        for t in enqueue_threads: t.start()
        gen_thread.start()

        def train_worker(coord, session, tid):
            step = 0
            print '*** start running train_worker %s' % tid
            while not coord.should_stop():
                step += 1
                if not verbose or step % self.log_interval or tid != 0:
                    session.run(self.tf_optimizer)
                else:
                    _, _, loss, summary,q_size = session.run(
                        [self.memory_var.assign(get_memory() / 10**9), self.tf_optimizer, self.tf_basic_loss,
                         merged, self.graph_nodes['q_size']]
                    )
                    mean_loss = np.mean(loss)
                    train_writer.add_summary(summary, step)
                    out_str = '\ttrain: step = {} loss = {}, q_size = {}'.format( step, mean_loss, q_size)
                    print out_str

            self.active_train_thread -= 1
            print '--- stop train_worker thread %s ---' % tid

        cpu_num = train_threads if train_threads else multiprocessing.cpu_count() - 1
        print 'train thread number %s' % cpu_num
        train_threads = [threading.Thread(target=train_worker, args=(coord,session,i)) for i in xrange(cpu_num)]
        threads.extend(train_threads)
        self.active_train_thread = cpu_num  # training threads
        for t in threads:
            t.start()

        for t in threads:
            t.join()
        gen_thread.join()
        for t in enqueue_threads: t.join()
        stop_subprocess(self.sub_process)
        for p in self.sub_process:
            p.join()
        print '\n--- end of training---'
        print 'used time %s' % (time.time() - start_time)
        return 0

    def gen_user_id(self, queue_input):
        print '* start running gen_user_id, tid=%s' % thread.get_ident()
        cnt = 0
        session = get_session()
        for epoch in range(self.epochs*100000):
            users = range(self.n_users)
            np.random.shuffle(users)
            for u in users:
                cnt += 1
                if cnt % self.input_step_size == 0:
                    queue_size = session.run(self.graph_nodes['q_size'])
                    sys.stdout.write('\tgen_user_id: epoch %3s, cnt %8s, queue_size %3s \n' % (epoch, cnt, queue_size))
                if cnt % self.truncate_interval == 0:
                    norm_sum, item_weights = session.run([self.graph_nodes['truncat'], self.graph_nodes['item_weights']])
                    print '\tafter truncatem norm_sum %s, min %s, max %s' % (norm_sum, np.min(item_weights), np.max(item_weights))
                if self.active_train_thread is not None and self.active_train_thread == 0:
                    print '- end gen_user_id'
                    return
                if epoch >= self.epochs:
                    self.coord.request_stop()
                    u = -1  # subprocess will stop for u < 0
                # print 'gen uid %s' % u
                try:
                    queue_input.put(u, timeout=10)
                except Exception:
                    print 'enqueue uid timeout'

        print '- end gen_user_id 2'

    def enqueue_func(self, queue_output):
        print '* start enqueue_func: feed tensorflow queue'
        session = get_session()
        cnt = 0
        while self.active_train_thread != 0:  # if this is not the only thread running
            try:
                ex = queue_output.get(timeout=5)
                cnt += 1
                if cnt % 1000 == 0:
                    print '\tenqueue_func %s, waiting %s' % (cnt, get_queue_size(queue_output))
                # Set a 5-second timeout.
                run_options = tf.RunOptions(timeout_in_ms=5000)
                session.run(self.graph_nodes['enqueue'], options= run_options,
                            feed_dict={self.tf_user_feature_cols: ex[0],
                                        self.tf_interaction_words_user: ex[1],
                                        self.tf_interaction_words_pos: ex[2],
                                        self.tf_interaction_words_neg: ex[3],
                                        self.tf_interaction_words_neg_len: ex[4]})
            except Exception as err:
                print 'exception enqueue_func', err
        print '- end enqueue_func'

    def get_train_example(self, tid, queue_input, queue_output, user_features, interactions, item_dict, neg_sample_limit=100):
        item_num = len(item_dict)
        print '* start running get_train_example', tid

        def get_input(u):
            uf = user_features.indices[user_features.indptr[u]:user_features.indptr[u+1]]
            ir = interactions.indices[interactions.indptr[u]:interactions.indptr[u+1]]
            if len(ir)<2:
                print 'interaction is not enough'
                return None
            pos_idx = np.random.randint(len(ir))
            words_user = [x for x in itertools.chain.from_iterable([item_dict.get(item, []) for item in ir[pos_idx+1:].tolist()+ir[:pos_idx].tolist()])]
            words_pos = item_dict.get(ir[pos_idx], [])
            neg_items = filter(lambda l:len(l) > 0, [item_dict.get(x, []) for x in np.random.randint(item_num, size=neg_sample_limit+5)])[:neg_sample_limit]
            if len(neg_items) != neg_sample_limit or len(words_user) == 0 or len(words_pos)==0 or len(uf) == 0:
                # print "\t\tneg_item/words_user/words_pos has not enough words (%s,%s,%s,pos_idx=%s)" % (len(neg_items),len(words_user), len(words_pos), pos_idx)
                return None
            words_neg = [x for x in itertools.chain.from_iterable(neg_items)]
            words_neg_len = [len(l) for l in neg_items]
            return uf, words_user, words_pos, words_neg, words_neg_len

        while self.active_train_thread != 0:
            try:
                u = queue_input.get(timeout=5)
                if u<0:
                    break
                # print 'tid %s, u %s' %(tid, u)
                r = get_input(u)
                if r:
                    # print 'tid %s, put example u %s' % (tid, u)
                    queue_output.put(r,timeout=5)
            except Exception as err:
                print 'get_train_example exception', err
        print '- end get_train_input'

    def save_model(self, directory_path):
        """
        Saves the model to files in the given directory.
        :param directory_path: str
        The path to the directory in which to save the model.
        :return:
        """
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
