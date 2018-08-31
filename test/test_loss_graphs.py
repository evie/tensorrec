from unittest import TestCase

from tensorrec import TensorRec
from tensorrec.loss_graphs import (
    WMRBLossGraph
)
from tensorrec.util import generate_dummy_data_with_indicator
import pickle
from scipy.sparse import coo_matrix
import numpy as np

class LossGraphsTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        # cls.interactions, cls.user_features, cls.item_features = generate_dummy_data_with_indicator(
        #     num_users=10, num_items=12, interaction_density=.5)
        cls.interactions, cls.user_features, cls.item_features,_,_,_ = pickle.load(open('/Users/jasonchen/tmp/test_data0801.data'))
        n_users = cls.interactions.shape[0]
        cls.user_features = coo_matrix(([1]*n_users, (range(n_users), [0]*n_users)), shape=(n_users,1))
        cls.n_test_item = cls.interactions.shape[1] #111
        rows=[];cols=[]; data=[]
        for i in range(len(cls.interactions.data)):
            if cls.interactions.col[i] < cls.n_test_item:
                cols.append(cls.interactions.col[i])
            else:
                cols.append(np.random.choice(cls.n_test_item, 1))
            rows.append(cls.interactions.row[i])
            data.append(1)
        cls.interactions = coo_matrix((data, (rows, cols)), shape=(cls.interactions.shape[0], cls.n_test_item))

        print cls.interactions.shape, cls.user_features.shape, cls.item_features.shape

    def test_wmrb_loss(self):
        model = TensorRec(loss_graph=WMRBLossGraph(), stratified_sample=True, logdir='/Users/jasonchen/tmp/test', log_interval=100)
        model.fit(self.interactions.tocsr(), self.user_features.tocsr(), self.item_features.tocsr()[:self.n_test_item], epochs=10, verbose=True, train_threads=5, use_reg=True)

    def test_wmrb_loss_biased(self):
        model = TensorRec(loss_graph=WMRBLossGraph(), biased=True)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)
