from unittest import TestCase

from tensorrec import TensorRec
from tensorrec.loss_graphs import (
    WMRBLossGraph
)
from tensorrec.util import generate_dummy_data_with_indicator
import pickle
from scipy.sparse import coo_matrix

class LossGraphsTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        # cls.interactions, cls.user_features, cls.item_features = generate_dummy_data_with_indicator(
        #     num_users=10, num_items=12, interaction_density=.5)
        cls.interactions, cls.user_features, cls.item_features = pickle.load(open('/Users/jasonchen/tmp/experiment.data'))
        n_users = cls.interactions.shape[0]
        cls.user_features = coo_matrix(([1]*n_users, (range(n_users), [0]*n_users)), shape=(n_users,1))
        print cls.interactions.shape, cls.user_features.shape, cls.item_features.shape


    def test_wmrb_loss(self):
        model = TensorRec(loss_graph=WMRBLossGraph(), stratified_sample=True, logdir='/Users/jasonchen/tmp/test')
        model.fit(self.interactions.tocsr()[:10], self.user_features.tocsr()[:10], self.item_features, epochs=5, verbose=True)

    def test_wmrb_loss_biased(self):
        model = TensorRec(loss_graph=WMRBLossGraph(), biased=True)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5, n_sampled_items=10)
