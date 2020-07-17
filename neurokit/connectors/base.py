import numpy as np

class Connector:
    def __init__(self, pre, post, w_mu, w_sigma, w_mu_inh, w_sigma_inh, d, is_exc_func):
        self._pre = pre
        self._post = post
        self._w_mu = w_mu
        self._w_sigma = w_sigma
        self._w_mu_inh = w_mu_inh
        self._w_sigma_inh = w_sigma_inh
        self._d = d
        self._is_exc_func = is_exc_func

    def connect(self):
        raise Exception('not implemented')

    def _get_weight(self, src_idx):
        if self._is_exc_func(src_idx):
            return np.random.normal(self._w_mu, self._w_sigma)
        else:
            return np.random.normal(self._w_mu_inh, self._w_sigma_inh)