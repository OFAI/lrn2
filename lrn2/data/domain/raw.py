 #!/usr/bin/env python


import scipy.sparse as sp
import numpy as np
import logging

from lrn2.data.domain.viewpoint import ViewPoint

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class RawVP(ViewPoint):
    """ Viewpoint which returns the original raw data. """
    def __init__(self, size, sparse = False):
        super(RawVP, self).__init__()
        self.size_ = size
        self.sparse = sparse

    @property
    def size(self):
        return self.size_

    def raw_to_repr(self, raw_data, label):
        data = np.array(raw_data)
        if self.sparse:
            return sp.csr_matrix(data, shape = data.shape)
        else:
            return data

    def repr_to_visual(self, binary_data):
        return binary_data