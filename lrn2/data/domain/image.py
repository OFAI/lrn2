 #!/usr/bin/env python

import abc
import logging
import numpy as np
import scipy.sparse as sp

from lrn2.data.domain.viewpoint import ViewPoint

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class ImageBinaryVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, shape):
        super(ImageBinaryVP, self).__init__()
        self.shape_ = shape

    @abc.abstractmethod
    def get_image_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return self.shape_[0] * self.shape_[1]
    
    @property
    def shape(self):
        return (self.shape_[0], self.shape_[1])

    def raw_to_repr(self, raw_data, label):
        data = self.get_image_from_raw_data(raw_data)
        data = np.reshape(data, (1, len(data)))
        return sp.csr_matrix(data, shape = data.shape)

    def repr_to_visual(self, binary_data):
        data_2d = np.reshape(binary_data, self.shape, order="F")
        return data_2d
    