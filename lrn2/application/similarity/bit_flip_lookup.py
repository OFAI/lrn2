'''
Created on Sep 18, 2014

@author: Stefan Lattner
'''

import logging
import numpy as np

from _collections import defaultdict
from lrn2.util.utils import ensure_list
from lrn2.nn_bricks.utils import project_to_feature_space

LOGGER = logging.getLogger(__name__)

class BitFlipLookup(object):
    '''
    classdocs
    '''
    def __init__(self, model, data, keywords, raw_converter=None):
        '''
        Constructor
        '''
        self.model = model
        self.dict = defaultdict(list)
        self.raw_converter = raw_converter
        self.init_dict(data, keywords)

    def init_dict(self, data, keywords):
        codes = project_to_feature_space(self.model, data)
        codes = np.around(codes).astype(bool)

        for i in range(len(codes)):
            self.dict[tuple(codes[i])].append(keywords[i])

        LOGGER.info("Dictionary created. Mapped {0} instances onto {1} distinct binary codes.".format(len(codes), len(self.dict)))

        for values in self.dict.itervalues():
            print len(values)


    def find_similar(self, code, n):
        LOGGER.debug("Starting similarity lookup...")
        code = ensure_list(code)
        result = []
        max_dist = 2
        dist = 0
        while len(result) < n and dist <= max_dist:
            LOGGER.debug("Searching with hamming distance {}...".format(dist))
            bit_flip_comb = self.xuniqueCombinations(range(len(code)), dist)
            close_codes = [[not code[i] if i in flip_bits else code[i] for i in range(len(code))] for flip_bits in bit_flip_comb]
            i = 0
            while len(result) < n and i < len(close_codes):
                result.extend(self.dict[tuple(close_codes[i])])
                i += 1
            dist += 1

        return result

    def neighbors(self, query, k = 1):
        return np.argsort(np.sum(np.logical_xor(self.dict.keys(), query).astype(np.int), 1))[:k]

    def lookup_by_keyword(self, keyword, n_min):
        for key, value in self.dict.iteritems():
            if keyword in value:
                return self.find_similar(key, n_min)

    def lookup_by_instance(self, instance, n_min):
        code = np.around(project_to_feature_space(self.model, instance)).astype(bool)
        return self.find_similar(code.flatten(), n_min)

    def lookup_by_raw_data(self, data, n_min):
        return self.lookup_by_instance(self.raw_converter(data), n_min)

    def xuniqueCombinations(self, items, n):
        if n==0: yield []
        else:
            for i in xrange(len(items)):
                for cc in self.xuniqueCombinations(items[i+1:],n-1):
                    yield [items[i]]+cc
