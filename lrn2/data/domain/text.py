#!/usr/bin/env python

import logging
import numpy as np

from lrn2.data.domain.viewpoint import ViewPoint
from lrn2.nn_bricks.utils import fx
import re
from lrn2.util.utils import shape

logging.basicConfig()
LOGGER = logging.getLogger(__name__)

class PartOfSpeechVP(ViewPoint):
    def __init__(self, dict_tags):
        self.dict_tags = dict_tags
        self.create_dict_map(dict_tags)
        self.create_one_hot_combinations(dict_tags)

    @property
    def size(self):
        return len(self.tag_dict.keys())

    def create_one_hot_combinations(self, dict_tags):
        re.compile("")
        one_hot = []
        for line in dict_tags:
            tags = line.split("\t")[0].split("+")
            idxs = [self.tag_dict[tag]-1 for tag in tags]
            curr_one_hot = np.zeros(self.size)
            curr_one_hot[idxs] = 1
            one_hot.append(curr_one_hot)

        self.one_hot = one_hot


    def create_dict_map(self, dict_tags):
        tag_groups = []
        for line in dict_tags:
            tag_groups.append(line.split("\t")[0])

        tags = []
        for group in tag_groups:
            tags.extend(group.split("+"))

        tag_dict = {}
        for tag in tags:
            try:
                _ = tag_dict[tag]
            except KeyError:
                tag_dict[tag] = len(tag_dict.keys())

        self.tag_dict = tag_dict

    def raw_to_repr(self, raw_data, label):
        rex = re.compile("([0-9]*):")
        words = rex.findall(raw_data)
#         print words[:10]
        one_hot = [self.one_hot[int(word)-1] for word in words]
#         print one_hot[:10]
        return np.vstack(one_hot)

class PartOfSpeechTargetsVP(ViewPoint):
    def __init__(self, dict_tags):
        self.dict_tags = dict_tags
#         self.create_dict_map(dict_tags)
#         self.create_one_hot_combinations(dict_tags)

    @property
    def size(self):
        return 1

    def raw_to_repr(self, raw_data, label):
        lines = raw_data.split("\n")
        result = []
        for line in lines:
            if len(line) > 0:
                label = 1 - (int(line[0]) - 1)
                word_count = len(line.split(" ")) -1
                result.extend([label] * word_count)

        
        return np.reshape(result, (-1,1))
    
class PartOfSpeechMarkerVP(ViewPoint):
    def __init__(self, dict_tags):
        self.dict_tags = dict_tags
#         self.create_dict_map(dict_tags)
#         self.create_one_hot_combinations(dict_tags)

    @property
    def size(self):
        return 1

    def raw_to_repr(self, raw_data, label):
        lines = raw_data.split("\n")
        result = []
        for line in lines:
            if len(line) > 0:
                word_count = len(line.split(" ")) -1
                result.extend([1] + [0] * (word_count - 1))
        
        return np.reshape(result, (-1,1))



class CharacterVP(ViewPoint):
    """
    Converts character sequences into one-hot representations

    Parameters
    ----------

    sample : string, optional
        a sample string used to create a char catalog
        all characters which should be represented have to be included in
        this string

    char_cat : dict, optional
        a dictionary with char -> integer pairs

    """
    def __init__(self, sample = None, char_cat = None,
                 suppress_warnings = False):
        super(CharacterVP, self).__init__()
        assert sample is not None or char_cat is not None, \
                                            "Both parameters cannot be 'None'."

        self.suppress_warnings = suppress_warnings

        if char_cat is None:
            self.char_cat = {}
        else:
            self.char_cat = char_cat

        if sample is not None:
            self.catalog_chars(sample)

    def catalog_chars(self, example):
        chars = [c for c in example]
        char_cat = dict(zip(chars, chars))
        for i, c in enumerate(char_cat.keys()):
            char_cat[c] = i

        LOGGER.debug("There are {0} chars in the catalog.".format(len(char_cat.keys())))
        self.char_cat.update(char_cat)

    @property
    def size(self):
        return len(self.char_cat.keys())

    def raw_to_repr(self, raw_data, label):
        char_cat = self.char_cat
        mapping = []
        for c in raw_data:
            try:
                mapping.extend([char_cat[c]])
            except KeyError:
                if not self.suppress_warnings:
                    LOGGER.warning("No catalog entry found for char '{0}'. ".format(c) +
                                   "If you want to represent this char, include it "
                                   "in the sample string at initialization.")

        result = np.zeros((len(mapping), self.size), dtype=fx)
        result[range(len(mapping)), mapping] = 1
        return result

#     def repr_to_visual(self, binary_data):
#         return np.rot90(binary_data, k=1)


if __name__ == '__main__':
    c = CharacterVP(sample="abcde fg HIJ klmn 234")
    print c.raw_to_repr("abcde K 2364", None)

