import re
import logging
from collections import OrderedDict

LOGGER = logging.getLogger(__name__)

class TXTInterface(object):
    def get_text_from_raw_data(self, data):
        return data

def load_txt_files(filenames, prepro_fun = lambda x : x):
    for n in filenames:
        with open(n, 'r') as f:
            content = prepro_fun(f.read())
            yield (content, n)

def idx_char(files, char = ' ', assume_del = False, prepro_fun = lambda x : x):
    """
    Finds the idx of all chars 'char' in a list of files

    Parameters
    ----------

    char : string
        A char (or string) whose start indices should be found

    assume_del : boolean
        Adjusts the indices as if the chars in question were already deleted

    Returns
    -------

    Dictionary: key == filename, value == list of char indices
    """

    n_chars = len(char)
    segments = OrderedDict()
    for f in files:
        with open(f, 'r') as curr_file:
            content = prepro_fun(curr_file.read())
            segments[f] = [c.start() - i*n_chars
                           for i, c in enumerate(re.finditer(char, content))]

    return segments