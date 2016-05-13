'''
Created on Nov 6, 2014

This module provides methods for extracting segment boundaries from .krn files

@author: Stefan Lattner
'''
import os
import glob

import numpy as np
from lrn2.util.utils import read_input_filelist
from collections import OrderedDict

def read_krn_segs(input_folder, rec=False, pat=None):
    if not pat:
        pat = "*"

    segments = []
    for root, _, _ in os.walk(input_folder):
        for fn in glob.glob1(root, pat):
            f = os.path.join(root, fn)
            with open(f, 'r') as curr_file:
                content = curr_file.read()
                notes = [line[:1] for line in content.split('\n') if is_note(line[:1])]
                segments.append(np.where(np.array(notes) == '{')[0].tolist())

        if not rec:
            break

    return np.array(segments)

def read_krn_segs_csv(input_csv, sort = False):
    files = read_input_filelist(input_csv)
    if sort:
        files = sorted(files, key = lambda x : x[1])
    segments = OrderedDict()
    for f, _ in files:
        with open(f, 'r') as curr_file:
            content = curr_file.read()
            notes = [line[:1] for line in content.split('\n') if is_note(line)]
            fn = f.split('/')[-1].split('.')[0]
            segments[fn] = np.where(np.array(notes) == '{')[0]

    return segments

def is_note(s):
    if 'r' in s:
        # For now, a rest is not a note
        return False

    if s[:1] == '{':
        return True

    try:
        int(s[:1])
    except ValueError:
        return False
    else:
        return True

if __name__ == '__main__':
    read_krn_segs("/mnt/data/data/lrn2cre8_data_unpacked/one")
