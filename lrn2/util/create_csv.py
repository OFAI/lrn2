#!/usr/bin/env python

'''
Created on Sep 2, 2014

@author: Stefan Lattner
'''

import csv
import os
import logging
import glob

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def create_csv(input_folder, out_file, label_fun=None, value_fun=None,
                      rec=False, pat="*", incl_dirs = False):
    """
    Creates a csv file listing all files of a folder (subfolders optional).

    The first column of the csv file is controlled by the value function.
    The second column is controlled by the label function.

    Parameters
    ----------

    input_folder : string
        path of the root folder containing the files of interest

    out_file : string
        (path + ) filename of the resulting csv file

    label_fun : function
        is called f("path+filename") and has to return a string

    value_fun : function
        is called f("path+filename") and has to return a string

    rec : boolean (optional)
        if True, subfolders are also recursively visited

    pat : string (optional)
        filename pattern (e.g. *.mid). Only matching files will be considered

    """

    if not pat:
        pat = "*"

    if label_fun is None:
        label_fun = lambda fn : fn

    if value_fun is None:
        value_fun = lambda fn : fn

    with open(out_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                                quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        for root, _, _ in os.walk(input_folder):
            for fn in glob.glob1(root, pat):
                f = os.path.join(root, fn)
                if not os.path.isdir(f) or incl_dirs:
                    writer.writerow([value_fun(f), label_fun(f)])
            if not rec:
                break