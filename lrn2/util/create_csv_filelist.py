#!/usr/bin/env python

'''
Created on Sep 2, 2014

@author: Stefan Lattner
'''

import os
import logging
import argparse
from lrn2.util.create_csv import create_csv

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def create_simple_file_list(input_folder, out_file, rec, pat, incl_dirs):
    """
    Creates a csv file listing all files of a folder (subfolders optional).

    The first column of the csv file contains the respective path + filenames
    The second column of the csv file contains only the filename

    Parameters
    ----------

    out_file : string
        (path + ) filename of the resulting csv file

    source_folder : string
        path of the root folder containing the files of interest

    rec : boolean (optional)
        if True, subfolders are also recursively visited

    pat : string (optional)
        filename pattern (e.g. *.txt). Only matching files will be considered

    """
    filename = lambda fn : os.path.split(fn)[-1]
    create_csv(input_folder=input_folder,
                      out_file=out_file,
                      label_fun=filename,
                      rec=rec,
                      pat=pat, incl_dirs=incl_dirs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Create a csv file containing a list of files. This file can be used as input to the demo applications.")

    parser.add_argument("source_folder", help = ("folder containing source files"))

    parser.add_argument("output", help = "filename of the resulting csv file")

    parser.add_argument("-r", action = "store_true", default = False,
                       help = "recursively follow subfolders")

    parser.add_argument("-p", help = "specify a filename pattern (enclosed by quotes)")
    
    parser.add_argument("--include-dirs", action = "store_true", default = False,
                       help = "include directories in csv file")

    args = parser.parse_args()

    create_simple_file_list(args.source_folder, args.output, args.r, args.p, args.include_dirs);