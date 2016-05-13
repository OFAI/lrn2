#!/usr/bin/env python

'''
Created on Sep 2, 2014

@author: Stefan Lattner
'''

import logging
import argparse

from lrn2.util.create_csv import create_csv
from lrn2.data.formats.midi import midi_key_signature

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def create_musical_key_labels(input_folder, out_file, rec, pat):
    """
    Creates a csv file listing all MIDI files of a folder (subfolders optional).

    The first column of the csv file contains the respective path + filenames
    The second column of the csv file contains the musical keys (if that
    information is stored in the MIDI files) - 'None' otherwise

    Parameters
    ----------

    output : string
        (path + ) filename of the resulting csv file

    source_folder : string
        path of the root folder containing the files of interest

    rec : boolean (optional)
        if True, subfolders are also recursively visited

    pat : string (optional)
        filename pattern (e.g. *.mid). Only matching files will be considered

    """
    label_f = lambda fn : midi_key_signature(fn)
    create_csv(input_folder=input_folder,
                      out_file=out_file,
                      label_fun=label_f,
                      rec=rec,
                      pat=pat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Create a csv file containing"
                                     " a list of MIDI files and corresponding key"
                                     "signature labels (if stored within files)."
                                     " The resulting csv file can be used as input"
                                     " to the demo applications.")

    parser.add_argument("source_folder", help = ("folder containing source files"))

    parser.add_argument("output", help = "filename of the resulting csv file")

    parser.add_argument("-r", action = "store_true", default = False,
                       help = "recursively follow subfolders")

    parser.add_argument("-p", help = "specify a filename pattern (enclosed by quotes)")

    args = parser.parse_args()

    create_musical_key_labels(args.source_folder, args.output, args.r, args.p);