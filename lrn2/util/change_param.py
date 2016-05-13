# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:03:33 2014

@author: Stefan Lattner
"""

import sys
from lrn2.util.config import get_config

usage = ("\nUsage:\nchange_param.py config_file command (e.g. config[\\'SECTION\\'][\\'param\\']=10)\n")

def change_param(config_file, command):

    config = get_config(config_file)
    exec command
    config.write()

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        config_file = str(sys.argv[1].strip())
        command = str(sys.argv[2].strip())

        change_param(config_file, command)
    else:
        print(usage)