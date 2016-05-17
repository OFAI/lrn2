#!/usr/bin/env python
# -*- coding: utf-8 -*-

from subprocess import call
import sys

if sys.argv[-1] == "install":
    call(["apt-get","install","python-setuptools"])
from setuptools import setup
import tarfile
import os
from shutil import copy2, rmtree

name = 'lrn2'
version = '2.1.0'

if sys.argv[-1] == "install":
    install = ["apt-get", "install"]
    call(install + ["python-dev"])
    call(install + ["gFortran"])
    call(install + ["libblas-dev"])
    call(install + ["liblapack-dev"])
    call(install + ["python-pip"])
    call(install + ["libpng-dev"])
    call(install + ["libfreetype6-dev"])
    call(["pip", "install", "numpy"])

setup(name = name,
      version = version,
      description = 'Lrn2 Python package for learning representations from data',
      url = 'http://lrn2cre8.ofai.at/lrn2/doc',
      author = 'Stefan Lattner, Maarten Grachten, Carlos Eduardo Cancino Chacón',
      author_email = 'stefan.lattner@ofai.at, maarten.grachten@ofai.at, carlos.cancino@ofai.at',
      packages = ['lrn2',
                  'lrn2.application',
                  'lrn2.application.classification',
                  'lrn2.application.segmentation',
                  'lrn2.application.similarity',
                  'lrn2.application.visualization',
                  'lrn2.data',
                  'lrn2.data.domain',
                  'lrn2.data.formats',
		  'lrn2.data.formats.midi_utils',
                  'lrn2.data.formats.midi_utils.midi_backend',
                  'lrn2.nn_bricks',
                  'lrn2.util'],
      install_requires = ['jinja2',
                          'scikit-learn',
                          'matplotlib',
		          'configobj',
		          'scipy',
                          'theano',
                          'liac-arff',
                          'mpld3'],
      data_files = [('lrn2/util/', ['lrn2/util/config_spec.ini'])]
      )

if sys.argv[-1] == "sdist":

    # copy examples and bin into tar.gz without copying them during installation

    print "Adding demo folders..."

    pkg_name = name + '-' + version
    pkg_file = pkg_name + '.tar.gz'

    files_to_add = [os.path.join("examples", "mnist_pretrain", "config_model.ini"),
                    os.path.join("examples", "mnist_pretrain", "run_demo.py"),
                    os.path.join("examples", "mnist_convolutional", "config_model.ini"),
                    os.path.join("examples", "mnist_convolutional", "run_demo.py"),
		    os.path.join("examples", "custom_layer", "config_model.ini"),
                    os.path.join("examples", "custom_layer", "run_demo.py"),
		    os.path.join("examples", "rnn_predict", "config_model.ini"),
                    os.path.join("examples", "rnn_predict", "run_demo.py"),
                    os.path.join("lrn2", "util", "config_spec.ini"),
		    "setup_mac.py",
                    ]

    # Extract existing tar (python cannot append on compressed tar files)
    tar = tarfile.open(os.path.join("dist", pkg_file), "r:gz")
    tar.extractall()
    tar.close()

    # create additional folders and copy additional files
    for f in files_to_add:
        new_folder = os.path.join(pkg_name, os.path.split(f)[0])
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        copy2(f, os.path.join(pkg_name, f))

    # pack whole folder structure in tar file
    with tarfile.open(os.path.join("dist", pkg_file), "w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(pkg_name):
            for f in filenames:
                tar.add(os.path.join(dirpath, f))

    rmtree(pkg_name)

    print "...done."
