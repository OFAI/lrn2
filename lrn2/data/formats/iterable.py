'''
Created on Oct 10, 2014

@author: Stefan Lattner
'''

def load_iter_items(iterable):
    # Loading function for any iterable
    for i, item in enumerate(iterable):
        yield (item, i)

def load_single(data):
    # Loading function for any iterable
    return [(data, "data")]