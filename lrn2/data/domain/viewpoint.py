'''
Created on Sep 11, 2014

@author: Maarten Grachten
'''

class ViewPoint(object):
    def repr_to_visual(self, binary_data):
        return binary_data
    
    @property
    def size(self):
        raise NotImplementedError("Please define a size() method in Viewpoint.")
    
