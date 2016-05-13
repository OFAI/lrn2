'''
Created on Sep 5, 2014

@author: Stefan Lattner
'''

from matplotlib import cm
import logging
import matplotlib
try:
    matplotlib.use('agg')
except:
    pass
import matplotlib.pyplot as plt
import numpy as np

import mpld3

from lrn2.util.utils import uniquifier, to_numpy

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def plot_fs_2d(data_set, labels=None, tooltip=None, size=(10,10), save_fn=None,
               interactive=False, cmap=None, show_centroids=True,
               label_centroids=True, style_data='o', style_centroid='o'):
    """
    Plot a given dataset and utilizes labels.
    Plots a given 2d dataset and uses different colors for datapoints belonging
    to different labels.
    Displays a centroid for each label cluster and names it with the label string.

    Parameters
    ----------

    data_set: array-like
        2D matrix of size M x N which is to be points

    labels: array-like, optional
        1D array of size M, preferrable of dtype string

    tooltip: array-like, optional
        1D array of size M, used as tooltip annotation for points

    size: tuple, optional
        The size of the plot

    save_fn: string, optional
        If not None, a file will be saved to that path

    cmap: matplotlib.cm, optional
        Colormap which will be used for plotting datapoints of each label in
        a different color

    show_centroids: boolean, optional
        Determines if centroids should be points

    label_centroids: boolean, optional
        Determines if centroids should be labelled

    Returns
    -------

    figure
        The figure. Use figure.save()

    """

    def color_iterator(cmap, n):
        return iter(cmap(np.linspace(0,1,n)))

    assert len(data_set[0]) == 2  # Input matrix has to be 2d.

    plt.interactive(True)

    if cmap == None:
        #cmap = cm.get_cmap("gist_ncar")
        cmap = cm.get_cmap("Paired")

    data_set, labels = to_numpy(data_set, labels)

    fig = plt.figure(figsize=size)
    ax  = fig.gca()
    points = None

    if labels is not None:
        labels_single = uniquifier(labels)
        colors = color_iterator(cmap, len(labels_single))

        for label in labels_single:
            col = next(colors)
            data_xy = (data_set[labels == label,0], data_set[labels == label,1])
            points = ax.plot(data_xy[0], data_xy[1], style_data, color=col, markersize = 2)

        colors = color_iterator(cmap, len(labels_single))
        if show_centroids:
            for label in labels_single:
                col = next(colors)
                mean_xy = (np.mean(data_set[labels==label,0]),
                           np.mean(data_set[labels==label,1]))
                ax.plot(mean_xy[0], mean_xy[1], style_centroid, color=col)
                bbox_props = dict(boxstyle="round,pad=0.3", fc=col, ec="grey",
                                  lw=1, alpha=0.9)
                if label_centroids:
                    plt.annotate(label, xy = mean_xy, xytext = (-15, 7),
                                 textcoords = 'offset points', bbox=bbox_props)
    else:
        col = cmap(0)
        points = ax.plot(data_set[:,0], data_set[:,1], style_data, color=col)

    if save_fn is not None:
        LOGGER.debug('Saving 2D feature space projection to {0}'.format(save_fn))
        plt.savefig(save_fn)

    if interactive:
        desc = [unicode(l, 'utf-8') for l in tooltip]
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], desc)
        mpld3.plugins.connect(fig, tooltip)
        mpld3.show()

    # while not plt.waitforbuttonpress(timeout=-1):
    #     pass

    #plt.interactive(False)

    return fig

if __name__ == '__main__':
    pass
