'''
Created on Mar 5, 2015

@author: Stefan Lattner & Maarten Grachten
'''
import os
import logging
import numpy as np
import numpy.ma as ma

from _functools import partial
from lrn2.nn_bricks.notifier import Notifier

from matplotlib import use
use('agg')

import matplotlib.pyplot as plt
import PIL.Image
from lrn2.util.utils import shape


LOGGER = logging.getLogger(__name__)

class Plotter(object):
    """
    Mix-in class to enable a layer or stack to plot data
    """
    
    # Plotting types 
    # use default for weights and basic parameters, 
    # hist for histogram, curve for functions, colormap for heat-plots
    ptypes = ('default', 'hist', 'curve', 'colormap')
    tiling_modes = ('corpus', 'default')

    def register_plot(self, data_fun, label, name_net = "NN",
                      tiling = 'corpus', ptype = 'default',
                      forward = True):
        """
        Parameters
        ----------
        
        data_fun : function
            Function returning the data to plot
            
        label : string
            Label which will be used in the filename of the resulting plot
            
        name_net : string
            Label of the net (will be prefix in filename)
            
        tiling : string
            tiling_modes = ('corpus', 'default')
            
        ptype : string
            Plot type, one of ('default', 'hist', 'curve', 'colormap')
            use 'default' for weights and basic parameters, 
            'hist' for histogram, 'curve' for functions and 'colormap' for heat-plots
    
        forward : boolean
            forward this registration to containers
            (e.g. from a layer to a surrounding stack)
            
        """
        assert tiling in self.tiling_modes
        assert ptype in self.ptypes
        
        self.callback_add(partial(self.plot_fun, data_fun = data_fun,
                                  label = label, tiling = tiling,
                                  ptype = ptype, name_net = name_net),
                                  Notifier.PLOTTING, forward = forward)

    def plot_fun(self, data_fun, data_batch, out_dir, label, epoch_nr, name_net,
                 tile_fun_corpus, tiling = "corpus", ptype = 'default'):
        assert tiling in self.tiling_modes
        assert ptype in self.ptypes
        
        tile_fun = tile_fun_corpus if tiling == 'corpus' else dummy_tiler

        hist_prefix = '_hist' if ptype == 'hist' else ''
        fn = os.path.join(out_dir, name_net +
                          "_{0}{1}_{2}.png".format(label, hist_prefix, epoch_nr))
        LOGGER.debug("Plotting to {0}".format(fn))
        
        try:
            if ptype == 'default':
                make_tiles(tile_fun(data_fun(data_batch)), fn)
            elif ptype == 'hist':
                save_hist(data_fun(data_batch), '{0} {1}'.format(label, name_net), fn)
            elif ptype == 'curve':
                plot_curve(data_fun(data_batch), '{0} {1}'.format(label, name_net), fn)
            elif ptype == 'colormap':
                if self.convolutional:
                    plot_colormap(data_fun(data_batch), '{0} {1}'.format(label, name_net), fn)
                else:
                    plot_colormap(tile_fun(data_fun(data_batch)), '{0} {1}'.format(label, name_net), fn)
        except KeyboardInterrupt:
            LOGGER.info("Plot interrupted")
            pass
        except Exception as e:
            LOGGER.exception(e)

def plot_colormap(values, title, filename):
    '''
    Save a colormap of values titled with title in file filename.
    '''
    #LOGGER.debug("img_size self-sim: {0}".format(values.shape))
    plt.imshow(values)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def make_tiles(tiles_in_parts, out_file,
               vp_separator_color = (100, 150, 100), tile_separator_color = (25, 75, 25),
               unit = True):

    #LOGGER.debug("plotting to file {0}, min = {1}, max = {2}, mean = {3}".format(out_file, np.min(tiles_in_parts), np.max(tiles_in_parts), np.mean(tiles_in_parts)))
    tiles = [make_tile(tile_parts, vp_separator_color, unit) for tile_parts in tiles_in_parts]

    assert len(tiles) > 0
    tile_height, tile_width, _ = tiles[0].shape

#     LOGGER.debug('Tile size: {tile_width} px wide, {tile_height} px high'
#                  .format(tile_width = tile_width, tile_height = tile_height))


    n_tiles = len(tiles)
    n_tiles_horz = max(1, min(n_tiles, int(( (tile_height * n_tiles ) / tile_width ) **.5)))
    n_tiles_vert = int(np.ceil(n_tiles / float(n_tiles_horz)))

    pane = np.zeros( ((n_tiles_vert * (tile_height + 1) ) - 1 ,
                      (n_tiles_horz * (tile_width + 1) ) - 1,
                      3), dtype = np.uint8 )

    offset_horz = 0
    offset_vert = 0

    #LOGGER.debug('Creating a pane for {n_tiles} tiles, ({n_tiles_horz} wide x {n_tiles_vert} high)'
    #              .format(n_tiles = n_tiles, n_tiles_horz = n_tiles_horz, n_tiles_vert = n_tiles_vert))

    for i, tile in enumerate(tiles):
        if i % n_tiles_horz == 0 and n_tiles_vert > 1:
            pane[offset_vert + tile_height, :, : ] = tile_separator_color
            if i > 0:
                offset_horz = 0
                offset_vert += tile_height + 1

        pane[offset_vert : offset_vert + tile_height, offset_horz : offset_horz + tile_width, :] = tile
        offset_horz += tile_width + 1
        if offset_horz - 1 < pane.shape[1]:
            pane[:, offset_horz - 1, : ] = tile_separator_color

    filt_img = PIL.Image.fromarray(pane, mode = "RGB")
    filt_img.save(out_file)
    return pane

def make_tile(tile_parts, vp_separator_color = (255, 0, 0), unit = True):
    # This function could do a simple hstack, if it weren't for the
    # separation lines that we want between viewpoints. Furthermore,
    # we need to rescale the values of the viewpoints jointly, but do
    # not want the separation lines to interfere with the
    # rescaling. Therefore, we use a masked array.

    # NOTE: the tile is transposed after it is constructed. This means
    # that with respect to the plots displaying the tiles, the meaning
    # of width and height in this function are swapped

    # compute shape of tile, including separation lines
    shapes = np.array([x.shape for x in tile_parts], np.int)
    # assume the viewpoints have all the same size in the first
    # dimension (the ngram size)
    assert np.std(shapes[:,0]) == 0

    # tile_height equals the ngram size
    tile_height = shapes[0,0]
    # tile_width equals the sum of the viewpoint sizes plus (nr of
    # viewpoints) - 1 for the viewpoint seperation lines

    tile_width = np.sum(shapes[:,1]) + len(tile_parts) - 1

    # create an empty masked array the size of the tile (data and mask
    # will be set)
    tile = ma.masked_array(np.empty((tile_height, tile_width)))

    # copy the viewpoint data to their respective locations in the
    # tile, set a mask on the line after each viewpoint (except the
    # last)
    offset = 0
    for part in tile_parts:

        # assign viewpoint data
        tile[:, offset : offset+part.shape[1]] = part
        offset += part.shape[1] + 1

        # we are not at the last part
        if offset < tile_width:
            # set mask
            tile[:, offset - 1] = ma.masked

    # scale the values to 0-255, round, and convert to uint8
    tile = np.round(scale_to_unit_interval(tile) * 255).astype(np.uint8)

    # duplicate tile for RGB data
    tr = tile.reshape(list(tile.shape) + [1]).repeat(3, axis = 2)

    offset = 0
    # set the separation lines between the viewpoints (overrides
    # the mask)
    for part in tile_parts:
        offset += part.shape[1] + 1
        if offset < tile_width:
            tr[:, offset - 1, :] = vp_separator_color

    return tr.transpose((1,0,2))

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    max_val = np.max((np.abs(ndar).max(), 1e-4))
    ndar /= (2 * max_val + eps)
    ndar += 0.5
    return ndar

def scale_to_unit_interval_bw(ndar, eps=1e-8):
    """ Yields black & white plots """
    ndar = ndar.copy()
    ndar -= np.min(ndar)
    ndar /= np.max(ndar)
    return 1 - ndar

def save_hist(values, title, filename):
    '''
    Save a histogram of values titled with title in file filename.
    '''

    plt.hist(values.flatten(), bins=50)
    plt.xlabel('Value')
    plt.ylabel('Amount')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def plot_curve(curve, title, fn):
    plt.clf()
    plt.plot(curve)
    plt.title(title)
    plt.savefig(fn)
    plt.close()

def normalize(array):
    array -= np.min(array)
    array /= np.max(array)
    return array

def dummy_tiler(data):
    if len(data.shape) == 4:
        # Convolutional
        return [[np.transpose(x.reshape((x.shape[0],-1), order='F'))] for x in data]
    else:
        return [[x.reshape((-1,1))] for x in data]


# Old way to plot
#     def plot(self, data, out_dir, tile_fun, current_epoch, name = "NN"):
#         """
#         Plots different images which visualize the current state of a layer
#         during training.
#
#         Parameters
#         ----------
#
#         net : NN mix-in layer
#             A mix-in layer, as returned by make.make_layer
#
#         data : array-like
#             some data to the net (do not use all the training data, restrict to
#             e.g. the batch size)
#
#         out_dir : string
#             a path to an existing directory for placing the generated files
#
#         tile_fun : function
#             a tiling function, which takes a 2D array (standard) or 4D array (for
#             convolutional nets) and returns a 4D representation for plotting
#             (a 2D outlay for 2D tiles).
#
#         current_epoch : int
#             filenames of output files will show this number
#
#         name : string, optional
#             a name for filenames of output files
#
#         """
#         try:
#             fn_data = os.path.join(out_dir, name + "_data_{0}.png".format(current_epoch))
#             fn_pps = os.path.join(out_dir, name + "_pps_{0}.png".format(current_epoch))
#             fn_vbias = os.path.join(out_dir, name + "_vbias_{0}.png".format(current_epoch))
#             fn_hbias = os.path.join(out_dir, name + "_hbias_{0}.png".format(current_epoch))
#             fn_hact = os.path.join(out_dir, name + "_hact_{0}.png".format(current_epoch))
#             fn_weights = os.path.join(out_dir, name + "_weights_{0}.png".format(current_epoch))
#             fn_recon = os.path.join(out_dir, name + "_data_recon_{0}.png".format(current_epoch))
#             fn_hist_hact = os.path.join(out_dir, name + '_hist_hact_{0}.png'.format(current_epoch))
#             fn_hist_hbias = os.path.join(out_dir, name + '_hist_hbias_{0}.png'.format(current_epoch))
#             fn_hist_vbias = os.path.join(out_dir, name + '_hist_vbias_{0}.png'.format(current_epoch))
#             fn_hist_weights = os.path.join(out_dir, name + '_hist_weights_{0}.png'.format(current_epoch))
#             fn_hist_dhbias = os.path.join(out_dir, name + '_hist_dhbias_{0}.png'.format(current_epoch))
#             fn_hist_dvbias = os.path.join(out_dir, name + '_hist_dvbias_{0}.png'.format(current_epoch))
#             fn_hist_dweights = os.path.join(out_dir, name + '_hist_dweights_{0}.png'.format(current_epoch))
#             fn_curve_cost = os.path.join(out_dir, name + '_cost_curve_{0}.png'.format(current_epoch))
#
#             if "pps" in self.__dict__:
#                 make_tiles(tile_fun(self.pps.get_value()), fn_pps)
#
#             if "bv" in self.__dict__:
#                 tile_vbias = dummy_tiler if self.convolutional else tile_fun
#                 make_tiles(tile_vbias(np.reshape(self.bv.get_value(), (1,-1))),
#                            fn_vbias)
#
#
#             bh = self.bh.get_value()
#             make_tiles(dummy_tiler(np.reshape(normalize(bh), (1, bh.shape[0]))), fn_hbias)
#
#             tile_hact = tile_fun if self.convolutional else dummy_tiler
#
#             if hasattr(self, 'out'):
#                 make_tiles(tile_hact(self.out(data)), fn_hact)
#             #print "hact min/max/avg", np.min(self.out(data)), np.max(self.out(data)), np.mean(self.out(data))
#
#             make_tiles(tile_fun(data), fn_data)
#             #print "data min/max/avg", np.min(data), np.max(data), np.mean(data)
#             if hasattr(self, 'recon'):
#                 make_tiles(tile_fun(self.recon(data)), fn_recon)
#
#             make_tiles(tile_fun(self.W.get_value()), fn_weights)
#
#             save_hist(self.W.get_value(), 'Weights {0}'.format(name),
#                       fn_hist_weights)
#
#             save_hist(self.bh.get_value(), 'Hidden biases {0}'.format(name),
#                       fn_hist_hbias)
#
#             if hasattr(self, 'bv'):
#                 save_hist(self.bv.get_value(), 'Visible bias {0}'.format(name),
#                           fn_hist_vbias)
#
#             if isinstance(self, Monitor):
#                 try:
#                     if len(self.cost_curve) > 1:
#                         plot_curve(self.cost_curve, 'Cost curve {0}'.format(name),
#                              fn_curve_cost)
#                     dweights = self.W.get_value() - self.last_params[0]
#                     dhbias = self.bh.get_value() - self.last_params[1]
#                     dvbias = self.bv.get_value() - self.last_params[2]
#                     save_hist(dweights, "dWeights {0}".format(name),
#                               fn_hist_dweights)
#                     save_hist(dhbias, "dHidden biases {0}".format(name),
#                               fn_hist_dhbias)
#                     save_hist(dvbias, "dVisible bias {0}".format(name),
#                               fn_hist_dvbias)
#                 except IndexError:
#                     pass
#
#             h_act_all = self.out(data)
#             sum_hidd = h_act_all.mean(axis=(0,2,3)) if self.convolutional \
#                                                         else h_act_all.mean(axis=0)
#             save_hist(sum_hidd, 'hAct ({0})'.format(name),
#                       fn_hist_hact)
#
#             """ plotting weights in ipython notebook """
#             if in_ipynb():
#                 import Image as Img
#                 from IPython import display
#                 im = Img.open(fn_weights)
#                 im.resize((np.asarray(im.size)*5)).save("resized.png")
#                 display.clear_output()
#                 display.display(display.Image(filename="resized.png", retina=True))
#         except:
#             LOGGER.warning("A problem occured during plotting, not all plots could be created.")
#             traceback.print_exc()
