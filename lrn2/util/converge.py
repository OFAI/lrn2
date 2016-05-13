'''
Created on Dec 9, 2014

@author: Stefan Lattner
'''
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

class ConvergeDetect(object):
    def __init__(self, thresh, lag, smooth):
        """
        Convergence detector. Calculates sliding average values v over a
        sequence of costs and uses d = v(t) - v(t-'lag') to detect convergence.

        Parameters
        ----------

        thresh : float
            convergence, if d < thresh

        lag : int
            d = v(t) - v(t-'lag')

        smooth : float
            The smoothing factor for sliding average.
            v(t) = v(t-1) * smooth + v(t) * (1 - smooth)
        """

        self.thresh = thresh
        self.lag = lag
        self.smooth = smooth
        self.costs = None

    def add_curr_cost(self, cost):
        """
        Adds a cost and updates the sliding average.

        Parameters
        ----------

        cost : float
            Add the cost of a current training step

        Returns
        -------

        converged : boolean
            True, if converged (evaluation after cost added)
        """
        if self.costs is None:
            self.costs = [cost]
        else:
            self.costs.extend([self.costs[-1] * self.smooth + cost * (1 - self.smooth)])
        return self.converged()

    def converged(self):
        """
        Checks if training has converged.

        Returns
        -------

        converged : boolean
            True, if converged
        """
        try:
            diff = self.costs[-self.lag] - self.costs[-1]
            print "Diff = {0}".format(diff)
            return diff < self.thresh
        except IndexError:
            return False
        except TypeError:
            raise ValueError("Use add_curr_cost() before you ask for convergence.")

class ConvergeDetect2(object):

    def __init__(self, thresh=.05, smooth=.95, min_epochs = 10):
        """ Detects convergence based on gradient of smoothed cost function.

        Parameters
        ----------

        thresh : float
            convergence, if d < thresh

        lag : int
            d = v(t) - v(t-'lag')

        smooth : float
            The smoothing factor for sliding average.
            v(t) = v(t-1) * smooth + v(t) * (1 - smooth)
        """

        self.thresh = thresh
        self.smooth = smooth
        self.min_epochs = min_epochs
        self.costs = None

    def add_curr_cost(self, cost):
        """
        Adds a cost and updates the sliding average.

        Parameters
        ----------

        cost : float
            Add the cost of a current training step

        Returns
        -------

        converged : boolean
            True, if converged (evaluation after cost added)
        """
        if self.costs is None:
            self.costs = [cost]
        else:
            self.costs.extend([self.costs[-1] * self.smooth + cost * (1 - self.smooth)])
        return self.converged()

    def converged(self):
        """
        Checks if training has converged.

        Returns
        -------

        converged : boolean
            True, if converged
        """
        if self.min_epochs > len(self.costs):
            return False

        try:
            diff = np.gradient(self.costs[-4:], edge_order=2)[-1] * -1
            LOGGER.debug("Slope cost curve = {0:.4f}".format(diff))
            return diff < self.thresh
        except IndexError:
            # Not enough past data available
            return False
        except ValueError:
            return False
#         except TypeError:
#             raise ValueError("Use add_curr_cost() before you ask for convergence.")