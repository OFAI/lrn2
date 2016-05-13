'''
Created on Jul 10, 2014

@author: Stefan Lattner
'''

import numpy as np
from matplotlib.pyplot import figure, show, ion, axvline, savefig, clf

#ion()

class PlotMusic(object):
    """
    Plots a piano roll representation of a MIDI Table, e.g. as returned by
    load_midi_files(). Optionally, adds a subplot with a continuous function,
    fitted to the notes.
    """
    def __init__(self):
        '''
        Constructor, no params
        '''
        self.segmentation = []
        self.annotation = []
        self.plot_nr = 0

    def add_annotation(self, annotation, offset=0, color = 'blue'):
        """
        Sets an annotation function (array) where the n-th value in the array
        corresponds to the n-th note of the song
        """
        assert len(annotation) > 0 # Length of annotation array has to be > 0
        self.annotation.append([annotation, offset, color])

    def add_segment(self, segmentation, color = "blue"):
        """
        Sets segmentation information where values in the array
        corresponds to first notes in segments
        """
        self.segmentation.append([segmentation, color])

    def set_music(self, music):
        """
        Sets the MIDI table as returned by load_midi_files()
        """
        self.music = music

    def plot(self, fn="default.png"):

        clf()
        fig = figure(1)

        ax = fig.add_subplot(211, autoscale_on=True)

        if len(self.annotation) > 0:
            m_v = np.max(self.annotation[-1][0])
        else:
            m_v = 1

        x_ticks = [i * 4 for i in range(int((4 + self.music['onset'][-1])/4))]
        ax.set_xticks(x_ticks, minor=False)

        ax.grid(color='grey', linestyle='-', linewidth=0.5)
        for i in range(self.music.shape[0]):
            curr_note = self.music[:][i]
            color = 0.5
            try:
                color = 1.0 * self.annotation[0][0][i-self.annotation[0][1]] / m_v
                color = 0.
            except:
                pass

            ax.broken_barh([(curr_note['onset'], curr_note['duration'])], (curr_note['pitch'], 1), facecolors=(color, color, color))



        if self.annotation is not None:
            ax2 = fig.add_subplot(212, autoscale_on=True, sharex=ax)
            ax2.grid(color='grey', linestyle='-', linewidth=0.5)

            for annot in self.annotation:
                off = len(self.music['pitch'][:]) - len(annot[0]) - annot[1]
                off = None if off == 0 else off * -1
                if off < 0:
                    ax2.plot(self.music['onset'][annot[1]:off], annot[0], color=annot[2])
                else:
                    ax2.plot(self.music['onset'][:], annot[0][:-off], color=annot[2])

        for i, s in enumerate(self.segmentation):
            for ss in s[0]:
                axvline(x=self.music['onset'][ss], ymin=1-0.25/(i+1), linewidth=4, color=s[1])

#         fig.show()
#         show()
        savefig(fn)



if __name__ == '__main__':
    p = PlotMusic()
    p.plot()
