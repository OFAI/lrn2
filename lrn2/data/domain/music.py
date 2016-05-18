#!/usr/bin/env python

import os
import re
import abc
import numpy as np
import scipy.sparse as sp
import logging

from math import ceil
from itertools import combinations
from gammatone.gtgram import gtgram
from lrn2.nn_bricks.utils import fx, load_pyc_bz, save_pyc_bz
from scipy.signal.signaltools import convolve2d
from lrn2.data.domain.viewpoint import ViewPoint

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

MAJOR = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
MINOR = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]

class PianoRollVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, min_pitch = 0, max_pitch = 128, time_div = 0.5,
                 info = False, max_length = 1e16):
        super(PianoRollVP, self).__init__()
        self.min_pitch = int(min_pitch)
        self.max_pitch = int(max_pitch)
        self.time_div = time_div
        self.notes_dis = 0.
        self.onset_dis = 0.
        self.note_count = 0.
        self.info = info
        self.max_length = max_length / self.time_div

    @property
    def size(self):
        return self.max_pitch - self.min_pitch

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    def raw_to_repr(self, raw_data, label):
        pitches = self.get_pitch_from_raw_data(raw_data)
        pitches = np.cast['int'](pitches)
        durs = self.get_duration_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        length = (onsets[-1] + durs[-1]) / self.time_div
        piano_roll = np.zeros((int(length), int(PianoRollVP.size.fget(self))))
        notes = sorted(zip(pitches, onsets, durs, range(len(pitches))), key = lambda x : x[0])
        for i in range(len(notes)-1):
            if notes[i][2] // self.time_div > 1:
                if notes[i][0] == notes[i+1][0]:
                    end = (notes[i][1] + notes[i][2]) // self.time_div
                    if end >= (notes[i+1][1] // self.time_div):
                        durs[notes[i][3]] -= self.time_div
                    
        for pitch, onset, dur in zip(pitches, onsets, durs):
            off = int((onset+dur) // self.time_div)
            on = int((onset // self.time_div))
            if on == off:
                self.notes_dis += 1
            
            off = int(np.minimum(on + self.max_length, off))
            try:
                if on > 0 and piano_roll[on-1, self.max_pitch-pitch] == 1:
                    self.onset_dis += 1                    
                piano_roll[on:off, self.max_pitch-pitch] = 1
                self.note_count += 1
            except IndexError as e:
                if self.info:
                    LOGGER.warning("Some notes are outside the range min/max pitch: {0}".format(e))
        
        if self.notes_dis > 0 and self.info:
            LOGGER.warning("Ratio notes disappeared (overall): {0}".format(self.notes_dis / self.note_count))
            
        if self.onset_dis > 0 and self.info:
            LOGGER.warning("Ratio onsets disappeared (overall): {0}".format(self.onset_dis / self.note_count))

#         emptyness = 100.0 * len(np.where(np.sum(piano_roll, axis = 1) == 0)) / length
#         print "Empty %:", emptyness
        
        return sp.csr_matrix(piano_roll)

class PianoRollIntervalsVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, time_div = 0.25):
        super(PianoRollIntervalsVP, self).__init__()
        self.time_div = time_div

    @property
    def size(self):
        return 12

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")
        
    def raw_to_repr(self, raw_data, label):
        pitches = self.get_pitch_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        offsets = self.get_duration_from_raw_data(raw_data) + onsets
        hsets, durations, _ = homophonic_sets(pitches, onsets, offsets)
        time_div = self.time_div
        
        length = offsets[-1] / time_div     
        result = np.zeros((length, self.size), dtype = np.bool)
        
        start = 0
        for pset, end in zip(hsets, np.cumsum(durations)):
            intervals = np.asarray([abs(a - b) for a,b in combinations(pset, 2)])
            s = int(start // time_div)
            e = int(end // time_div)
            intervals = np.delete(intervals, np.where(intervals > 11))
            ints = np.cast['int16'](intervals % 12)
            result[s:e, ints] = True
            start = end
        return np.fliplr(result)
    
class PianoRollApproachIntervalsVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, time_div = 0.25):
        super(PianoRollApproachIntervalsVP, self).__init__()
        self.time_div = time_div

    @property
    def size(self):
        return 12

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")
        
    def raw_to_repr(self, raw_data, label):
        pitches = self.get_pitch_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        offsets = self.get_duration_from_raw_data(raw_data) + onsets
        hsets, durations, _ = homophonic_sets(pitches, onsets, offsets)
        time_div = self.time_div
        
        length = offsets[-1] / time_div     
        result = np.zeros((length, self.size), dtype = np.bool)
        
        start = 0
        last_pset = []
        for pset, end in zip(hsets, np.cumsum(durations)):
            intervals = np.asarray([abs(a - b) for a in pset for b in last_pset])
            s = int(start // time_div)
            e = int(end // time_div)
            intervals = np.delete(intervals, np.where(intervals > 11))
            ints = np.cast['int16'](intervals % 12)
            result[s:e, ints] = True
            start = end
            last_pset = pset
        return np.fliplr(result)
    
class PianoRollPerceivedTensionVP(PianoRollVP):
    
    TENSION_ORDER = [0, 7, 5, 4, 3, 2, 6, 1]
    
    def __init__(self, min_pitch, max_pitch, time_div = 0.25, size = 8,
                 binary = True):
        PianoRollVP.__init__(self, min_pitch, max_pitch, time_div)
        self.size_ = size if binary else 1
        self.binary = binary
        self.time_div = time_div
        
    @property
    def size(self):
        return self.size_
    
    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")
        
    def interval_tension(self, intervals):
        result = []
        for ints in intervals:
            sum = 0.
            for i in ints:
                try:
                    sum += self.TENSION_ORDER.index(i)
                except ValueError:
                    sum += self.TENSION_ORDER.index(12 - i)
            if len(ints) > 0:
                result.append(sum / len(ints))
            else:
                result.append(0.)
        return np.asarray(result)
        
    def raw_to_repr(self, raw_data, label):
        pitches = self.get_pitch_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        offsets = self.get_duration_from_raw_data(raw_data) + onsets
        hsets, durations, _ = homophonic_sets(pitches, onsets, offsets)
                
        intervals_vert = []
        # Vertical intervals 
        for pset in hsets:
            intervals = np.asarray([abs(a - b) for a,b in combinations(pset, 2)])
            ints = np.unique(np.cast['int16'](intervals % 12))
            intervals_vert.append(ints)
            
                    
        tension_vert = self.interval_tension(intervals_vert)
        
        tension_comb = 1.0 * tension_vert
        tension_comb -= np.min(tension_comb)
        tension_comb /= (np.max(tension_comb) + 1e-7) 
        
        result = np.zeros((offsets[-1] / self.time_div, self.size), dtype = np.bool)
        start = 0
        
        if self.binary:
            tension_comb *= (self.size - 1)
            tension_comb = np.round(tension_comb)    
            
            for i, (t, end) in enumerate(zip(tension_comb, np.cumsum(durations))):
                s = int(start / self.time_div)
                e = int(end / self.time_div)
                result[s:e,:int(t)] = True
                if len(hsets[i]) == 0:
                    result[s:e,-1:] = True
                start = end
                
            return np.fliplr(result)
        else:
            for i, (t, end) in enumerate(zip(tension_comb, np.cumsum(durations))):
                s = int(start / self.time_div)
                e = int(end / self.time_div)
                result[s:e,0] = t
                if len(hsets[i]) == 0:
                    result[s:e,0] = 0.
                start = end
                
            return result
    
class PianoRollPerceivedTensionAppVP(PianoRollVP):
    
    TENSION_ORDER = [0, 7, 5, 4, 3, 2, 6, 1]
    
    def __init__(self, min_pitch, max_pitch, time_div = 0.25, size = 8, binary = True):
        PianoRollVP.__init__(self, min_pitch, max_pitch, time_div)
        self.size_ = size if binary else 1
        self.binary = binary
        self.time_div = time_div
        
    @property
    def size(self):
        return self.size_
    
    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")
        
    def interval_tension(self, intervals):
        result = []
        for ints in intervals:
            sum = 0.
            for i in ints:
                try:
                    sum += self.TENSION_ORDER.index(i)
                except ValueError:
                    sum += self.TENSION_ORDER.index(12 - i)
            if len(ints) > 0:
                result.append(sum / len(ints))
            else:
                result.append(0.)
        return np.asarray(result)
        
    def raw_to_repr(self, raw_data, label):
        pitches = self.get_pitch_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        offsets = self.get_duration_from_raw_data(raw_data) + onsets
        hsets, durations, _ = homophonic_sets(pitches, onsets, offsets)
            
        intervals_appr = [[],]
        last_pset = hsets[0]
        # Approach intervals 
        for pset in hsets[1:]:
            intervals = np.asarray([abs(a - b) for a in last_pset for b in pset])
            ints = np.unique(np.cast['int16'](intervals % 12))
            intervals_appr.append(ints)
            last_pset = pset
            
        tension_appr = self.interval_tension(intervals_appr)
        
        
        tension_comb = 1.0 * tension_appr
        tension_comb -= np.min(tension_comb)
        tension_comb /= (np.max(tension_comb) + 1e-7) 
        
        result = np.zeros((offsets[-1] / self.time_div, self.size), dtype = np.bool)
        start = 0

        if self.binary:
            tension_comb *= (self.size - 1)
            tension_comb = np.round(tension_comb)

            for i, (t, end) in enumerate(zip(tension_comb, np.cumsum(durations))):
                s = int(start / self.time_div)
                e = int(end / self.time_div)
                result[s:e,:int(t)] = True
                if len(hsets[i]) == 0:
                    result[s:e,-1:] = True
                start = end
                
            return np.fliplr(result)
        else:
            for i, (t, end) in enumerate(zip(tension_comb, np.cumsum(durations))):
                s = int(start / self.time_div)
                e = int(end / self.time_div)
                result[s:e,0] = t
                if len(hsets[i]) == 0:
                    result[s:e,0] = 0.
                start = end
                
            return result
                
class PianoRollTonalTensionVP(PianoRollVP):
    def __init__(self, min_pitch, max_pitch, time_div = 0.25, size = 8, context_width = 17,
                 binary = True):
        PianoRollVP.__init__(self, min_pitch, max_pitch, time_div)
        self.size_ = size if binary else 1     
        self.binary = binary 
        self.context_width = context_width
        
    @property
    def size(self):
        return self.size_
    
    def raw_to_repr(self, raw_data, label):
        piano_roll = PianoRollVP.raw_to_repr(self, raw_data, label).todense()
        piano_roll = np.asarray(piano_roll)
        poly = np.squeeze(np.sum(piano_roll, axis = 1))
        
        length = piano_roll.shape[0]
        
        chroma_maj = chromagram(piano_roll, MAJOR, filt_width = self.context_width)
        chroma_min = chromagram(piano_roll, MINOR, filt_width = self.context_width)
        chroma = np.hstack((chroma_maj, chroma_min))
        chroma -= (np.min(chroma, axis = 1, keepdims = True))
        chroma /= (np.max(chroma, axis = 1, keepdims = True) + 1e-6)
        
        chroma_maj_loc = chromagram(piano_roll, MAJOR, filt_width = 1)
        chroma_min_loc = chromagram(piano_roll, MINOR, filt_width = 1)
        chroma_loc = np.hstack((chroma_maj_loc, chroma_min_loc))
        chroma_loc -= (np.min(chroma_loc, axis = 1, keepdims = True))
        chroma_loc /= (np.max(chroma_loc, axis = 1, keepdims = True) + 1e-6)
        
        max_vals = np.maximum(chroma, chroma_loc)
        min_vals = np.minimum(chroma, chroma_loc)
        
        tonal_tension = np.sum(max_vals / (min_vals + 1e-7), axis = 1)
        tonal_tension[np.where(poly == 0.)] = 0.
        tonal_tension -= np.min(tonal_tension)
        tonal_tension /= np.max(tonal_tension)
        
        result = np.zeros((length, self.size), dtype = np.bool)

        if self.binary:
            tonal_tension *= self.size - 1
            tonal_tension = np.round(tonal_tension) 
                
            for i, t in enumerate(tonal_tension):
                result[i,:t] = True
                
            # mark empty parts
            result[np.where(poly == 0.),-1:] = True 
            
            return np.fliplr(result)
        else:
            for i, t in enumerate(tonal_tension):
                result[i,0] = t
                
            # mark empty parts
            result[np.where(poly == 0.),0] = 0. 
            
            return result
        

class PianoRollPolyphonyVP(PianoRollVP):
    def __init__(self, min_pitch = 0, max_pitch = 128, time_div = 0.250, size = 1):
        PianoRollVP.__init__(self, min_pitch, max_pitch, time_div)
        self.size_ = size
        
    @property
    def size(self):
        return self.size_
    
    def raw_to_repr(self, raw_data, label):
        piano_roll = PianoRollVP.raw_to_repr(self, raw_data, label).todense()

        N = piano_roll.shape[0]
        data = np.ones(N, dtype = np.bool)
    
        piano_roll = np.asarray(piano_roll)
        poly = np.squeeze(np.sum(piano_roll, axis = 1))
        
        if np.max(poly) >= self.size:
            LOGGER.warning("Polyphony size is too small, representing {0} as {1}".format(np.max(poly), self.size-1))
            poly[np.where(poly >= self.size)] = self.size-1
        
#         indptr = np.arange(N+1, dtype = np.int)
#         
#         return sp.csr_matrix((data, poly, indptr), shape = (N, self.size))

        result = np.zeros((N, self.size), dtype = np.bool)
        
        for i, p in enumerate(poly):
            result[i,:p] = True
            
        return np.fliplr(result)
            
class PianoRollPitchClassVP(PianoRollVP):
    def __init__(self, min_pitch = 0, max_pitch = 128, time_div = 0.250):
        PianoRollVP.__init__(self, min_pitch, max_pitch, time_div)
        
    @property
    def size(self):
        return 12
    
    def raw_to_repr(self, raw_data, label):
        piano_roll = PianoRollVP.raw_to_repr(self, raw_data, label).todense()
        piano_roll = np.asarray(piano_roll)
        
        pitch_class = [np.sum(piano_roll[:,pitch::12], axis = 1).tolist() for pitch in range(12)]        
        pitch_class = np.vstack(pitch_class)
        
        pitch_class[np.where(pitch_class > 1)] = 1
        
        return sp.csr_matrix(np.transpose(pitch_class))
        
    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)
    

    
class PianoRollRhythmVP(PianoRollVP):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, min_pitch = 0, max_pitch = 128, time_div = 0.250, size = 1):
        PianoRollVP.__init__(self, min_pitch, max_pitch, time_div)
        self.size_ = size

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")
    @property
    def size(self):
        return self.size_
    
    def raw_to_repr(self, raw_data, label):
        pitches = self.get_pitch_from_raw_data(raw_data)
        durs = self.get_duration_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        
        notes = sorted(zip(pitches, onsets, durs, range(len(pitches))), key = lambda x : x[0])
        for i in range(len(notes)-1):
            if notes[i][2] // self.time_div > 1:
                if notes[i][0] == notes[i+1][0]:
                    end = (notes[i][1] + notes[i][2]) // self.time_div
                    if end >= (notes[i+1][1] // self.time_div):
                        durs[notes[i][3]] -= self.time_div
        
        rng = self.max_pitch - self.min_pitch
        quant = rng // self.size + 1
        length = (onsets[-1] + durs[-1]) / self.time_div
        result = np.zeros((length, self.size))
        for onset, pitch in zip(onsets, pitches):
            on = (onset/self.time_div)
            try:
                result[on, (self.max_pitch - pitch) // quant] = 1
            except IndexError:
                # very seldomly we can exceed the range
                result[on, (self.max_pitch - pitch) // quant - 1] = 1
                #LOGGER.warning("RhythmVP: min/max pitch problem")

        return sp.csr_matrix(result)
    
class PianoRollTonalityVP(PianoRollVP):

    def __init__(self, min_pitch = 0, max_pitch = 128, time_div = .25, filt_width = 17):
        PianoRollVP.__init__(self, min_pitch, max_pitch, time_div)
        self.filt_width = filt_width
        
    @property
    def size(self):
        return 24
        
    def raw_to_repr(self, raw_data, label):
        piano_roll = PianoRollVP.raw_to_repr(self, raw_data, label).todense()
        piano_roll = np.asarray(piano_roll)
        
        chroma_maj = chromagram(piano_roll, MAJOR, self.filt_width)
        chroma_min = chromagram(piano_roll, MINOR, self.filt_width)
        
        chroma = np.hstack((chroma_maj, chroma_min))
        chroma -= (np.min(chroma, axis = 1, keepdims = True))
        chroma /= (np.max(chroma, axis = 1, keepdims = True) + 1e-6)

        return chroma
        
    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)


def chromagram(piano_roll, profile, filt_width = 17):
    filt = np.array([profile])
    filt = np.repeat(filt, filt_width, axis=0)

    conv_out = convolve2d(
        in1 = piano_roll,
        in2 = filt[:,::-1],
        mode = 'same',
    )
    
    key_prof = conv_out[:,:-(conv_out.shape[1] % 12)]
    key_prof = np.reshape(key_prof, newshape=(conv_out.shape[0], -1, 12))
    key_prof = np.sum(key_prof, axis = 1)
    
    return key_prof 
   
    
class PianoRollBeatInMeasureVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    """
    Parameters
    ----------
    
    time_div: float
        Time division (in beats)
        
    max_depth: integer
        The max depth of the binary rhythm tree. Is the negative power of 2.
        e.g. max_depth of 3 can represent a denominator of 2**3 = 8 
        (for e.g. 6/8). Depth of 2 can represent e.g. 4/4  
    """
    
    t22 = [[0,1],]
    t24 = [[0,0], [0,1]]
    t34 = [[0,1,1],[0,0,1]]
    t44 = [[0,0,1,1], [0,1,0,1]]
    t38 = [[0,0,0], [0,1,1], [0,0,1]]
    t68 = [[0,0,0,1,1,1], [0,1,1,0,1,1], [0,0,1,0,0,1]]
    
    time_sigs = {"22": t22,
                 "24": t24,
                 "34": t34,
                 "44": t44,
                 "38": t34,
                 "68": t68}
    
    def __init__(self, time_div = 0.5, max_depth = 3):
        super(PianoRollBeatInMeasureVP, self).__init__()
        self.time_div = time_div
        self.max_depth = max_depth
        
    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")
        
    def get_time_signature_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")
        
    @property
    def size(self):
        return self.max_depth

    def raw_to_repr(self, raw_data, label):
        num, denom = self.get_time_signature_from_raw_data(raw_data)
        durs = self.get_duration_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        
        length = (onsets[-1] + durs[-1]) / self.time_div
        
        try:
            measure = self.time_sigs[str(num)+ str(denom)]
        except KeyError:
            raise Exception("Time signature not found: {0}/{1}, please add it in the 'PianoRollBeatInMeasureVP' viewpoint class.".format(num, denom))
            
        to_add = self.size - len(measure)
        
        measure = np.vstack((np.zeros((to_add, len(measure[0]))), measure))
#         measure = np.fliplr(measure)
        measure = np.repeat(measure, int(1 / self.time_div), axis = 1)
                
        to_tile = ceil(length / len(measure[0]))
        
        result = np.tile(measure, to_tile)
        
        result = result[:,:length]
        return sp.csr_matrix(np.transpose(result))

    def repr_to_visual(self, binary_data):
        return binary_data

class PianoRollOnsetVP(PianoRollVP):

    def __init__(self, min_pitch = 0, max_pitch = 128, time_div = 0.250):
        PianoRollVP.__init__(self, min_pitch, max_pitch, time_div)

    def raw_to_repr(self, raw_data, label):
        pitches = self.get_pitch_from_raw_data(raw_data)
        durs = self.get_duration_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        length = (onsets[-1] + durs[-1]) / self.time_div
        piano_roll = np.zeros((length, self.size))
        for pitch, onset, dur in zip(pitches, onsets, durs):
            on = (onset/self.time_div)
            piano_roll[on, self.max_pitch-pitch] = 1

        return sp.csr_matrix(piano_roll)

class PianoRollOffsetVP(PianoRollVP):

    def __init__(self, min_pitch = 0, max_pitch = 128, time_div = 0.250):
        PianoRollVP.__init__(self, min_pitch, max_pitch, time_div)

    def raw_to_repr(self, raw_data, label):
        pitches = self.get_pitch_from_raw_data(raw_data)
        durs = self.get_duration_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        length = (onsets[-1] + durs[-1]) / self.time_div
        piano_roll = np.zeros((length, self.size))
        for pitch, onset, dur in zip(pitches, onsets, durs):
            off = (onset+dur) / self.time_div
            piano_roll[off-1, self.max_pitch-pitch] = 1

        return sp.csr_matrix(piano_roll)

class PianoRollOnBeatVP(PianoRollVP):

    def __init__(self, min_pitch = 0, max_pitch = 128, time_div = 0.250):
        PianoRollVP.__init__(self, min_pitch, max_pitch, time_div)

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    def raw_to_repr(self, raw_data, label):
        sig = self.get_time_signature_from_raw_data(raw_data)
        draw_every = (1.0 * (4 * sig[0] / sig[1]) / self.time_div)
        durs = self.get_duration_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        length = (onsets[-1] + durs[-1]) / self.time_div
        piano_roll = np.zeros((length, self.size))
        piano_roll[::draw_every, :] = 1

        return sp.csr_matrix(piano_roll)

class BoundaryStrengthVP(ViewPoint):

    def __init__(self, bs_file, min = 10, max = 50, n_batches = 5, offset = 0):
        super(BoundaryStrengthVP, self).__init__()
        self.curves = load_pyc_bz(bs_file)
        self.min = min
        self.max = max
        self.offset = offset
        self.quantize = n_batches

    @property
    def size(self):
        return self.quantize if self.quantize else 1

    def raw_to_repr(self, raw_data, label):
        N = raw_data.shape[0]
        data = np.ones(N, dtype = np.bool)

        label = label.split(".")[0]
        try:
            ic_curve = self.curves[label]
            ic_curve = np.r_[[np.average(ic_curve)]*self.offset, ic_curve]
        except KeyError:
            ic_curve = np.zeros((N + self.offset,))

        # trim data to allowed range
        ic_curve[ic_curve < self.min] = self.min
        ic_curve[ic_curve >= self.max] = self.max-1
        ic_curve -= self.min

        if self.quantize:
            ic_curve = (ic_curve * self.quantize) / self.max

        indptr = np.arange(N+1, dtype = np.int)

        return sp.csr_matrix((data, ic_curve, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

class DurationVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, min_dur=0, max_dur=4, n_batches=16):
        super(DurationVP, self).__init__()
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.quantize = n_batches

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return self.quantize if self.quantize else self.max_dur - self.min_dur

    def raw_to_repr(self, raw_data, label):
        N = raw_data.shape[0]
        data = np.ones(N, dtype = np.bool)

        indices =  self.get_duration_from_raw_data(raw_data)

        # trim data to allowed range
        indices[indices < self.min_dur] = self.min_dur
        indices[indices >= self.max_dur] = self.max_dur-1
        indices -= self.min_dur

        if self.quantize:
            indices = (indices * self.quantize) / self.max_dur

        indptr = np.arange(N+1, dtype = np.int)

        return sp.csr_matrix((data, indices, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

class IoiVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, min_ioi=0, max_ioi=4, n_batches=16):
        super(IoiVP, self).__init__()
        self.min_ioi = min_ioi
        self.max_ioi = max_ioi
        self.quantize = n_batches

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return self.quantize if self.quantize else self.max_ioi - self.min_ioi

    def raw_to_repr(self, raw_data, label):
        N = raw_data.shape[0]
        data = np.ones(N, dtype = np.bool)

        indices = np.r_[0, np.diff(self.get_onset_from_raw_data(raw_data))]

        # trim data to allowed range
        indices[indices < self.min_ioi] = self.min_ioi
        indices[indices >= self.max_ioi] = self.max_ioi-1
        indices -= self.min_ioi

        if self.quantize:
            indices = (indices * self.quantize) / self.max_ioi

        indptr = np.arange(N+1, dtype = np.int)

        return sp.csr_matrix((data, indices, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

class OoiVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, min_ooi=0, max_ooi=2, n_batches=4):
        super(OoiVP, self).__init__()
        self.min_ooi = min_ooi
        self.max_ooi = max_ooi
        self.quantize = n_batches

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return self.quantize if self.quantize else self.max_ooi - self.min_ooi

    def raw_to_repr(self, raw_data, label):
        N = raw_data.shape[0]
        data = np.ones(N, dtype = np.bool)

        indices = np.r_[0, np.diff(self.get_onset_from_raw_data(raw_data))]
        indices = indices - np.r_[0, self.get_duration_from_raw_data(raw_data)[:-1]]

        # trim data to allowed range
        indices[indices < self.min_ooi] = self.min_ooi
        indices[indices >= self.max_ooi] = self.max_ooi-1
        indices -= self.min_ooi

        if self.quantize:
            indices = (indices * self.quantize) / self.max_ooi

        indptr = np.arange(N+1, dtype = np.int)

        return sp.csr_matrix((data, indices, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

class PitchIntervalVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, max_abs_int = 24):
        super(PitchIntervalVP, self).__init__()
        self.max_abs_int = max_abs_int

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return 1 + 2 * self.max_abs_int

    def raw_to_repr(self, raw_data, label):
        N = raw_data.shape[0]
        data = np.ones(N, dtype = np.bool)

        indices =  np.r_[0, np.diff(self.get_pitch_from_raw_data(raw_data))]

        # trim data to allowed range
        indices[indices < -self.max_abs_int] = -self.max_abs_int
        indices[indices > self.max_abs_int] = self.max_abs_int
        indices += self.max_abs_int

        indptr = np.arange(N+1, dtype = np.int)
        return sp.csr_matrix((data, indices, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

class PitchContourVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(PitchContourVP, self).__init__()

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return 3

    def raw_to_repr(self, raw_data, label):
        N = raw_data.shape[0]
        data = np.ones(N, dtype = np.bool)

        indices =  np.r_[0, np.sign(np.diff(self.get_pitch_from_raw_data(raw_data)))] + 1

        indptr = np.arange(N+1, dtype = np.int)
        return sp.csr_matrix((data, indices, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

class PitchIntervalAbsVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, max_abs_int = 12):
        super(PitchIntervalAbsVP, self).__init__()
        self.max_abs_int = max_abs_int

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return 1 + self.max_abs_int

    def raw_to_repr(self, raw_data, label):
        N = raw_data.shape[0]
        data = np.ones(N, dtype = np.bool)

        indices =  np.r_[0, np.abs(np.diff(self.get_pitch_from_raw_data(raw_data)))]

        # trim data to allowed range
        indices[indices > self.max_abs_int] = self.max_abs_int

        indptr = np.arange(N+1, dtype = np.int)
        return sp.csr_matrix((data, indices, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

class PitchVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, min_pitch = 0, max_pitch = 128):
        super(PitchVP, self).__init__()
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return self.max_pitch - self.min_pitch

    def raw_to_repr(self, raw_data, label):
        indices = self.get_pitch_from_raw_data(raw_data)
        N = len(indices)
        data = np.ones(N, dtype = np.bool)

        # trim data to allowed range
        indices[indices < self.min_pitch] = self.min_pitch
        indices[indices >= self.max_pitch] = self.max_pitch -1
        indices -= self.min_pitch

        indptr = np.arange(N+1, dtype = np.int)
        return sp.csr_matrix((data, indices, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)


class PitchClassVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, modulo=12):
        super(PitchClassVP, self).__init__()
        self.modulo=modulo

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return self.modulo

    def raw_to_repr(self, raw_data, label):
        indices = self.get_pitch_from_raw_data(raw_data)
        N = len(indices)
        data = np.ones(N, dtype = np.bool)

        # modulo
        indices =  np.mod(indices, self.modulo)

        indptr = np.arange(N+1, dtype = np.int)
        
        return sp.csr_matrix((data, indices, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

class OctaveVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self, max_pitch=127):
        super(OctaveVP, self).__init__()
        self.max_pitch = max_pitch

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return int(self.max_pitch / 12)

    def raw_to_repr(self, raw_data, label):
        indices = self.get_pitch_from_raw_data(raw_data)
        N = len(indices)
        data = np.ones(N, dtype = np.bool)

        indices =  (indices / 12).astype('int')

        indptr = np.arange(N+1, dtype = np.int)
        return sp.csr_matrix((data, indices, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

class BeatInMeasureNaiveVP(ViewPoint):
    # Naively assumes 4/4 division
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_batches = 16):
        super(BeatInMeasureNaiveVP, self).__init__()
        self.quantize = n_batches

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return self.quantize

    def raw_to_repr(self, raw_data, label):
        indices = self.get_onset_from_raw_data(raw_data)
        N = len(indices)
        data = np.ones(N, dtype = np.bool)

        # 4000 = length of one measure
        indices =  (np.mod(indices, 4000) / float(4000) * self.quantize).astype('int')

        indptr = np.arange(N+1, dtype = np.int)
        return sp.csr_matrix((data, indices, indptr), shape = (N, self.size))

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)
       
class HomophonicVP(ViewPoint):
    # __metaclass__ = abc.ABCMeta

    harmonics = np.asarray([0, 12, 19, 24, 28, 31])
    
    def __init__(self, quant_thresh = 0.1, shuffle = False, pitch_class = False,
                 min_sim_notes = 1, min_dur_slice = 0., add_harmonics = False,
                 min_dur_note = 0.):
        super(HomophonicVP, self).__init__()
        self.pitch_count = []
        self.quant_thresh = quant_thresh
        self.shuffle = shuffle
        self.pitch_class = pitch_class
        self.min_sim_notes = min_sim_notes
        self.min_dur_slice = min_dur_slice
        self.min_dur_note = min_dur_note
        self.durations = []
        self.simult_start = []
        self.add_harmonics = add_harmonics

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return 12 if self.pitch_class else 128

    def raw_to_repr(self, raw_data, label):

        if len(raw_data) == 0:
            return []
        
        pitches = self.get_pitch_from_raw_data(raw_data)
        onsets = self.get_onset_from_raw_data(raw_data)
        durs = self.get_duration_from_raw_data(raw_data)
        offsets = durs + onsets
        
        quant_thresh = self.quant_thresh
        min_dur_slice = self.min_dur_slice
        min_dur_note = self.min_dur_note
        min_sim_notes = self.min_sim_notes
        add_harmonics = self.add_harmonics
        size = self.size
        shuffle = self.shuffle
        harmonics = self.harmonics
                   
        pitches = pitches[np.where(durs >= min_dur_note)]
        onsets = onsets[np.where(durs >= min_dur_note)]
        offsets = offsets[np.where(durs >= min_dur_note)]
        
        pitches = pitches % self.size
        
        hsets, durations, simult_start = homophonic_sets(pitches, onsets, 
                                                         offsets, quant_thresh, 
                                                         min_dur_slice, 
                                                         min_sim_notes)
        
        print hsets
            
        for i in range(len(hsets)-1):
            assert hsets[i] != hsets[i+1], "adjacent homophonic sets have to be unequal."
            
        self.durations.append(durations)
        self.simult_start.append(simult_start)
        
        result = []
        for pitch_set in hsets:
            binary = np.zeros(shape = size, dtype=fx)
            
            harmonics = harmonics if add_harmonics else 0
            for p in np.cast['int'](pitch_set) % size:
                binary[harmonics + p] = 1
            
            if len(result) == 0 or binary.tolist() != result[-1].tolist(): 
                result.append(binary)

        if shuffle:
            np.random.shuffle(result)
            
        if len(result) == 0:
            return []
            
        return np.vstack(result)

    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)
    
def homophonic_sets(pitches, onsets, offsets, quant_thresh = 0.1, 
                    min_dur_slice = 0., min_sim_notes = 0,
                    include_multi_onsets = False):

    # saves [pitch, offset] tuples
    curr_pitches = np.asarray([-1,np.iinfo(np.int32).max], dtype = np.int32).reshape((1,-1))
    result = []
    
    simult_starts = []
    durations = []
        
    start = 0
    j = 0
    while start < max(offsets):
        # Find all notes starting at the same time as note nr i
        simult_start = 0
        while j < len(onsets) and onsets[j] - start < quant_thresh:
            # store pitch and corresponding offset
            if pitches[j] in curr_pitches[:,0]:
                curr_pitches[np.where(pitches[j] == curr_pitches[:,0]), 1] = offsets[j]
            else:
                curr_pitches = np.vstack((curr_pitches, [pitches[j], offsets[j]]))
                simult_start += 1
            j += 1
        
        try:
            next_start = max(curr_pitches[1:,1]) if j >= len(onsets) else \
                                    np.minimum(onsets[j], min(curr_pitches[:,1]))
            valid_slice = True
            if simult_start >= min_sim_notes:
                simult_starts.append(simult_start)
            else:
                valid_slice = False
    
            # j is now on the first onset of the next slice
            duration = next_start - start
            if min_dur_slice <= duration:
                durations.append(duration)
            else:
                valid_slice = False
            
            if valid_slice:
                new_pitches = sorted(curr_pitches[:,0].tolist()[1:])
                if len(result) == 0 or new_pitches != result[-1] or include_multi_onsets:
                    # Sort - first entry is dummy
                    result.append(sorted(curr_pitches[:,0].tolist()[1:]))
                
            # Delete notes ending at or before next_start                
            curr_pitches = np.delete(curr_pitches, np.where(curr_pitches[:,1] - next_start < quant_thresh), 0)
            start = next_start
        except ValueError as e:
            LOGGER.warning("Error calculating homophonic slices: {0}".format(e))
            break
        
    return result, durations, simult_starts
    
class GammatoneVP(ViewPoint):
    # __metaclass__ = abc.ABCMeta
    note_names = np.array(('C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'))
    note_names2 = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B', 'c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b']

    def __init__(self, audiofile_sets, cache_dir = None, sample_rate = 44100):
        super(GammatoneVP, self).__init__()
        self.pitch_count = []
        self.sample_rate = sample_rate
        self.gamma_spectra = {}
        self.notename_pat = re.compile("([ABCDEFG]b?)([0-9])")
        self.audiofiles = dict(((label, self.pitch_from_filename(fn)), fn)
                               for label, audiofiles in audiofile_sets.items()
                               for fn in audiofiles)
        self.audiofile_set_labels = audiofile_sets.keys()
        self.spectrum_options = {'window_time' : 1e-1,
                                 'hop_time' : 5e-2,
                                 'channels' : 2**8,
                                 'f_min' : 0}
        self.cache_fn = "gammatone_cache.pyc.bz"
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            try:
                self.gamma_spectra = load_pyc_bz(fn)(os.path.join(self.cache_dir,
                                                              self.cache_fn))
            except Exception as e:
                LOGGER.exception(e)
                pass

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return 150
        #return self.spectrum_options['channels']

    def pitch_from_filename(self, fn):
        """Extract the pitch symbol from a filename of the form
        "Piano.mf.Ab7.flac" and return the corresponding MIDI number.

        """
        notename = os.path.basename(fn).split('.')[2]
        notename_parts = self.notename_pat.search(notename)
        if notename_parts is None:
            raise ValueError("Cannot extract midi pitch from filename {0}".format(fn))

        pitch_name, octave = notename_parts.groups()
        idx_nn = np.where(self.note_names == pitch_name)[0][0]
        midi_nr = int(octave) * 12 + idx_nn
        #print "fn, idx_nn, pitch_name, octave, midi_nr", fn, idx_nn, pitch_name, octave, (midi_nr + 12)
        return midi_nr + 12

    def pitch_from_label(self, fn):
        """Extract the pitch symbol from a filename of the form
        "Piano.mf.Ab7.flac" and return the corresponding MIDI number.

        """
        
        try:
            notename = os.path.basename(fn).split('_')[2]
        except IndexError:
            LOGGER.warning("could not extract pitch nr from label: {0}".format(fn))
            return 0
        pitch_nr = int(self.note_names2.index(notename)) % 12
        return pitch_nr

    def raw_to_repr(self, raw_data, label):
        """This viewpoint is now used to transform ngrams of representations
        created by other viewpoints, therefore the method is called
        ngram_to_binary rather than raw_to_binary. A bit ad-hoc, but
        it suits our needs for now.

        pitch_ngrams is an M x N array containing M ngrams (size N) of integer MIDI pitch values

        """

        indices = self.get_pitch_from_raw_data(raw_data)
        indices += self.pitch_from_label(label)
        indices = self.center_ngrams(np.asarray([indices]), 48, 84)[0]

        self.pitch_count.extend(indices)
        for curr_label in self.audiofile_set_labels:
            gamma_repr = []
            valid_ngram = True
            if indices is not None:
                for pitch in indices:
                    try:
                        _ = self.gamma_spectra[(curr_label, pitch)]
                    except KeyError:
                        try:
                            audio_fn = self.audiofiles[(curr_label, pitch)]
                        except KeyError:
                            valid_ngram = False
                            LOGGER.warning("Pitch not found in audiofiles: {0}".format((curr_label, pitch)))
                            break

                        self.gamma_spectra[(curr_label, pitch)] = self.extract_gammatone(audio_fn)
#                         self.gamma_spectra[(curr_label, pitch)] = preprocessing.scale(self.gamma_spectra[(curr_label, pitch)])
                        self.gamma_spectra[(curr_label, pitch)] = self.normalize(self.gamma_spectra[(curr_label, pitch)])
                    if valid_ngram:
                        gamma_repr.append(self.gamma_spectra[(curr_label, pitch)])
                        #gamma_repr.append(self.gamma_spectra[(curr_label, np.random.randint(48,85))])
            else:
                LOGGER.warning("Skipping ngram, could not center trivially.")

        if self.cache_dir is not None:
            save_pyc_bz(self.gamma_spectra,
                        os.path.join(self.cache_dir, self.cache_fn))
        return np.vstack(gamma_repr)

    def normalize2(self, data):
        data = data - np.min(data)
        data = data / np.sum(data)
        return data

    def normalize(self, data):
        data = data - np.min(data)
        data = data / np.max(data)
        return data

    def center_ngrams(self, ngrams, min_pitch, max_pitch):
        """Transpose each n-gram by an integer multiple of an octave, to fit
        the pitch range to the n-gram between a minimum and maximum pitch.

        Parameters
        ----------
        ngrams : ndarray
            a 2D array of type int and size: `len(pitches)` x `n`, where
            each row contains an ngram of pitches
        min_pitch : int
            minimal allowed pitch (inclusive)
        max_pitch : int
            maximal allowed pitch (exclusive)

        Returns
        -------
        ngrams : ndarray
            a 2D array of type int and size: `len(pitches)` x `n`, where
            each row contains an ngram of pitches, transposed by zero or
            more octaves to fit between `min_pitch` and `max_pitch`

        """

        center = (max_pitch + min_pitch) / 2.

        min_pitches = np.min(ngrams, 1)
        max_pitches = np.max(ngrams, 1)
        mean_pitches = np.mean(ngrams, 1)
        off = (center - mean_pitches) % 12

        transp = np.zeros((ngrams.shape[0]), np.int)

        transp_idx = np.logical_or(min_pitches < min_pitch, max_pitches > max_pitch)
        transp[transp_idx] = np.floor((center - mean_pitches[transp_idx]) / 12).astype(np.int)
        transp[transp_idx][off[transp_idx] > 6] += 1

        transp_idx = np.logical_and(min_pitches + transp * 12 < min_pitch,
                                    max_pitches + transp * 12 < max_pitch - 12)
        transp[transp_idx] += 1

        transp_idx = np.logical_and(max_pitches + transp * 12 >= max_pitch,
                                    min_pitches + transp * 12 >= min_pitch + 12)
        transp[transp_idx] -= 1
        transp = transp.reshape((-1,1))
        return ngrams + transp * 12



    def extract_gammatone(self, audio_fn):
        import matplotlib.pyplot as plt
        LOGGER.info("extracting gamma representation of {0}".format(audio_fn))
        audio_wave, fs = self.read_audio_audiolab(audio_fn, mono = True)

        gtg = gtgram(audio_wave, self.sample_rate, **self.spectrum_options)
        gtg = np.mean(gtg[:150,:15], axis=1)

#         plt.clf()
#         plt.plot(gtg)
#         plt.show()
        return gtg


    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

    def read_audio_audiolab(self, fn, mono = False):
        """
        Get the signal from an audio file.

        :param fn: filename of audio file
        :param mono: if True, mix channels into a single channel

        :returns: A tuple (signal [nframes x nchannels], sample_freq)

        """

        # TODO: make sure we always return the same
        # type of data (int/float) and range

        from scikits.audiolab import Sndfile

        audio = Sndfile(fn)
        X = audio.read_frames(audio.nframes)

        if mono and audio.channels == 2:
            X = np.mean(X,1)

        return X, audio.samplerate



class GammatoneVPPoly(ViewPoint):
    # __metaclass__ = abc.ABCMeta
    note_names = np.array(('C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'))
    note_names2 = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B', 'c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b']

    def __init__(self, audiofile_sets, cache_dir = None, sample_rate = 44100,
                 quant_thresh = 0.01):
        super(GammatoneVPPoly, self).__init__()
        self.pitch_count = []
        self.sample_rate = sample_rate
        self.pitch_range = range(48, 84)
        self.gamma_spectra = {}
        self.pitches_not_available = []
        self.quant_thresh = quant_thresh
        self.notename_pat = re.compile("([ABCDEFG]b?)([0-9])")
        self.audiofiles = dict(((label, self.pitch_from_filename(fn)), fn)
                               for label, audiofiles in audiofile_sets.items()
                               for fn in audiofiles)
        self.audiofile_set_labels = audiofile_sets.keys()
        self.spectrum_options = {'window_time' : 2,
                                 'hop_time' : 10,
                                 'channels' : 2**8,
                                 'f_min' : 0}
        self.cache_fn = "gammatone_poly_cache.pyc.bz"
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            try:
                self.gamma_spectra = load_pyc_bz(os.path.join(self.cache_dir,
                                                              self.cache_fn))
            except Exception as e:
                LOGGER.exception(e)
                pass

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_duration_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @abc.abstractmethod
    def get_onset_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return 256
        #return self.spectrum_options['channels']

    def pitch_from_filename(self, fn):
        """Extract the pitch symbol from a filename of the form
        "Piano.mf.Ab7.flac" and return the corresponding MIDI number.

        """
        notename = os.path.basename(fn).split('.')[2]
        notename_parts = self.notename_pat.search(notename)
        if notename_parts is None:
            raise ValueError("Cannot extract midi pitch from filename {0}".format(fn))

        pitch_name, octave = notename_parts.groups()
        idx_nn = np.where(self.note_names == pitch_name)[0][0]
        midi_nr = int(octave) * 12 + idx_nn
        #print "fn, idx_nn, pitch_name, octave, midi_nr", fn, idx_nn, pitch_name, octave, (midi_nr + 12)
        return midi_nr + 12

    def pitch_from_label(self, fn):
        """Extract the pitch symbol from a label (fn) of the form
        "fugue_1_D.mid" and return the corresponding MIDI number.

        """
        try:
            notename = os.path.basename(fn).split('.')[0].split('_')[2]
        except IndexError:
            LOGGER.warning("could not extract pitch nr from label: {0} - assuming C/c.".format(fn))
            return 0
        pitch_nr = int(self.note_names2.index(notename)) % 12
        return pitch_nr

    def raw_to_repr(self, raw_data, label):
        """This viewpoint is now used to transform ngrams of representations
        created by other viewpoints, therefore the method is called
        ngram_to_binary rather than raw_to_binary. A bit ad-hoc, but
        it suits our needs for now.

        pitch_ngrams is an M x N array containing M ngrams (size N) of integer MIDI pitch values

        """

        if len(raw_data) == 0:
            return []
        
        indices = self.get_pitch_from_raw_data(raw_data)
#         indices += self.pitch_from_label(label)
        indices = self.center_ngrams(np.asarray([indices]), 48, 84)[0]

        onsets = self.get_onset_from_raw_data(raw_data)
        offsets = self.get_duration_from_raw_data(raw_data) + onsets
        
        quant_thresh = self.quant_thresh
        
#         print onsets
#         print offsets

        curr_pitches = np.asarray([0,np.iinfo(np.int64).max], dtype = fx).reshape((1,-1))

        gamma_repr = []
        i = 0
        while i < len(onsets):
            j = i
            while j < len(onsets) and onsets[j] - onsets[i] < quant_thresh:
                if int(indices[j]) in self.pitch_range:
                    if indices[j] in curr_pitches[:,0]:
                        curr_pitches[np.where(indices[j] == curr_pitches[:,0]), 1] = offsets[j]
                    else:
                        curr_pitches = np.vstack((curr_pitches, [indices[j], offsets[j]]))
                j += 1
#             print "on =", onsets[i]
#             print "where", np.where(onsets[i] - curr_pitches[:,1] > quant_thresh * -1)
#             print "pitches before", curr_pitches
            curr_pitches = np.delete(curr_pitches, np.where(onsets[i] - curr_pitches[:,1] > quant_thresh * -1), 0)
            curr_pitches_tup = tuple(sorted(curr_pitches[:,0].tolist()[1:]))
#             print "resulting pitches", curr_pitches_tup

            gamma = None
            new_entries = False
            if len(curr_pitches_tup) is not 0:
                try:
    #                 _ = self.gamma_spectra[tuple(curr_pitches.tolist())]
                    gamma = self.gamma_spectra[curr_pitches_tup]
                except KeyError:
                    new_entries = True
                    gamma = self.to_gammatone(curr_pitches_tup)

            if gamma is not None:
                gamma.shape = (len(gamma))
#                 gamma = preprocessing.scale(gamma)
                self.gamma_spectra[curr_pitches_tup] = gamma
                gamma_repr.append(gamma)

            i = j

        if new_entries:
            if self.cache_dir is not None:
                save_pyc_bz(self.gamma_spectra,
                            os.path.join(self.cache_dir, self.cache_fn))
            
        print "current dictionary size = ", len(self.gamma_spectra.keys())
        return np.vstack(gamma_repr)


    def get_fn(self, pitch):
        try:
            return self.audiofiles[(self.audiofile_set_labels[0], pitch)]
        except KeyError:
            LOGGER.warning("Pitch not found in audiofiles: {0}".format(pitch))
            self.pitches_not_available.extend([int(pitch)])
            return None

    def normalize2(self, data):
        data = data - np.min(data)
        data = data / np.sum(data)
        return data

    def normalize(self, data):
        data = data - np.min(data)
        data = data / np.max(data)
        return data

    def center_ngrams(self, ngrams, min_pitch, max_pitch):
        """Transpose each n-gram by an integer multiple of an octave, to fit
        the pitch range to the n-gram between a minimum and maximum pitch.

        Parameters
        ----------
        ngrams : ndarray
            a 2D array of type int and size: `len(pitches)` x `n`, where
            each row contains an ngram of pitches
        min_pitch : int
            minimal allowed pitch (inclusive)
        max_pitch : int
            maximal allowed pitch (exclusive)

        Returns
        -------
        ngrams : ndarray
            a 2D array of type int and size: `len(pitches)` x `n`, where
            each row contains an ngram of pitches, transposed by zero or
            more octaves to fit between `min_pitch` and `max_pitch`

        """

        center = (max_pitch + min_pitch) / 2.

        min_pitches = np.min(ngrams, 1)
        max_pitches = np.max(ngrams, 1)
        mean_pitches = np.mean(ngrams, 1)
        off = (center - mean_pitches) % 12

        transp = np.zeros((ngrams.shape[0]), np.int)

        transp_idx = np.logical_or(min_pitches < min_pitch, max_pitches > max_pitch)
        transp[transp_idx] = np.floor((center - mean_pitches[transp_idx]) / 12).astype(np.int)
        transp[transp_idx][off[transp_idx] > 6] += 1

        transp_idx = np.logical_and(min_pitches + transp * 12 < min_pitch,
                                    max_pitches + transp * 12 < max_pitch - 12)
        transp[transp_idx] += 1

        transp_idx = np.logical_and(max_pitches + transp * 12 >= max_pitch,
                                    min_pitches + transp * 12 >= min_pitch + 12)
        transp[transp_idx] -= 1
        transp = transp.reshape((-1,1))
        return ngrams + transp * 12


    def to_gammatone(self, pitches):
        import matplotlib.pyplot as plt
        LOGGER.info("extracting gamma representation of {0}".format(pitches))
        waves = []
        for pitch in pitches:
            fn = self.get_fn(pitch)
            if fn is not None:
#                 waves = np.asarray([[1]])
                audio_wave, _ = self.read_audio_audiolab(fn, mono=True)
                waves.append(audio_wave)
                

        if len(waves) == 0:
            return None
        
        wave_sum = np.zeros(max(len(wave) for wave in waves), dtype=float)
        for wave in waves:
            wave_sum[:wave.shape[0]] += wave
            
        gtg = gtgram(wave_sum, self.sample_rate, **self.spectrum_options)
        
#         gtg = np.sqrt(np.abs(gtg))
        
#         gtg = np.mean(gtg[:150,:15], axis=1)
#         plt.clf()
#         plt.plot(gtg)
#         plt.show()
        
#         gtg = np.exp(gtg) / np.sum(np.exp(gtg))
#         plt.clf()
#         plt.plot(gtg)
#         plt.show()

        return gtg



    def extract_gammatone(self, audio_fn):
        import matplotlib.pyplot as plt
        LOGGER.info("extracting gamma representation of {0}".format(audio_fn))
        audio_wave, fs = self.read_audio_audiolab(audio_fn, mono = True)

        gtg = gtgram(audio_wave, self.sample_rate, **self.spectrum_options)
        gtg = np.mean(gtg[:150,:15], axis=1)

#         plt.clf()
#         plt.plot(gtg)
#         plt.show()
        return gtg


    def repr_to_visual(self, binary_data):
        return np.fliplr(binary_data)

    def read_audio_audiolab(self, fn, mono = False):
        """
        Get the signal from an audio file.

        :param fn: filename of audio file
        :param mono: if True, mix channels into a single channel

        :returns: A tuple (signal [nframes x nchannels], sample_freq)

        """

        # TODO: make sure we always return the same
        # type of data (int/float) and range

        from scikits.audiolab import Sndfile

        audio = Sndfile(fn)
        X = audio.read_frames(audio.nframes)

        if mono and audio.channels == 2:
            X = np.mean(X,1)

        return X, audio.samplerate
