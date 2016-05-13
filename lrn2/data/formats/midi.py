#!/usr/bin/env python

import logging
import numpy as np

import lrn2.data.formats.midi_utils.midi as midi
from lrn2.data.formats.midi_utils.midi import TimeSigEvent

LOGGER = logging.getLogger(__name__)

class MIDIInterface(object):
    """A class that defines methods to extract musical information from
    the data returned by the `load_midi` function. This class can be
    used in conjunction with ViewPoint mixin classes to create
    viewpoint information from MIDI files.

    """

    def get_pitch_from_raw_data(self, data):
        """
        Return a sequence of pitches as midi note values
        """
        return data['pitch']

    def get_onset_from_raw_data(self, data):
        """
        Return a sequence of onsets in quarter notes
        """
        return data['onset']

    def get_duration_from_raw_data(self, data):
        """
        Return a sequence of durations in quarter notes
        """
        return data['duration']

    def get_velocity_from_raw_data(self, data):
        """
        Return a sequence of midi velocity values
        """
        return data['velocity']

    def get_channel_from_raw_data(self, data):
        """
        Return a sequence of midi channel values
        """
        return data['channel']

    def get_track_from_raw_data(self, data):
        """
        Return a sequence of midi track numbers, which identify the
        (zero-based) track index in which the notes occur.

        """
        return data['track']
    
    def get_time_signature_from_raw_data(self, data):
        return data['timesig'][0]['num'], data['timesig'][0]['denom']


def load_midi_files(filenames, ignore_channels = [10]):
    for fn, label in filenames:
        LOGGER.debug("loading midi {0}".format(fn))
        try:
            yield (load_midi(fn, ignore_channels = ignore_channels), label)
        except Exception as e:
            LOGGER.error(e)
            yield [[], "dummy"]

def load_midi(fn, flatten = True, ignore_channels = [10], transpose = 0):
    """
    Read a MIDI file and return pitch, onset, duration, velocity, MIDI
    channel, and MIDI track number. Onset and duration are specified
    in quarter beats. The MIDI notes in each track of the MIDI file
    are concatenated. If `flatten' is True, the notes are ordered by
    onset, rather than by MIDI track number.

    Parameters
    ----------
    fn : str
      MIDI filename

    flatten : bool (default: True)

      if True, order
      
    Returns
    -------
    ndarray
        A numpy array containing the note information

    """

    # create a MidiFile object from an existing midi file
    m = midi.MidiFile(fn)


    if not len(m.tracks) == 1:
        LOGGER.warning("More than one track found in file {0}:{1}".format(fn, len(m.tracks)))
                       
    if not len(m.tracks[0].get_events(TimeSigEvent)) <= 1:
        LOGGER.warning("More than one time signature found in track 0: {0}".format(len(m.tracks[0].get_events(TimeSigEvent))))
        
    #convert the object to type 0 (by merging all tracks into a single track)
    if flatten:
        m = midi.convert_midi_to_type_0(m)

    div = float(m.header.time_division)

    note_information = []

    for track_nr in range(len(m.tracks)):
        try:
            num = m.tracks[track_nr].get_events(TimeSigEvent)[0].num
            den = 2**m.tracks[track_nr].get_events(TimeSigEvent)[0].den
        except Exception as e:
            LOGGER.warning("No time signature event found - assuming 4/4.")
            num = 4
            den = 4
            
        note_inf_track = np.array([(n.note, n.onset/div, n.duration/div, n.velocity, n.channel, track_nr, {"num": num, "denom": den})
                                    for n in m.tracks[track_nr].notes if n.channel not in ignore_channels],
                                  dtype = [('pitch', np.int),
                                           ('onset', np.float),
                                           ('duration', np.float),
                                           ('velocity', np.int),
                                           ('channel', np.int),
                                           ('track', np.int),
                                           ('timesig', dict),])
        note_inf_track['pitch'] += transpose
        note_information.append(note_inf_track)

    note_information = np.hstack(note_information)

    if flatten:
        note_information = note_information[np.argsort(note_information['onset'])]

    return note_information

def midi_key_signature(fn):
    """
    Reads the key signature information of a MIDI file

    Parameters
    ----------
    fn: string
        Path to the MIDI file

    Returns
    -------
    KEY: String
        The key of the midi file (a string)
    """

    # Read midi file metadata
    W = midi.MidiFile(fn)

    if not W.tracks[0].get_events(midi.KeySigEvent):
        raise Exception("No key signature found in MIDI file " + fn)

    Wkey = W.tracks[0].get_events(midi.KeySigEvent)[0].key

    Wscale = W.tracks[0].get_events(midi.KeySigEvent)[0].scale

    # Convert from unsigned to signed integer
    if Wkey > 127:
        kysgn = Wkey - 256
    else:
        kysgn = Wkey

    if kysgn > 7:
        kysgn = np.mod(Wkey,7)

    # Dictionary of keys
    KEYS = {0:["C","a"],1:["G","e"],2:["D","b"],3:["A","f#"],4:["E","c#"],5:["B","g#"],6:["F#","d#"],7:["C#","a#"],-1:["F","d"],-2:["A#","g"],-3:["D#","c"],-4:["G#","f"],-5:["C#","a#"],-6:["F#","d#"]}

    return KEYS[kysgn][Wscale]


def midifile_key_signature_new(fn):
    """
    Reads the key signature information of a MIDI file

    Parameters
    ----------
    fn: string
        name of the MIDI file

    Returns
    -------
    key: string
        the key of the midi file (a string)

    """

    # Read midi file and convert to type0 (so everything is in one track)
    m = midi.convert_midi_to_type_0(midi.MidiFile(fn))
    return midi_key_signature_new(m)

def midi_key_signature_new(m):
    """
    Reads the key signature information of a MidiFile object

    Parameters
    ----------
    m: `MidiFile`
        a `MidiFile` object

    Returns
    -------
    key: string
        the key of the midi file (a string)

    """

    if not isinstance(m, midi.MidiFile) and isinstance(m, str):
        m = midi.convert_midi_to_type_0(midi.MidiFile(m))

    key_sigs = m.get_track().get_events(midi.KeySigEvent)
    if len(key_sigs) == 0:
        # raise Exception("No key signature found in MIDI file " + fn)
        LOGGER.warning("No key signature found in MIDI file.")
        return None

    m_key = key_sigs[0].key
    m_scale = key_sigs[0].scale

    # Convert from unsigned to signed integer
    if m_key > 127:
        key_sign = m_key - 256
    else:
        key_sign = m_key

    if key_sign > 7:
        key_sign = np.mod(m_key, 7)

    # Dictionary of keys
    keys = {0: ["C","a"], 1: ["G","e"], 2: ["D","b"], 3: ["A","f#"], 4: ["E","c#"],
            5: ["B","g#"], 6: ["F#","d#"], 7: ["C#","a#"], -1: ["F","d"], -2: ["A#","g"],
            -3: ["D#","c"], -4: ["G#","f"], -5: ["C#","a#"], -6: ["F#","d#"]}

    return keys[key_sign][m_scale]

if __name__ == "__main__":
    pass
