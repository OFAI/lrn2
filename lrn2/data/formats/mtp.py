import csv
import numpy as np

class MTPInterface(object):
    def get_pitch_from_raw_data(self, data):
        return data[:,5]

def load_mtp_files(filenames):
    for fn in filenames:
        for v in load_mtp_file(fn):
            yield v, None


def load_mtp_file(fn):
    """
    Read music data from a csv file, as used by Marcus T. Pearce. The
    file is supposed to contain composition_id and dataset_id fields,
    and the rows are grouped along the key (dataset_id,
    composition_id).

    Parameters
    ----------

    fn : str
        filename of the CSV file to parse

    Returns
    -------

    dict
        a dictionary where the key is the piece name (consisting of
        dataset_id and composition_id), and the value is an ndarray
        with the data for that piece

    """

    c = csv.reader(open(fn,'r'))
    header = [x.strip() for x in c.next()]
    comp_id_field = header.index('composition_id')
    dataset_id_field = header.index('dataset_id')

    prev_key = None
    data = []
    for l in c:
        l_ints = [int(x) if x != '\N' else np.nan for x in l]
        key = (l_ints[dataset_id_field], l_ints[comp_id_field])
        if key == prev_key or prev_key is None:
            data.append(l_ints)
        else:
            #key_string = 'dataset_{0:02d}_composition_{1:04d}'.format(*key)
            adata = np.array(data)
            data = []
            yield adata
        prev_key = key

