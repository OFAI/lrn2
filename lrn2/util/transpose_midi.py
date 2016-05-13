'''
Created on Dec 22, 2015

@author: Stefan Lattner
'''
import re
import os
import glob
import argparse
import numpy as np

def transpose_to_all_keys(input, shift, rec=False, pat="*"):
    if not pat:
        pat = "*"

    rng = range(-6, 6)
    if shift != None:
        rng = (int(shift),)
        
    if not os.path.isdir(input):
        fn = os.path.basename(input)
        root = os.path.dirname(input)
        for trans in rng:
            print "transposing {0} by {1}".format(input, trans)
            out_fn = os.path.join(root, fn.split(".")[0] + "_" + str(trans) + ".mid")
            transpose(input, trans, out_fn)
    else:
        for root, _, _ in os.walk(input):
            for fn in glob.glob1(root, pat):
                f = os.path.join(root, fn)
                if not os.path.isdir(f):
                    for trans in rng:
                        print "transposing {0} by {1}".format(f, trans)
                        out_fn = os.path.join(root, fn.split(".")[0] + "_" + str(trans) + ".mid")
                        transpose(f, trans, out_fn)
            if not rec:
                break
        
def transpose(fn, shift, out_fn):
    tmp_file = "/tmp/mf2t.tmp"
    os.system("mf2t {0} > {1}".format(fn, tmp_file))
    
    with open(tmp_file, 'r+') as f:
        content = f.read()
#         print content
        indices = [(m.start()+2, m.end()) for m in re.finditer('n=[0-9]*', content)]
        pitches = []
        for start, end in indices:
            pitches.append(content[start:end])
             
        pitches = np.cast['int16'](np.asarray(pitches))
                
        lens_before = np.asarray([len(str(n)) for n in pitches])
        pitches += shift
        pitches[np.where(pitches < 0)] += 12
        pitches[np.where(pitches > 128)] -= 12
          
        assert np.all(pitches > -1) and np.all(pitches < 129)
        lens_after = np.asarray([len(str(n)) for n in pitches])
         
        lens_corr = lens_after - lens_before
        corr_eff = np.cumsum(lens_corr) 
          
        for pitch, (start, end), shift in zip(pitches, indices, corr_eff):
            content = content[:start+shift] + str(pitch) + content[end+shift:]
              
        f.seek(0)
        f.truncate()
        f.write(content)
          
    os.system("t2mf {0} > {1}".format(tmp_file, out_fn))
        
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Transpose a single midi file or a set of midis (i.e. all midis in a given folder) into all keys.")

    parser.add_argument("source", help = ("file or folder containing source files"))

    parser.add_argument("-t", help = "specify a single offset (default is +-6)", default = None)
    
    parser.add_argument("-r", action = "store_true", default = False,
                       help = "recursively follow subfolders")

    parser.add_argument("-p", help = "specify a filename pattern (enclosed by quotes)")

    args = parser.parse_args()

    transpose_to_all_keys(args.source, args.t, args.r, args.p);