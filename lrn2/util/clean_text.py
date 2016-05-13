import os
import re
import sys

rootDir = "."
min_chars_par = 250
out_folder = "cleaned"

def clean():
    pat = re.compile("[^\n]{%s,}" % min_chars_par)
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
	    print "processing file {0}".format(fname)
            with open(fname, 'r') as f:
		content = f.read().decode('iso-8859-15').encode('utf-8')
		paragraphs = pat.findall(content)
		with open(os.path.join(out_folder, fname), 'w') as fw:
		    for p in paragraphs:
			re.sub(' +',' ','The     quick brown    fox')
		    	fw.write(re.sub(' +',' ',p) + "\n\n")

if __name__ == "__main__":
	clean()
