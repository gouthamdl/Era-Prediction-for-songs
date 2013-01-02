'''
Created on Nov 8, 2012

@author: gouthamdl
'''
"""
For getting statistics about various features in a song
"""

import os
import glob
import hdf5_getters
import sys
import numpy as np
from numpy import array,mean,cov,savetxt
from collections import defaultdict
from tables.exceptions import HDF5ExtError
import traceback

def get_all_titles(basedir,ext='.h5') :
    
    global cap
    features = []    
    decade = []
    decadecount = defaultdict(int)
    i = 0
    
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            try:
                h5 = hdf5_getters.open_h5_file_read(f)
            except HDF5ExtError as e:
                print "Unexpected error:", sys.exc_info()[0]
                print traceback.format_exc()
                h5.close()
                continue
            year = hdf5_getters.get_year(h5)
            if year == 0:
                h5.close()
                continue
            print i
            #label = getbin(year)
            label = (year/10)*10
            if decadecount[label] > cap:
                flag = checkforcompletion(decadecount)
                h5.close()
                if flag:
                    for dec in decadecount.keys():
                        print 'Decade : ' + str(dec) + ' Count : ' + str(decadecount[dec])
                    return features
                continue
            feature = []
            decade.append(label)
            
            ldness = hdf5_getters.get_loudness(h5)
            feature.append(ldness)
            
            duration = hdf5_getters.get_duration(h5)
            feature.append(duration)
            
#            energy = hdf5_getters.get_energy(h5)
#            feature.append(energy)
#            
#            key = hdf5_getters.get_key(h5)
#            feature.append(key)
#            
#            mode = hdf5_getters.get_mode(h5)
#            feature.append(mode)
#            
#            tempo = hdf5_getters.get_tempo(h5)
#            feature.append(tempo)
#            
#            timesig = hdf5_getters.get_time_signature(h5)
#            feature.append(timesig)
#            
#            songname = hdf5_getters.get_title(h5)
#            songlen = len(songname)
#            feature.append(songlen)
#            
#            artistname = hdf5_getters.get_artist_name(h5)
#            artistlen = len(artistname)
#            feature.append(artistlen)
            feature.append(label)

            features.append(feature)
            
            decadecount[label] += 1
            h5.close()
            i+=1
    for dec in decadecount.keys():
        print str(dec) + ' : ' + str(decadecount[dec])
    return features

def checkforcompletion(decadecount):
    
    global cap
    flag = True
    for dec in decadecount.keys():
        if decadecount[dec] < cap:
            flag = False
    return flag

def getbin(year):
    
    if year < 1975:
        return 1
    elif year < 1995:
        return 2
    else:
        return 3
    
cap = 50000
traindir = "C:\Target"
loudness = get_all_titles(traindir)

savetxt('features.txt',loudness)
