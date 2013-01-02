'''
Created on Nov 24, 2012

@author: gouthamdl
'''

"""
Extracts the relevant hdf5 files and copies them to a different location
"""

import os
import glob
import hdf5_getters
from collections import defaultdict
from shutil import copy
import time

"""
1 - Till 1975
2 - From 1975 to 1995
3 - From 1995 to present
"""

def get_all_titles(basedir,targetdir,ext='.h5') :
    
    global cap
    
    decadecount = defaultdict(int)
    i = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            h5 = hdf5_getters.open_h5_file_read(f)
            year = hdf5_getters.get_year(h5)
            print i
            i+=1
            if year == 0:
                h5.close()
                continue
            label = getbin(year)
            if decadecount[label] > cap:
                flag = checkforcompletion(decadecount)
                h5.close()
                if flag: # All the bins have exceeded their count. We can proceed with the training
                    return decadecount
                continue
            
            # Copy files
            copy(f,targetdir)
            
            decadecount[label] += 1
            h5.close()
    
    return decadecount

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

starttime = time.clock()
cap = 1800
traindir = "C:\MSD\L"
targetdir = "C:\Target\I"
decadecount = get_all_titles(traindir,targetdir)
for dec in decadecount.keys():
        print 'Decade : ' + str(dec) + ' Count : ' + str(decadecount[dec])

endtime = time.clock()
print 'Total time taken : ' + str((endtime - starttime)/60.0) + ' mins' 