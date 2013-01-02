'''
Created on Oct 4, 2012

@author: gouthamdl
'''

from collections import defaultdict

hard_rock = ['Led Zepellin','Thin Lizzy','Deep Purple']
afile = open('unique_artists.txt','r')
artistID = None
for artistInfo in afile:
    artistInfo = artistInfo.split('<SEP>')
    artist = artistInfo[-1].strip()
    artistID = artistInfo[-2]
    if artist == 'Darkthrone':
        print artistInfo
afile.close()
    
sabbathtracks=[]
songfile = open('unique_tracks.txt','r')
for songinfo in songfile:
    songinfo = songinfo.split('<SEP>')
    artist = songinfo[-2].strip()
    songname = songinfo[-1]
    songid = songinfo[1]
    trackid = songinfo[0]
    
    if artist == 'Muddy Waters':
        sabbathtracks.append(trackid)
        print songname + ' ' + str(songid)  +' ' + str(trackid)
    
songfile.close()

wordmap={}
sabbath = defaultdict(int)
lyricsfile = open('mxm_dataset_train.txt','r')
for info in lyricsfile:
    if info.startswith('%'):
        # Remove the % at the start and split the entire string on comma
        info = info.replace('%','').split(',')
        for i,word in enumerate(info):
            wordmap[i+1]=word
    
    else:
        if not info.startswith('#'):
            info = info.split(',')
            msdID = info[0]
            mmatchID = info[1]
            
            if msdID in sabbathtracks:
                for wordinfo in info[2:]:
                    wordinfo = wordinfo.split(':')
                    word = wordmap[int(wordinfo[0])]
                    sabbath[word] += int(wordinfo[1])
lyricsfile.close()                    

def extractwords(songinfo,wordmap):
    info = songinfo.split(',')
    msdID = info[0]
    mmatchID = info[1]
    wordinfo = info[2:]
    wordcmap = defaultdict(int)
    for word in wordinfo:
        winfo = word.split(':')
        word = wordmap[int(winfo[0])]
        wordcmap[word] += int(winfo[1])
    for word in wordcmap.keys():
        print word + ' ' + str(wordcmap[word])



info = 'TRMNWOP128F426015C,1723041,2:13,5:3,6:9,8:1,10:2,13:9,17:2,18:3,19:1,26:1,39:1,48:1,54:1,69:1,71:1,85:1,87:1,89:1,97:1,139:1,141:1,162:1,165:3,182:1,196:1,206:1,214:1,224:3,244:1,252:1,294:4,302:1,305:1,352:1,389:1,426:1,474:1,521:1,529:1,546:1,619:1,641:1,714:3,776:1,812:2,848:1,1040:1,1107:2,1265:1,1283:1,1301:1,1483:1,1580:1,1642:1,1850:1,1938:1,2264:2,2347:1,2444:2,2672:1,2905:1,3128:1,3257:1,3401:1,3911:5'
extractwords(info,wordmap)
#for word in sabbath.keys():
#    print word + ' ' + str(sabbath[word])
##print len(sabbath)
            

            
            
    
