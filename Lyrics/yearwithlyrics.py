'''
Created on Nov 2, 2012

@author: CT
'''

import sys
from collections import defaultdict
from numpy import array
from sklearn.metrics import zero_one_score
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation, neighbors
import sys
from sklearn.linear_model.logistic import LogisticRegression

sys.stdout = open("output.txt", "w")
print "Start"

f = open("tracks_per_year.txt")
yeardict = defaultdict(int)
deccount = defaultdict(int)
for line in f:
    words = line.split('<SEP>')
    year = int(words[0])
    artistid = words[1]
    decade = (year/10)*10
    deccount[decade] += 1
    yeardict[artistid] = decade
f.close()
    
wordmap={}
lyricsfile = open('mxm_dataset_train.txt','r')
features = []
labels = []
decadecount = defaultdict(int)
for info in lyricsfile:
    if info.startswith('%'):
        # This the line with word mappings
        # Remove the % at the start and split the entire string on comma
        info = info.replace('%','').split(',')
        for i,word in enumerate(info):
            wordmap[i+1]=word
    
    else:
        if not info.startswith('#'):
            info = info.split(',')
            msdID = info[0]
            mmatchID = info[1]
            
            decade = yeardict[msdID]    # Find the decade for this track id.
            if decade < 1960 or decade > 2000:
                continue
            if decade == 0:             # If decade is 0, it means we dont have the track's year
                continue
            decadecount[decade] += 1
            # Keeps track of the number of songs seen for each decade
            count = decadecount[decade]
            if count > 3100:
                continue
            # Iterate over the wordcounts now
            
            numofwords = len(wordmap)   # Total Number of words in the dataset
            wordlist = [0]*numofwords   # Corresponds to each sample
            for wordinfo in info[2:]:
                wordinfo = wordinfo.split(':')      # The wordcount is of the form word:count
                wordnum = int(wordinfo[0])          # wordnum is the number that maps to a word
                if wordlist[wordnum-1] == 1:        # Checking only for presence or absence of the word
                    continue
                wordcount = int(wordinfo[1])
                wordlist[wordnum-1] = 1
            labels.append(decade)
            features.append(wordlist)
                
#            if count > 2800:
#                test_labels.append(decade)
#                test_samples.append(wordlist)
#            else:
#                train_labels.append(decade)
#                train_samples.append(wordlist)

print 'Finished Iteration'
features = array(features)
labels = array(labels)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=0)
print 'Finished train-test split'

clf = LinearSVC()
clf = clf.fit(X_train,y_train)
dec_pred1 = clf.predict(X_test)
#decmap = defaultdict(int)
#for x,y in zip(test_labels,dec_pred1):
#    decmap[(x,y)] += 1
#for x,y in decmap.keys():
#    print 'Actual Decade : ' + str(x) + ' Decade Predicted ' + str(y) + ' Count : ' + str(decmap[(x,y)])

accuracy = zero_one_score(y_test, dec_pred1)
print 'Accuracy with SVM : ' + str(accuracy)

clf3 = MultinomialNB().fit(X_train,y_train)
dec_pred3 = clf3.predict(X_test)
accuracy = zero_one_score(y_test, dec_pred1)
print 'Accuracy with Naive Bayes : ' + str(accuracy)

n_neighbors = 15
clf2 = neighbors.KNeighborsClassifier(n_neighbors)
clf2 = clf2.fit(X_train,y_train)
dec_pred2 = clf2.predict(X_test)
accuracy = zero_one_score(y_test, dec_pred2)
print 'Accuracy with Nearest Neighbors : ' + str(accuracy)

clf2 = LogisticRegression().fit(X_train,y_train)
dec_pred2 = clf2.predict(X_test)
accuracy = zero_one_score(y_test, dec_pred2)
print 'Accuracy with Logistic Regression : ' + str(accuracy)


lyricsfile.close()    


