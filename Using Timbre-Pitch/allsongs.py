import os
import glob
import hdf5_getters
import sys
import numpy as np
from numpy import array,mean,cov
from collections import defaultdict
from tables.exceptions import HDF5ExtError
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_score
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import cross_validation
import beat_aligned_feats as bt
from sklearn import preprocessing
import traceback
import time
from sklearn.ensemble import GradientBoostingClassifier

"""
1 - Till 1975
2 - From 1975 to 1995
3 - From 1995 to present
"""

def get_all_titles(basedir,ext='.h5') :
    
    global errorcount
    global count
    global cap
    global truecount
    
    features = []
    decade = []
    decadecount = defaultdict(int)
    timbre = None
    i = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            feature = []
            try:
                h5 = hdf5_getters.open_h5_file_read(f)
            except HDF5ExtError as e:
                errorcount += 1
                print "Unexpected error:", sys.exc_info()[0]
                print traceback.format_exc()
                continue
            year = hdf5_getters.get_year(h5)
            print i
            i+=1
            if year == 0:
                h5.close()
                continue
            label = getbin(year)
            #label = (year/10)*10
            truecount[label] += 1
            if decadecount[label] > cap:
                flag = checkforcompletion(decadecount)
                h5.close()
                if flag:
                    for dec in decadecount.keys():
                        print 'Decade : ' + str(dec) + ' Count : ' + str(decadecount[dec])
                    return features,decade
                continue
#            dec = (year/10)*10
#            if dec < 1960:
#                h5.close()
#                continue
            
#            try:
#                
#                bttimbre = bt.get_bttimbre(h5)
#                timbres = bttimbre.argmax(axis = 0) + 1   # Is a vector of timbre values sutiable for training an HMM
#                for timbre in timbres:
#                    timbredict[timbre] += 1
#                for i in range(1,13):
#                    feature.append(timbredict[i])
#            except:
#                h5.close()
#                continue
#            clustercount = {}
#            for x in range(12):
#                clustercount[x] = 0
   
            
#            try:
#                bttimbre = bt.get_bttimbre(h5)
#                btT = bttimbre.T
#                for x in btT:
#                    timbre = x.argmax(axis = 0)
#                    clustercount[timbre]+=1  
#            except:
#                h5.close()
#                continue
#            for y in range(12):
#                features.append(clustercount[y])

            try:
                btchromas = bt.get_btchromas(h5)
                for chroma in btchromas:
                    feature.append(mean(chroma))
                covmat = get_covariance(btchromas)
                feature.extend(covmat)
                bttimbre = bt.get_bttimbre(h5)
                for timbre in bttimbre:
                    feature.append(mean(timbre))
                covmat = get_covariance(bttimbre)
                feature.extend(covmat)
                
#                btT = bttimbre.T
#                for x in btT:
#                    timbre = x.argmax(axis = 0)
#                    clustercount[timbre]+=1 
#                for y in range(12):
#                    feature.append(clustercount[y])
            except:
                errorcount += 1
                h5.close()
                continue
            loudness = hdf5_getters.get_loudness(h5)
            feature.append(loudness)
            duration = hdf5_getters.get_duration(h5)
            feature.append(duration)
            features.append(feature)
            decade.append(label)
            decadecount[label] += 1
            count += 1
            h5.close()
#            title = hdf5_getters.get_title(h5)
#            segstarts = hdf5_getters.get_segments_start(h5)
#            segstarts = np.array(segstarts).flatten()
#            btstarts = hdf5_getters.get_beats_start(h5)
#            btstarts = np.array(btstarts).flatten()

    for dec in decadecount.keys():
        print 'Decade : ' + str(dec) + ' Count : ' + str(decadecount[dec])
    return features,decade

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
    
    
def get_covariance(mat):
    # Returns the lower diagonal and diagonal entries of the covariance matrix of mat
    covariance = cov(mat)
    x = []
    for i in range(0,12):
        for j in range(0,12):
            if i>= j:
                x.append(covariance[i][j])
    return x

truecount = defaultdict(int)
starttime = time.clock()
#sys.stdout = open("output.txt", "w")
cap = 20500
errorcount = 0
count = 0
#basedir = sys.argv[1]
traindir = "C:\Target"
#output1 = "C:\Users\gouthamdl\Desktop\features.npy"
#output2 = "C:\Users\gouthamdl\Desktop\decade.npy"
#traindir = "C:\Users\gouthamdl\Desktop\I"
features,decade = get_all_titles(traindir)

for dec in truecount.keys():
    print 'Decade : ' + str(dec) + ' Count : ' + str(truecount[dec])
datatime = time.clock()

features = array(features)
decade = array(decade)

features = preprocessing.scale(features)

#pca = PCA(n_components=2, whiten=True).fit(features)
#features = pca.transform(features)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, decade, test_size=0.1, random_state=0)

#clf = LinearSVC()
#clf = clf.fit(X_train,y_train)
clf = SVC(kernel='rbf', degree=3)
clf = clf.fit(X_train,y_train)
dec_pred1 = clf.predict(X_test)
accuracy = zero_one_score(y_test, dec_pred1)
print 'Accuracy with SVM : ' + str(accuracy)

n_neighbors = 20
clf2 = neighbors.KNeighborsClassifier(n_neighbors)
clf2 = clf2.fit(X_train,y_train)
dec_pred2 = clf2.predict(X_test)
accuracy = zero_one_score(y_test, dec_pred2)
print 'Accuracy with Nearest Neighbors : ' + str(accuracy)

#clf3 = GaussianNB().fit(features,decade)
#dec_pred3 = clf3.predict(ldtest)
#accuracy = zero_one_score(dec_test, dec_pred3)
#print 'Accuracy with Naive Bayes : ' + str(accuracy)

print 'Gradient Boosting'
clf2 = GradientBoostingClassifier().fit(X_train,y_train)
scores = cross_validation.cross_val_score(clf2, features,decade, cv=5)
print 'Accuracy with Gradient Boosting : ' + str(scores.mean())

print 'Logistic Regression'
clf2 = LogisticRegression().fit(X_train,y_train)
scores = cross_validation.cross_val_score(clf2, features,decade, cv=5)
print 'Accuracy with Logistic Regression : ' + str(scores.mean())
#dec_pred2 = clf2.predict(X_test)
#accuracy = zero_one_score(y_test, dec_pred2)
#print 'Accuracy with Logistic Regression : ' + str(accuracy)

#dec_pred2 = clf2.predict(X_test)
#accuracy = zero_one_score(y_test, dec_pred2)
#print 'Accuracy with Gradient Boosting : ' + str(accuracy)

#for x,y in zip(y_test,dec_pred2):
#    print 'Actual Decade : ' + str(x) + ' Predicted Decade : ' + str(y)

#correct = defaultdict(int)
#incorrect = defaultdict(int)
#for i in range(len(dec_pred2)):
#    corr = y_test[i]
#    incorr = dec_pred2[i]
#    if corr == incorr:
#        correct[corr] += 1
#    else:
#        incorrect[corr] += 1
#
#for dec in correct.keys():
#    print 'Decade : ' + str(dec) + ' Correct : ' + str(correct[dec]) + ' InCorrect : ' + str(incorrect[dec])
    
endtime = time.clock()
print 'Time taken for data processing : ' + str((datatime - starttime)/60.0) + ' mins'
print 'Total time taken : ' + str((endtime - starttime)/60.0) + ' mins' 

#print 'Random Forest'
#rf = RandomForestClassifier()
#dec_pred4 = rf.fit(X_train,y_train)
#accuracy = zero_one_score(y_test, dec_pred4)
#print 'Accuracy with Random Forest : ' + str(accuracy)
