
from sklearn import metrics,cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_score
from sklearn.cluster import KMeans
from numpy import array
import sys
import os
import glob
import hdf5_getters
from collections import defaultdict
import beat_aligned_feats as bt

def getclusters(basedir,ext='.h5') :
    print 'inside clusters'
    features = []
    cfeatures = [] 
    decadecount = defaultdict(int)
    deccount = defaultdict(int)
    i=0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            h5 = hdf5_getters.open_h5_file_read(f)
            year = hdf5_getters.get_year(h5)
            if year == 0:
                h5.close()
                continue
            bins = getbin(year)
#            decade.append(bin)
#            dec = (year/10)*10
#            decadecount[dec] += 1
#            if dec < 1960:
#                h5.close()
#                continue
             
            deccount[bin] += 1
            if decadecount[bins] > cap:
                flag = checkforcompletion(decadecount)
                h5.close()
                if flag:
                    for dec in decadecount.keys():
                        print 'Dec : ' + str(dec) + ' Count : ' + str(decadecount[dec])
                    return features,cfeatures
                continue
            i += 1
            print i
            try:
                bttimbre = bt.get_bttimbre(h5)
                btT = bttimbre.T
                for x in btT:
                    features.append(x)
                decadecount[bins] += 1
                btchroma = bt.get_btchromas(h5)
                btc = btchroma.T
                for x in btc:
                    cfeatures.append(x)
                decadecount[bins] += 1
            except:
                h5.close()
                continue
            h5.close()
    for dec in deccount.keys():
        print 'Dec : ' + str(dec) + ' Count : ' + str(decadecount[dec])
    features = array(features)
    cfeatures = array(cfeatures)
    return features,cfeatures

def buildfeatures(basedir,cluster,ccluster,ext='.h5'):
    
    global cap
    i = 0
    features = []
    decade = []
    decadecount = defaultdict(int)
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            h5 = hdf5_getters.open_h5_file_read(f)
            year = hdf5_getters.get_year(h5)
            if year == 0:
                h5.close()
                continue
            bins = getbin(year)
#            dec = (year/10)*10
#            if dec < 1960:
#                h5.close()
#                continue
            if decadecount[bins] > cap:
                flag = checkforcompletion(decadecount)
                h5.close()
                if flag:
                    return features,decade
                continue
            i += 1
            print i
            
            clustercount = {}
            cclustercount = {}
            
            for x in range(50):
                clustercount[x] = 0
                cclustercount[x] = 0 
            
            feature = []
            cfeature =[]
            
            try:
                bttimbre = bt.get_bttimbre(h5)
                btT = bttimbre.T
                for x in btT:
                    label = kmeans.predict(x)
                    clustercount[label[0]] += 1
                for cl in clustercount.keys():
                    feature.append(clustercount[cl])
             
                
                btchroma = bt.get_btchromas(h5)
                btc = btchroma.T
                for y in btc :
                    clabel = ckmeans.predict(y)
                    cclustercount[clabel[0]] +=1
                for cl in cclustercount.keys():
                    feature.append(cclustercount[cl])
                
                features.append(feature)
                
                decade.append(bins)
                decadecount[bins] += 1
            
            except:
                h5.close()
                continue
            h5.close()
            
    print len(features)
    print len(decade)
    return features,decade

def getbin(year):
    
    if year < 1970:
        return 1
    elif year < 1990:
        return 2
    else:
        return 3
    

def checkforcompletion(decadecount):
    
    global cap
    flag = True
#    for dec in range(1960,2020,10):
#        if decadecount[dec] < cap:
#            flag = False
#    return flag
    if decadecount[1]<cap or decadecount[2]<cap or decadecount[3]<cap :
        flag = False 
    return flag
cap = 40
#sys.stdout = open("output.txt", "w")
traindir = "C:\Users\gouthamdl\Desktop\data"
segments,csegments = getclusters(traindir)
#n_samples, n_features = segments.shape
#print segments
#print 'csegments'
#print csegments
print 'Performing Clustering'
estimator = KMeans(init='k-means++', n_clusters=50, n_init=1)
cestimator = KMeans(init='k-means++', n_clusters=50, n_init=1)
kmeans = estimator.fit(segments)
ckmeans = estimator.fit(csegments)
features,labels = buildfeatures(traindir,kmeans,ckmeans,ext='.h5')
features = array(features)
labels = array(labels)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=0)

print 'Performing Classification'
clf2 = LogisticRegression().fit(X_train,y_train)
dec_pred2 = clf2.predict(X_test)
accuracy = zero_one_score(y_test, dec_pred2)
print 'Accuracy with Logistic Regression : ' + str(accuracy)

#for x,y in zip(y_test,dec_pred2):
#    print 'Actual Decade : ' + str(x) + ' Predicted Decade : ' + str(y)