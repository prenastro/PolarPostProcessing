''' Calculates accuracies for different test : train splits for 4 different algorithms 
(Naïve Bayes, SVM, Neural Network, Random Forest). Run this script on a folder 
containing extracted content files from a given set of URLs ''' 

import numpy as np
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.model_selection import train_test_split
from numpy import array
from sklearn.naive_bayes import GaussianNB
import os
from sklearn.feature_extraction.text import CountVectorizer
import requests
from tika import parser
from tempfile import TemporaryFile
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import urllib
import io
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def loadKeywords(keyPath, ngram=False):
    if os.path.exists(keyPath):
        with open(keyPath, 'rb') as f:
            keywords_content = f.read()
    else:
        print("Keyword path is not valid!")
        return None
    if ngram:
        count_vect = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), token_pattern=r'\b\w+\b',
                                     min_df=1)
    else:
        count_vect = CountVectorizer(lowercase=True, stop_words='english')
    count_vect.fit_transform([keywords_content])
    keywords = count_vect.vocabulary_
    return count_vect


def download_file(url, i):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True, headers={'User-agent': 'Mozilla/5.0'})
    with open('/Users/prerana/Desktop/Post_Processing/200_files/' + str(i), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # f.flush() commented by recommendation from J.F.Sebastian
    return local_filename


def transformPCA(x_n):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    x_transformed = pca.fit_transform(x_n)
    x_transformed = StandardScaler().fit_transform(x_n)
    return x_transformed


def mergeAllContents():            
    all_files = os.listdir("otherstotext/")
    big_f = open("all200Files.txt", "w")
    for i in all_files:
        f=open("otherstotext/"+str(i), "r")
        big_f.write(f.read())

def closeWords(model, word, topN):
    try:
        indexes, metrics = model.cosine(word[0],n=10)
    except KeyError:
        indexes = 0
        metrics = 0
    list = model.generate_response(indexes, metrics).tolist()
    return list[:topN]


def closeWordsList(modelBin, keywords, i):
    import word2vec
    model = word2vec.load('/Users/prerana/Desktop/Post_Processing/ocean.bin')
    listTopN = []
    for word in keywords:
        for k in (closeWords(model, word, i)):
            listTopN.append(k)
    return listTopN


def addCloseCounts(listTopN, x):
    for k in range(1,np.array(listTopN).shape[1]):
        all_files = os.listdir("otherstotext/")
        big_f = open("all200Files.txt", "w")
        m = 0
        for i in all_files:
            f=io.open("otherstotext/"+str(i), "r", encoding="utf-8", errors='ignore')
            content = f.read()
            
            if content is None:
                continue
            else:
                for s in range(len(listTopN[k])):
                        str1 = str(listTopN[k][s])
                        if str1 in content:
                            x[m][k] += 1
                            m+=1
    return x


def sortingDict(x):
    import operator
    sorted_x = sorted(x.items(), key=operator.itemgetter(1))
    return sorted_x


def cosineSimilarityScore(test_url, gold_standard_url):
    import sparse as sparse
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy import sparse
    import numpy as np
    A = np.array([test_url, gold_standard_url])
    sparse_A = sparse.csr_matrix(A)
    similarities_sparse = cosine_similarity(sparse_A, dense_output=False)
    return similarities_sparse[(1)]


def accuracy(y_pred, y_test):
        accNum = 0
        for a in range(len(y_test)):
            if y_pred[a] == y_test[a]:
                accNum += 1
            else:
                if y_pred[a] in [1, 2, 3, 4, 5] and y_test[a] in [1, 2, 3, 4, 5]:
                    accNum += 1
        return accNum

def main():
        keywordPath = "/Users/prerana/Desktop/Post_Processing/features.txt"  # this should be the same keywords list/order used for training the ML Model
        count_vect = loadKeywords(keywordPath, False)
        keywords = count_vect.vocabulary_
        sorted_keywords = sortingDict(keywords)
        kList = []
        for item in sorted_keywords:
            kList.append(item)
        listTopN = closeWordsList('/Users/prerana/Desktop/Post_Processing/ocean.bin', kList, 5)

        x_train = []
        y_train = []

        with open('/Users/prerana/Desktop/Post_Processing/train.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # Relevancy score for each url taken from the csv file. Relevancy is in 2nd column.
                y_train.append(row[1])
        noneContents = []
        x_n = None
        y_n = array(y_train)

        all_files = os.listdir("otherstotext/")
        big_f = open("all200Files.txt", "w")
        for i in all_files:
            f=open("otherstotext/"+str(i), "r")
            content = f.read()
            content = unicode(content, errors='ignore')
            if content is not None:
                tempX = count_vect.transform(content.split())
                x_train.append(tempX)
                if x_n is None:
                    x_n = array([tempX.toarray().sum(axis=0)])
                else:
                    x_n = np.concatenate((x_n, [tempX.toarray().sum(axis=0)]), axis=0)
            else:
                noneContents.append(i)


        np.savetxt('/Users/prerana/Desktop/Post_Processing/x_n.txt', x_n, fmt='%d')

        x = np.loadtxt('/Users/prerana/Desktop/Post_Processing/x_n.txt', dtype=int)

        y = np.loadtxt('/Users/prerana/Desktop/Post_Processing/y_n.txt', dtype=int)
        x_with_closeWords = addCloseCounts(listTopN, x)

        mergeAllContents()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

        cv = ShuffleSplit(n_splits=5, test_size=0.2)

        clf = GaussianNB()
        scoreNB = cross_val_score(clf, x, y, cv=cv)
        clf11 = GaussianNB()

        scoreNB2 = cross_val_score(clf11, x_with_closeWords, y, cv=cv)
        clf1 = GaussianNB().fit(x_train, y_train)
        y_pred = clf1.predict(x_test)
        accNum = accuracy(y_pred, y_test)

        print("\n\nModel: Naive Bayes")
        acc = (y_test == y_pred).sum() / float(len(y_test))
        print("Test Accuracy:" + str(acc))
        acc_train = (y_train == clf1.predict(x_train)).sum() / float(len(y_train))
        print("Train Accuracy:" + str(acc_train))
        print("******************")

        from sklearn import linear_model

        clf22 = linear_model.SGDClassifier(max_iter=2000,learning_rate='optimal')
        scoreSVM = cross_val_score(clf22, x, y, cv=cv)
        clf222 = linear_model.SGDClassifier(max_iter=2000,learning_rate='optimal')

        scoreSVM2 = cross_val_score(clf222, x_with_closeWords, y, cv=cv)
        clf2 = linear_model.SGDClassifier().fit(x_train, y_train)

        y_pred2 = clf2.predict(x_test)
        accNum2 = accNum = accuracy(y_pred2, y_test)

        print("\n\nModel: SVM")
        acc = (y_test == y_pred2).sum() / float(len(y_test))
        print("Test Accuracy:" + str(acc))
        acc_train = (y_train == clf2.predict(x_train)).sum() / float(len(y_train))
        print("Train Accuracy:" + str(acc_train))
        print("******************")

        clf33 = MLPClassifier(max_iter=2000, learning_rate='adaptive')
        scoreNN = cross_val_score(clf33, x, y, cv=cv)
    
        clf333 = MLPClassifier(max_iter=2000, learning_rate='adaptive')

        scoreNN3 = cross_val_score(clf333, x_with_closeWords, y, cv=cv)

        clf3 = MLPClassifier(max_iter=2000, learning_rate='adaptive').fit(x_train, y_train)
        y_pred3 = clf3.predict(x_test)

        accNum3 = accNum = accuracy(y_pred3, y_test)

        print("\n\nModel: Neural Network")
        acc = (y_test == y_pred3).sum() / float(len(y_test))
        print("Test Accuracy:" + str(acc))
        acc_train = (y_train == clf3.predict(x_train)).sum() / float(len(y_train))
        print("Train Accuracy:" + str(acc_train))
        print("******************")

        from sklearn.ensemble import RandomForestClassifier

        clf44 = RandomForestClassifier(n_estimators=100)
        scoreRF = cross_val_score(clf44, x, y, cv=cv)
        clf444 = RandomForestClassifier(n_estimators=100)

        scoreRF4 = cross_val_score(clf444, x_with_closeWords, y, cv=cv)

        clf4 = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
        y_pred4 = clf4.predict(x_test)
        accNum4 = accNum = accuracy(y_pred4, y_test)

        print("\n\nModel: Random Forest")
        acc = (y_test == y_pred4).sum() / float(len(y_test))
        print("Test Accuracy:" + str(acc))
        acc_train = (y_train == clf4.predict(x_train)).sum() / float(len(y_train))
        print("Train Accuracy:" + str(acc_train))
        noneContents = array(noneContents)
        xOut = TemporaryFile()
        yOut = TemporaryFile()
        noneContentsOut = TemporaryFile()
        np.save(xOut, x_n)
        np.save(yOut, y_n)
        np.save(noneContentsOut, noneContents)

if __name__ == '__main__':
    main()
