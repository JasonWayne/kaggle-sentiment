from gensim.models.word2vec import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
import os
import csv
import constants
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import time
from datetime import datetime
import numpy as np
import pickle
from sklearn import cross_validation

model = Word2Vec.load_word2vec_format(constants.GOOGLE_WORD2VEC, binary=True)

train = pd.read_csv(os.path.join(
	os.path.dirname(__file__), '..', 'fixtures', 'labeledTrainData.tsv'),
	header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
test = pd.read_csv(os.path.join(
	os.path.dirname(__file__), '..', 'fixtures', 'testData.tsv'),
	header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
y = train["sentiment"]
print "Cleaning and parsing movie reviews...\n"
traindata = []
for i in xrange(0, len(train["review"])):
    traindata.append(" ".join(
        KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))
testdata = []
for i in xrange(0, len(test["review"])):
    testdata.append(" ".join(
        KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

X_all = traindata + testdata
lentrain = len(traindata)

print "fitting pipeline... ",
vectorizer = CountVectorizer(min_df=4)
vectorizer.fit(X_all)

start = time.time()

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
words = vectorizer.get_feature_names()
num_clusters = len(vectorizer.get_feature_names()) / 5

print "Get Word Vectors --> " + str(datetime.now())
word_vectors = []
for word in words:
	try:
		word_vectors.append(model[word])
		# word_vectors.append(model[word].reshape((1, 300)))
	except KeyError:
		continue
# word_vectors = np.concatenate(word_vectors)
# word_vectors = np.array(word_vectors, dtype='float')

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters=num_clusters)

idx = kmeans_clustering.fit_predict(word_vectors)

pickle.dump(kmeans_clustering, file('kmeans', 'w'))

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."


# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip(words, idx))


# For the first 10 clusters
for cluster in xrange(0, 10):
    # Print the cluster number
    print "\nCluster %d" % cluster
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if(word_centroid_map.values()[i] == cluster):
            words.append(word_centroid_map.keys()[i])
    print words


def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros((train["review"].size, num_clusters), \
    dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in traindata:
	review = review.split()
	train_centroids[counter] = create_bag_of_centroids(
		review, word_centroid_map)
	counter += 1

# Repeat for test reviews 
test_centroids = np.zeros((
	test["review"].size, num_clusters), dtype="float32" )

counter = 0
for review in testdata:
	review = review.split()
	test_centroids[counter] = create_bag_of_centroids(review, \
        word_centroid_map )
	counter += 1

# lr = SGDClassifier(loss='log')
lr = LogisticRegression()

print "20 Fold CV Score: ", np.mean(
    cross_validation.cross_val_score(lr, train_centroids, y, cv=20, scoring='roc_auc'))

print "20 Fold CV Score: ", np.mean(
    cross_validation.cross_val_score(
    	lr, train_centroids, y, cv=20, scoring='accuracy'))

print "Retrain on all training data, predicting test labels...\n"
lr.fit(train_centroids, y)
result = lr.predict_proba(test_centroids)[:, 1]
# result = model.predict(X_test)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

# Use pandas to write the comma-separated output file
output.to_csv('word2vec_model.csv', index=False, quoting=3)
print "Wrote results to word2vec_model.csv"
