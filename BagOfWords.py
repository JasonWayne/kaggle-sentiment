import pandas as pd
import csv
from nltk.corpus import stopwords
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from BagOfWordsUtility import BagOfWordsUtility
import numpy as np
import pickle

start = time.clock()
train = pd.read_csv("original/labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=csv.QUOTE_NONE)

stops = set(stopwords.words("english"))
num_reviews = train['review'].size
clean_train_reviews = []
for i in xrange(num_reviews):
    if (i+1) % 1000 == 0:
        print "Review %d of %d" % (i+1, num_reviews) 
    clean_train_reviews.append(BagOfWordsUtility.review_to_words(train['review'][i], stops))

vectorizer = CountVectorizer(analyzer="word", 
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()

dist = np.sum(train_data_features, axis=0)
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train['sentiment'])
pickle.dump(forest, file('output/model_dump', 'w'))

test = pd.read_csv("original/testData.tsv", header=0, \
                   delimiter="\t", quoting=csv.QUOTE_NONE)
num_reviews = test["review"].size
clean_test_reviews = []
for i in xrange(num_reviews):
    if (i + 1) % 1000 == 0:
        print "Review %d of %d" % (i+1, num_reviews)
    clean_test_reviews.append(BagOfWordsUtility.review_to_words(test['review'][i], stops))
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("output/Bag_of_Words_model.csv", index=False, quoting=csv.QUOTE_NONE)
end = time.clock()
print end-start