import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
import pandas as pd
import numpy as np
import csv

train = pd.read_csv(os.path.join(
	os.path.dirname(__file__), '..', '..', 'fixtures', 'labeledTrainData.tsv'),
	header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
test = pd.read_csv(os.path.join(
	os.path.dirname(__file__), '..', '..', 'fixtures', 'testData.tsv'),
	header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
y = train["sentiment"]
print "Cleaning and parsing movie reviews...\n"
traindata = []
for i in xrange(0, len(train["review"])):
    traindata.append(" ".join(
        KaggleWord2VecUtility.review_to_wordlist(train["review"][i], False)))
testdata = []
for i in xrange(0, len(test["review"])):
    testdata.append(" ".join(
        KaggleWord2VecUtility.review_to_wordlist(test["review"][i], False)))
print 'vectorizing... ',
tfv = TfidfVectorizer(max_features=50000,
                      strip_accents='unicode', analyzer='word',
                      token_pattern=r'\w{1,}', ngram_range=(1, 2),
                      use_idf=1, smooth_idf=1, sublinear_tf=1,
                      stop_words=None)
# tfv = CountVectorizer(strip_accents='unicode',
#                       stop_words=None, token_pattern=r"\w{1,}",
#                       ngram_range=(1, 1), analyzer='word',
#                       min_df=3, max_features=None)
X_all = traindata + testdata
lentrain = len(traindata)

print "fitting pipeline... ",
tfv.fit(X_all)
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

# model = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
#                            C=1, fit_intercept=True, intercept_scaling=1.0,
#                            class_weight=None, random_state=None)
model = LogisticRegression()
# model = SGDClassifier(loss='log', penalty='l2')
# model = MultinomialNB()
# model = BernoulliNB()
# model = GradientBoostingClassifier()

print "20 Fold CV Score: ", np.mean(
    cross_validation.cross_val_score(model, X, y, cv=3, scoring='roc_auc'))
print "20 Fold CV Score: ", np.mean(
    cross_validation.cross_val_score(
      model, X, y, cv=3, scoring='accuracy'))

print "Retrain on all training data, predicting test labels...\n"
model.fit(X, y)
result = model.predict_proba(X_test)[:, 1]
# result = model.predict(X_test)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

# Use pandas to write the comma-separated output file
output.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'fixtures', 'Bag_of_Words_model.csv'), index=False, quoting=3)
print "Wrote results to Bag_of_Words_model.csv"