import gensim
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import constants
import csv
from bs4 import BeautifulSoup
from datetime import datetime
import os
import string
import pickle
import random
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import cross_validation


LabeledSentence = gensim.models.doc2vec.LabeledSentence

if not os.path.exists('x_train'):
	train = pd.read_csv(
		constants.TRAIN_DATA, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

	test = pd.read_csv(
		constants.TEST_DATA, header=0, delimiter='\t', quoting=csv.QUOTE_NONE)

	unlabeled = pd.read_csv(
		constants.UNLABELED_DATA, header=0, delimiter='\t', quoting=csv.QUOTE_NONE)

	y = train["sentiment"]
	train_data = train["review"]
	unlabeled_data = unlabeled["review"]
	test_data = test["review"]


	def cleanText(corpus):
		punctuation = set(string.punctuation)
		corpus = [z.lower().replace('\n', '') for z in corpus]
		# remove html tags
		print 'remove tags --> ' + str(datetime.now())
		corpus = [BeautifulSoup(z).get_text() for z in corpus]
		# keep the quntuation information
		# add space for future tokenization
		print 'format punctuations --> ' + str(datetime.now())
		for c in punctuation:
			corpus = [z.replace(c, ' %s ' % c) for z in corpus]
		corpus = [z.split() for z in corpus]
		return corpus

	train_data = cleanText(train_data)
	test_data = cleanText(test_data)
	unlabeled_data = cleanText(unlabeled_data)

	#pickle.dump

else:
	#picle.load
	pass


def labelizeReviews(reviews, label_type):
	labelized = []
	for i, v in enumerate(reviews):
		label = '%s_%s' % (label_type, i)
		labelized.append(LabeledSentence(v, [label]))
	return labelized


print 'Labelizing Reviews --> ' + str(datetime.now())
train = labelizeReviews(train_data, 'TRAIN')
test = labelizeReviews(test_data, 'TEST')
unsup_reviews = labelizeReviews(unlabeled_data, 'UNSUP')

size = 400

opts = {}
opts['min_count'] = 1
opts['window'] = 10
opts['size'] = size
opts['sample'] = 1e-3
opts['negative'] = 5
opts['workers'] = 3
model_dm = gensim.models.Doc2Vec(**opts)

# opts['dm'] = 0
# model_dbow = gensim.models.Doc2Vec(**opts)

model_dm.build_vocab(np.concatenate((train, test, unsup_reviews)))
all_train_reviews = np.concatenate((train, unsup_reviews))
print 'Training vecs for training set --> ' + str(datetime.now())
for epoch in range(10):
	print 'Epoch --> ' + str(epoch)
	perm = np.random.permutation(all_train_reviews.shape[0])
	model_dm.train(all_train_reviews[perm])


def getVecs(model, corpus, size):
	vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
	return np.concatenate(vecs)


train_vecs_dm = getVecs(model_dm, train, size)

train_vecs = train_vecs_dm

test = np.array(test)

print 'Training vecs for test set --> ' + str(datetime.now())
for epoch in range(10):
	print 'Epoch --> ' + str(epoch)
	perm = np.random.permutation(test.shape[0])
	model_dm.train(test[perm])

test_vecs_dm = getVecs(model_dm, test, size)

test_vecs = test_vecs_dm

pickle.dump(model_dm, file('model_dm', 'w'))

# lr = SGDClassifier(loss='log')
lr = LogisticRegression()

# fpr, tpr, _ = roc_curve(y_test, pred_probs)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.legend(loc='lower right')

# plt.show()
print "20 Fold CV Score: ", np.mean(
    cross_validation.cross_val_score(
    	lr, train_vecs, y, cv=20, scoring='roc_auc'))
print "20 Fold CV Score: ", np.mean(
    cross_validation.cross_val_score(
    	lr, train_vecs, y, cv=20, scoring='accuracy'))

print "Retrain on all training data, predicting test labels...\n"
lr.fit(train_vecs, y)
result = lr.predict_proba(test_vecs)[:, 1]
# result = model.predict(X_test)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

# Use pandas to write the comma-separated output file
output.to_csv('doc2vec_model.csv', index=False, quoting=3)
print "Wrote results to doc2vec_model.csv"
