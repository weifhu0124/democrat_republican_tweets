#!/bin/python

def train_classifier(X, y, reg=1):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression(random_state=0, C=reg, solver='lbfgs', max_iter=10000)
	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls, name='data'):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	f1 = metrics.f1_score(yt, yp)
	print(metrics.confusion_matrix(yt, yp))
	print("  Accuracy on %s  is: %s" % (name, acc))
	print("  F1 on %s  is: %s" % (name, f1))
	return acc

def parameter_search(trainX, trainY, devX, devY, regularizations):
	""" search for the best regularization term based on dev set performance

	:param trainX: training text data
	:param trainY: training text label
	:param devX: dev text data
	:param devY: dev text label
	:param regularizations: a list of regularization term to search
	:return: best regularization term
	"""
	best_acc = -1
	best_c = 1
	for c in regularizations:
		cls = train_classifier(trainX, trainY, reg=c)
		acc = evaluate(devX, devY, cls)
		if acc > best_acc:
			best_acc = acc
			best_c = c
	return best_c
