#!/bin/python
from task2 import supervise, classify


def read_files(tarfname):
	"""Read the training and development data from the sentiment tar file.
	The returned object contains various fields that store sentiment data, such as:

	train_data,dev_data: array of documents (array of words)
	train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
	train_labels,dev_labels: the true string label for each document (same length as data)

	The data is also preprocessed for use with scikit-learn, as:

	count_vec: CountVectorizer used to process the data (for reapplication on new data)
	trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
	le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
	target_labels: List of labels (same order as used in le)
	trainy,devy: array of int labels, one for each document
	"""
	# import tarfile
	# tar = tarfile.open(tarfname, "r:gz")
	trainname = "task2/data/politics/train.tsv"
	devname = "task2/data/politics/test.tsv"
	# for member in tar.getmembers():
	# 	if 'train.tsv' in member.name:
	# 		trainname = member.name
	# 	elif 'dev.tsv' in member.name:
	# 		devname = member.name


	class Data: pass
	sentiment = Data()
	print("-- train data")
	sentiment.train_data, sentiment.train_labels = read_tsv(trainname)
	print(len(sentiment.train_data))

	print("-- dev data")
	sentiment.dev_data, sentiment.dev_labels = read_tsv(devname)
	print(len(sentiment.dev_data))
	print("-- transforming data and labels")
	supervise.feature_engineering(sentiment)
	return sentiment

def read_unlabeled(sentiment):
	"""Reads the unlabeled data.

	The returned object contains three fields that represent the unlabeled data.

	data: documents, represented as sequence of words
	fnames: list of filenames, one for each document
	X: bag of word vector for each document, using the sentiment.vectorizer
	"""
	import tarfile
	tar = tarfile.open(tarfname, "r:gz")
	class Data: pass
	unlabeled = Data()
	unlabeled.data = []

	unlabeledname = "unlabeled.tsv"
	for member in tar.getmembers():
		if 'unlabeled.tsv' in member.name:
			unlabeledname = member.name

	print(unlabeledname)
	tf = tar.extractfile(unlabeledname)
	for line in tf:
		line = line.decode("utf-8")
		text = line.strip()
		unlabeled.data.append(text)

	unlabeled = supervise.process_unlabeled(unlabeled, sentiment)
	print(unlabeled.X.shape)
	tar.close()
	return unlabeled

def read_tsv(fname):
	# member = tar.getmember(fname)
	# print(member.name)
	# tf = tar.extractfile(member)
	data = []
	labels = []
	with open(fname, 'r') as fp:
		for line in fp:
			#line = line.decode("utf-8")
			try:
				(label,text) = line.strip().split("\t")
				labels.append(label)
				data.append(text)
			except:
				print(line)
	return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
	"""Writes the predictions in Kaggle format.

	Given the unlabeled object, classifier, outputfilename, and the sentiment object,
	this function write sthe predictions of the classifier on the unlabeled data and
	writes it to the outputfilename. The sentiment object is required to ensure
	consistent label names.
	"""
	yp = cls.predict(unlabeled.X)
	labels = sentiment.le.inverse_transform(yp)
	f = open(outfname, 'w')
	f.write("ID,LABEL\n")
	for i in range(len(unlabeled.data)):
		f.write(str(i+1))
		f.write(",")
		f.write(labels[i])
		f.write("\n")
	f.close()


def write_gold_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the truth.

	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("ID,LABEL\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(label,review) = line.strip().split("\t")
			i += 1
			f.write(str(i))
			f.write(",")
			f.write(label)
			f.write("\n")
	f.close()

def write_basic_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the naive baseline.

	This baseline predicts POSITIVE for all the instances.
	"""
	f = open(outfname, 'w')
	f.write("ID,LABEL\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(label,review) = line.strip().split("\t")
			i += 1
			f.write(str(i))
			f.write(",")
			f.write("POSITIVE")
			f.write("\n")
	f.close()

if __name__ == "__main__":
	print("Reading data")
	tarfname = "data/sentiment.tar.gz"
	sentiment = read_files(tarfname)
	print("\nTraining classifier")

	print('\nTuning the hyperparameter')
	import numpy as np
	hyperparameters = np.linspace(3.0, 4.0, 10)
	print('\nHyperparameter range:', hyperparameters)
	c = classify.parameter_search(sentiment.trainX, sentiment.trainy, sentiment.devX, sentiment.devy, hyperparameters)
	print('\nBest regularization term', c)
	cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, reg=c)
	print("\nEvaluating")
	classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
	classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
	import pickle
	with open('task2/model/cls.pkl', 'wb') as fp:
		pickle.dump(cls, fp)
	with open('task2/model/sentiment_tfidf.pkl', 'wb') as fp:
		pickle.dump(sentiment.tfidf_vect, fp)
	with open('task2/model/sentiment_dev.pkl', 'wb') as fp:
		pickle.dump(sentiment.dev_data, fp)
	with open('task2/model/sentiment_train.pkl', 'wb') as fp:
		pickle.dump(sentiment.train_data, fp)

	# print("\nReading unlabeled data")
	# unlabeled = read_unlabeled(tarfname, sentiment)
	# print("Writing predictions to a file")
	# write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
	#write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

	# You can't run this since you do not have the true labels
	# print "Writing gold file"
	# write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
